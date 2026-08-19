#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux text-to-speech service for AWS SageMaker (HTTP/2 BiDi transport)."""

import asyncio
import json
from dataclasses import dataclass

from loguru import logger

try:
    from aws_sdk_sagemaker_runtime_http2.models import ResponseStreamEventPayloadPart
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Deepgram Flux on SageMaker, you need to `uv add "pipecat-ai[sagemaker]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

from pipecat.services.aws.sagemaker.bidi_client import SageMakerBidiClient
from pipecat.services.deepgram.flux.tts_base import (
    DeepgramFluxTTSBase,
    DeepgramFluxTTSSettings,
)
from pipecat.services.tts_service import TextAggregationMode


@dataclass
class DeepgramFluxSageMakerTTSSettings(DeepgramFluxTTSSettings):
    """Settings for the Deepgram Flux SageMaker TTS service.

    Inherits all fields from :class:`DeepgramFluxTTSSettings`.
    """

    pass


class DeepgramFluxSageMakerTTSService(DeepgramFluxTTSBase):
    """Deepgram Flux text-to-speech service for AWS SageMaker.

    Provides real-time speech synthesis using Deepgram Flux voices deployed on
    AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for low-latency
    audio generation, with the same Flux turn protocol as the hosted API: a
    single session keeps acoustic state across turns, and a barge-in cancels the
    active turn with ``Interrupt`` rather than reconnecting.

    Requirements:

    - AWS credentials configured (via environment variables, AWS CLI, or instance metadata)
    - A deployed SageMaker endpoint with a Deepgram Flux TTS model:
      https://developers.deepgram.com/docs/deploy-amazon-sagemaker
    - ``pipecat-ai[sagemaker]`` installed

    Event handlers:

    - on_connected: Called when the SageMaker session is established.
    - on_disconnected: Called when the session is closed.
    - on_connection_error: Called on connection failure.

    Example::

        tts = DeepgramFluxSageMakerTTSService(
            endpoint_name="my-deepgram-flux-tts-endpoint",
            region="us-east-2",
            settings=DeepgramFluxSageMakerTTSService.Settings(
                voice="flux-alexis-en",
                speed=1.05,
            ),
        )
    """

    Settings = DeepgramFluxSageMakerTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str,
        sample_rate: int | None = None,
        mip_opt_out: bool | None = None,
        tag: list[str] | None = None,
        text_aggregation_mode: TextAggregationMode = TextAggregationMode.TOKEN,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Deepgram Flux SageMaker TTS service.

        Args:
            endpoint_name: Name of the SageMaker endpoint with a Deepgram Flux TTS
                model deployed (e.g., "my-deepgram-flux-tts-endpoint").
            region: AWS region where the endpoint is deployed (e.g., "us-east-2").
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                default. Must be one of :attr:`SUPPORTED_SAMPLE_RATES`.
            mip_opt_out: Opt out of the Deepgram Model Improvement Program. See
                https://dpgr.am/deepgram-mip for pricing impacts before setting to True.
            tag: Tags to label requests for identification during usage reporting.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
                Defaults to ``TextAggregationMode.TOKEN``, streaming LLM tokens
                straight to Flux for the lowest latency.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        default_settings = self.Settings(
            model=None,
            voice="flux-heather-en",
            language=None,
            speed=None,
            expressivity=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            mip_opt_out=mip_opt_out,
            tag=tag,
            text_aggregation_mode=text_aggregation_mode,
            settings=default_settings,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region

        self._client: SageMakerBidiClient | None = None
        self._response_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Transport interface implementation
    # ------------------------------------------------------------------

    async def _transport_send_json(self, message: dict):
        if (
            self._client is None
        ):  # should never happen — caller should gate on _transport_is_active()
            return
        await self._client.send_json(message)

    def _transport_is_active(self) -> bool:
        return self._client is not None and self._client.is_active

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to the SageMaker endpoint and start the BiDi session."""
        logger.debug("Connecting to Deepgram Flux TTS on SageMaker...")

        self._validate_sample_rate()

        self._client = SageMakerBidiClient(
            endpoint_name=self._endpoint_name,
            region=self._region,
            model_invocation_path="v2/speak",
            model_query_string=self._build_query_string(),
        )

        try:
            await self._client.start_session()

            self._response_task = self.create_task(self._process_responses())

            logger.debug("Connected to Deepgram Flux TTS on SageMaker")
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect(self):
        """Disconnect from the SageMaker endpoint."""
        if not self._client:
            return

        logger.debug("Disconnecting from Deepgram Flux TTS on SageMaker...")

        await self.stop_all_metrics()

        if self._response_task:
            await self.cancel_task(self._response_task)
            self._response_task = None

        # No `Close` message here: in Flux, `Close` asks the server to drain the
        # active turn, generating all of its remaining audio, which a teardown
        # has no use for. Closing the session ends it outright.
        await self._client.close_session()
        self._client = None

        logger.debug("Disconnected from Deepgram Flux TTS on SageMaker")
        await self._call_event_handler("on_disconnected")

    # ------------------------------------------------------------------
    # Response receiving
    # ------------------------------------------------------------------

    async def _process_responses(self):
        """Process streaming responses from Deepgram Flux TTS on SageMaker."""
        try:
            while self._client and self._client.is_active:
                result = await self._client.receive_response()

                if result is None:
                    break

                if isinstance(result, ResponseStreamEventPayloadPart):
                    payload = result.value.bytes_
                    if not payload:
                        continue

                    # Endpoints that label their payload parts set a data
                    # type; where it is absent, only audio fails to parse as
                    # JSON.
                    data_type = result.value.data_type
                    if data_type == "BINARY":
                        await self._handle_audio(payload)
                        continue

                    try:
                        message = json.loads(payload.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        if data_type == "UTF8":
                            logger.error(f"Invalid JSON message: {payload}")
                        else:
                            await self._handle_audio(payload)
                        continue

                    await self._handle_message(message)

        except asyncio.CancelledError:
            logger.debug("TTS response processor cancelled")
            raise
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            logger.debug("TTS response processor stopped")
