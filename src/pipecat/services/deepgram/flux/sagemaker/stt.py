#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux speech-to-text service for AWS SageMaker (HTTP/2 BiDi transport)."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
)
from pipecat.services.aws.sagemaker.bidi_client import SageMakerBidiClient
from pipecat.services.deepgram.flux.base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
)


@dataclass
class DeepgramFluxSageMakerSTTSettings(DeepgramFluxSTTSettings):
    """Settings for the Deepgram Flux SageMaker STT service.

    Inherits all fields from :class:`DeepgramFluxSTTSettings`.
    """

    pass


class DeepgramFluxSageMakerSTTService(DeepgramFluxSTTBase):
    """Deepgram Flux speech-to-text service for AWS SageMaker.

    Provides real-time speech recognition using Deepgram Flux models deployed on
    AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for low-latency
    transcription with advanced turn detection (StartOfTurn, EndOfTurn,
    EagerEndOfTurn, TurnResumed).

    Unlike the Nova-based SageMaker STT service, Flux handles turn detection
    natively, so no external VAD is needed for turn boundaries. Use
    ``ExternalUserTurnStrategies`` in your pipeline.

    Requirements:

    - AWS credentials configured (via environment variables, AWS CLI, or instance metadata)
    - A deployed SageMaker endpoint with Deepgram Flux model

    Event handlers available:

    - on_connected: Called when the SageMaker session is established
    - on_disconnected: Called when the session is closed
    - on_connection_error: Called on connection failure
    - on_start_of_turn: Deepgram Flux detected start of speech
    - on_end_of_turn: Deepgram Flux detected end of turn
    - on_eager_end_of_turn: Deepgram Flux predicted end of turn
    - on_turn_resumed: User resumed speaking after EagerEndOfTurn
    - on_update: Interim transcript update during a turn

    Example::

        stt = DeepgramFluxSageMakerSTTService(
            endpoint_name="my-deepgram-flux-endpoint",
            region="us-east-2",
            settings=DeepgramFluxSageMakerSTTService.Settings(
                model="flux-general-en",
                eot_threshold=0.7,
                eager_eot_threshold=0.5,
            ),
        )
    """

    Settings = DeepgramFluxSageMakerSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str,
        encoding: str = "linear16",
        sample_rate: int | None = None,
        mip_opt_out: bool | None = None,
        tag: list | None = None,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Deepgram Flux SageMaker STT service.

        Args:
            endpoint_name: Name of the SageMaker endpoint with Deepgram Flux model
                deployed (e.g., "my-deepgram-flux-endpoint").
            region: AWS region where the endpoint is deployed (e.g., "us-east-2").
            encoding: Audio encoding format. Defaults to "linear16".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            mip_opt_out: Opt out of Deepgram model improvement program.
            tag: Tags to label requests for identification during usage reporting.
            should_interrupt: Whether to interrupt the bot when Flux detects that
                the user is speaking. Defaults to True.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        # Initialize default settings
        default_settings = self.Settings(
            model="flux-general-en",
            language=None,
            eager_eot_threshold=None,
            eot_threshold=None,
            eot_timeout_ms=None,
            keyterm=[],
            min_confidence=None,
            language_hints=None,
        )

        # Apply settings delta
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            encoding=encoding,
            mip_opt_out=mip_opt_out,
            tag=tag,
            should_interrupt=should_interrupt,
            settings=default_settings,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region

        self._client: SageMakerBidiClient | None = None
        self._response_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Transport interface implementation
    # ------------------------------------------------------------------

    async def _transport_send_audio(self, audio: bytes):
        await self._client.send_audio_chunk(audio)

    async def _transport_send_json(self, message: dict):
        await self._client.send_json(message)

    def _transport_is_active(self) -> bool:
        return self._client is not None and self._client.is_active

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to the SageMaker endpoint and start the BiDi session.

        Starts the HTTP/2 session and waits for the Flux ``Connected`` message
        before returning, ensuring audio is not sent before the model is ready.
        """
        logger.debug("Connecting to Deepgram Flux on SageMaker...")

        self._connection_established_event.clear()

        self._client = SageMakerBidiClient(
            endpoint_name=self._endpoint_name,
            region=self._region,
            model_invocation_path="v2/listen",
            model_query_string=self._build_query_string(),
        )

        try:
            await self._client.start_session()

            # Start response processor first so we can receive the Connected message
            self._response_task = self.create_task(self._process_responses())

            # Wait for Flux to confirm the connection is ready
            logger.debug("SageMaker session started, waiting for Flux connection confirmation...")
            await self._connection_established_event.wait()

            # Note: Flux does not support KeepAlive messages (only CloseStream and
            # Configure are valid). The watchdog task handles keeping the connection
            # alive by sending silence when needed.
            self._watchdog_task = self.create_task(self._watchdog_task_handler())

            logger.debug("Connected to Deepgram Flux on SageMaker")
            await self._call_event_handler("on_connected")

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect(self):
        """Disconnect from the SageMaker endpoint."""
        self._connection_established_event.clear()

        if self._client and self._client.is_active:
            logger.debug("Disconnecting from Deepgram Flux on SageMaker...")

            await self._send_close_stream()

            if self._watchdog_task and not self._watchdog_task.done():
                await self.cancel_task(self._watchdog_task)
                self._watchdog_task = None
                self._last_stt_time = None

            if self._response_task and not self._response_task.done():
                await self.cancel_task(self._response_task)

            await self._client.close_session()

            logger.debug("Disconnected from Deepgram Flux on SageMaker")
            await self._call_event_handler("on_disconnected")

    # ------------------------------------------------------------------
    # Audio sending and response receiving
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio data to Deepgram Flux for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via BiDi stream callbacks).
        """
        if not self._connection_established_event.is_set():
            return

        if self._client and self._client.is_active:
            try:
                self._last_stt_time = time.monotonic()
                await self._client.send_audio_chunk(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
        yield None

    async def _process_responses(self):
        """Process streaming responses from Deepgram Flux on SageMaker."""
        try:
            while self._client and self._client.is_active:
                result = await self._client.receive_response()

                if result is None:
                    break

                if hasattr(result, "value") and hasattr(result.value, "bytes_"):
                    if result.value.bytes_:
                        response_data = result.value.bytes_.decode("utf-8")

                        try:
                            parsed = json.loads(response_data)
                            await self._handle_message(parsed)
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON response: {response_data}")

        except asyncio.CancelledError:
            logger.debug("Response processor cancelled")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            logger.debug("Response processor stopped")
