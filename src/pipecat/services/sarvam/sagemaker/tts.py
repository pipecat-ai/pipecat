#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam text-to-speech service for AWS SageMaker.

This module connects to Sarvam Bulbul models deployed behind SageMaker's
HTTP/2 bidirectional streaming endpoint. Text, audio, and control messages use
Sarvam's UTF-8 JSON streaming protocol.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from loguru import logger

try:
    from aws_sdk_sagemaker_runtime_http2.models import ResponseStreamEventPayloadPart

    from pipecat.services.sarvam.tts import (
        TTS_MODEL_CONFIGS,
        SarvamTTSSettings,
        language_to_sarvam_language,
    )
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Sarvam on SageMaker, you need to `uv add "pipecat-ai[sarvam,sagemaker]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.aws.sagemaker.bidi_client import (
    SageMakerBidiClient,
    SageMakerBidiSessionError,
    classify_sagemaker_bidi_error,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import assert_given, is_given

_MAX_PROVIDER_PAYLOAD_BYTES = 4 * 1024 * 1024
_MAX_ERROR_MESSAGE_CHARS = 512


def _bounded_error_message(value: object) -> str:
    """Return a bounded, single-line provider error message."""
    return " ".join(str(value).split())[:_MAX_ERROR_MESSAGE_CHARS]


@dataclass
class SarvamSageMakerTTSSettings(SarvamTTSSettings):
    """Settings for :class:`SarvamSageMakerTTSService`.

    Parameters:
        model: Fixed to ``bulbul:v3`` for the published Marketplace package.
        voice: Bulbul v3 speaker name, such as ``shubh`` or ``priya``.
        language: Output language locale, such as :attr:`Language.EN_IN`.
        enable_preprocessing: Whether Bulbul normalizes numbers, dates, and
            English words. Bulbul v3 always enables this option.
        pace: Speaking-rate multiplier from 0.5 to 2.0.

    The inherited ``min_buffer_size`` and ``max_chunk_length`` settings belong
    to Sarvam's managed WebSocket API and are not sent to SageMaker.
    """


class SarvamSageMakerTTSService(InterruptibleTTSService):
    """Sarvam Bulbul text-to-speech service using SageMaker bidirectional streaming.

    The service uses AWS credentials for authentication and does not require a
    Sarvam API key. Sarvam does not document an in-band cancellation message, so
    interruptions close and reopen the SageMaker session.

    Example::

        tts = SarvamSageMakerTTSService(
            endpoint_name="sarvam-bulbul",
            region="ap-south-1",
            settings=SarvamSageMakerTTSService.Settings(
                model="bulbul:v3",
                voice="shubh",
                language=Language.EN_IN,
            ),
        )
    """

    Settings = SarvamSageMakerTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str,
        sample_rate: int | None = None,
        output_audio_codec: str = "linear16",
        output_audio_bitrate: str = "128k",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Sarvam SageMaker TTS service.

        Args:
            endpoint_name: Name of the SageMaker endpoint hosting Bulbul.
            region: AWS region containing the endpoint.
            sample_rate: Output audio sample rate. Must be 8000, 16000, 22050,
                or 24000. Defaults to 24000.
            output_audio_codec: Audio codec requested from Sarvam.
                ``linear16`` is recommended for Pipecat audio frames.
            output_audio_bitrate: Audio bitrate requested from Sarvam.
            settings: Runtime-updatable Bulbul settings.
            **kwargs: Arguments forwarded to :class:`InterruptibleTTSService`.
        """
        default_settings = self.Settings(
            model="bulbul:v3",
            voice="shubh",
            language="en-IN",
            enable_preprocessing=False,
            min_buffer_size=50,
            max_chunk_length=150,
            pace=1.0,
            pitch=None,
            loudness=None,
            temperature=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        model = assert_given(default_settings.model)
        if model != "bulbul:v3":
            raise ValueError(
                f"Unsupported Sarvam SageMaker model '{model}'. Allowed value: bulbul:v3."
            )

        self._config = TTS_MODEL_CONFIGS[model]
        self._apply_model_constraints(default_settings)
        effective_sample_rate = sample_rate or self._config.default_sample_rate
        if effective_sample_rate not in {8000, 16000, 22050, 24000}:
            raise ValueError(
                "Sarvam SageMaker TTS sample_rate must be 8000, 16000, 22050, or 24000."
            )
        if output_audio_codec.lower() == "flac":
            raise ValueError("Sarvam SageMaker bidirectional TTS does not support FLAC.")

        super().__init__(
            sample_rate=effective_sample_rate,
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region
        self._speech_sample_rate = str(effective_sample_rate)
        self._output_audio_codec = output_audio_codec.lower()
        self._output_audio_bitrate = output_audio_bitrate

        self._connect_lock = asyncio.Lock()
        self._client: SageMakerBidiClient | None = None
        self._response_task: asyncio.Task | None = None

    def _classify_error(self, exception: Exception) -> ErrorCategory | None:
        """Classify SageMaker session failures."""
        return classify_sagemaker_bidi_error(exception)

    def can_generate_metrics(self) -> bool:
        """Return whether the service can generate usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Pipecat language to Sarvam's locale format."""
        return language_to_sarvam_language(language)

    @staticmethod
    def _apply_model_constraints(settings: SarvamSageMakerTTSSettings):
        model = assert_given(settings.model)
        assert model is not None
        config = TTS_MODEL_CONFIGS[model]

        pace = assert_given(settings.pace)
        if pace is not None:
            pace_min, pace_max = config.pace_range
            if pace < pace_min or pace > pace_max:
                logger.warning(
                    f"Pace {pace} is outside model range ({pace_min}-{pace_max}). Clamping."
                )
                settings.pace = max(pace_min, min(pace_max, pace))

        if config.preprocessing_always_enabled:
            settings.enable_preprocessing = True
        if not config.supports_pitch:
            settings.pitch = None
        if not config.supports_loudness:
            settings.loudness = None
        if not config.supports_temperature:
            settings.temperature = None

    def _build_query_string(self) -> str:
        """Build the URL-encoded model connection query string."""
        return urlencode({"model": str(assert_given(self._settings.model))})

    def _config_data(self) -> dict[str, Any]:
        language = self._settings.language
        if isinstance(language, Language):
            language = language_to_sarvam_language(language)

        data: dict[str, Any] = {
            "language_code": language,
            "speaker": self._settings.voice,
            "speech_sample_rate": self._speech_sample_rate,
            "enable_preprocessing": self._settings.enable_preprocessing,
            "output_audio_codec": self._output_audio_codec,
            "output_audio_bitrate": self._output_audio_bitrate,
            "pace": self._settings.pace,
            "model": self._settings.model,
        }
        if self._config.supports_pitch and self._settings.pitch is not None:
            data["pitch"] = self._settings.pitch
        if self._config.supports_loudness and self._settings.loudness is not None:
            data["loudness"] = self._settings.loudness
        if self._config.supports_temperature and self._settings.temperature is not None:
            data["temperature"] = self._settings.temperature
        return data

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect to SageMaker."""
        await super().setup(setup)
        self._speech_sample_rate = str(self.sample_rate)
        await self._connect()

    async def _connect(self):
        """Open the SageMaker session and start response processing."""
        async with self._connect_lock:
            await super()._connect()
            if not self.is_usable:
                return
            await self._connect_websocket()
            if self._client and self._client.is_active and not self._response_task:
                self._response_task = self.create_task(self._process_responses())

    async def _disconnect(self):
        """Close the SageMaker session and response task."""
        await super()._disconnect()

        task = self._response_task
        self._response_task = None
        if task and task is not asyncio.current_task() and not task.done():
            await self.cancel_task(task)

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the transport used by the websocket-compatible TTS base."""
        if self._client and self._client.is_active:
            return

        client = SageMakerBidiClient(
            endpoint_name=self._endpoint_name,
            region=self._region,
            model_query_string=self._build_query_string(),
        )
        self._client = client
        try:
            await client.start_session()
            await self._send_config()
            await self._call_event_handler("on_connected")
        except Exception as e:
            if client.is_active:
                await client.close_session()
            if self._client is client:
                self._client = None
            await self.push_error(
                error_msg=f"Failed to connect to Sarvam TTS on SageMaker: {e}",
                exception=e,
                force_treat_as_permanent=isinstance(e, SageMakerBidiSessionError),
            )
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect_websocket(self):
        """Close the SageMaker transport used by the websocket-compatible base."""
        client = self._client
        self._client = None
        if not client:
            return

        if client.is_active:
            await client.close_session()
            await self._call_event_handler("on_disconnected")

    async def _verify_connection(self) -> bool:
        return bool(self._client and self._client.is_active)

    async def _receive_messages(self):
        """Receive messages for the websocket-compatible base interface."""
        await self._process_responses()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply settings, reconnecting when the model query changes."""
        if is_given(delta.model):
            model = delta.model
            if model != "bulbul:v3":
                raise ValueError(
                    f"Unsupported Sarvam SageMaker model '{model}'. Allowed value: bulbul:v3."
                )

        changed = await super()._update_settings(delta)
        if not changed:
            return changed

        if "model" in changed:
            model = assert_given(self._settings.model)
            assert model is not None
            self._config = TTS_MODEL_CONFIGS[model]
            if not is_given(delta.voice):
                self._settings.voice = self._config.default_speaker
            self._apply_model_constraints(self._settings)
            await self._disconnect()
            await self._connect()
        elif self._client and self._client.is_active:
            self._apply_model_constraints(self._settings)
            await self._send_config()

        return changed

    async def _send_config(self):
        """Send Sarvam's initial configuration message as UTF-8 JSON."""
        if not self._client or not self._client.is_active:
            raise RuntimeError("SageMaker BiDi session not active")
        await self._client.send_json({"type": "config", "data": self._config_data()})

    async def flush_audio(self, context_id: str | None = None):
        """Flush buffered Sarvam text for the active audio context."""
        if self._client and self._client.is_active:
            try:
                await self._client.send_json({"type": "flush"})
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error sending flush to Sarvam TTS on SageMaker: {e}",
                    exception=e,
                )

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to Bulbul; audio arrives on the response task."""
        if not self._client or not self._client.is_active:
            await self._connect()

        if not self._client or not self._client.is_active:
            yield ErrorFrame(error=f"{self} is not connected")
            yield TTSStoppedFrame(context_id=context_id)
            return

        try:
            await self._client.send_json({"type": "text", "data": {"text": text}})
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Sarvam SageMaker TTS error: {e}", exception=e)
            yield TTSStoppedFrame(context_id=context_id)
            await self._disconnect()
            await self._connect()

    async def _process_responses(self):
        """Receive and process UTF-8 JSON responses from Bulbul."""
        try:
            while self._client and self._client.is_active:
                result = await self._client.receive_response()
                if result is None:
                    break
                if not isinstance(result, ResponseStreamEventPayloadPart):
                    continue

                payload = result.value.bytes_
                if not payload:
                    continue
                if len(payload) > _MAX_PROVIDER_PAYLOAD_BYTES:
                    await self.push_error(
                        error_msg="Sarvam TTS response exceeds the 4 MiB safety limit"
                    )
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    await self.push_error(
                        error_msg=f"Invalid Sarvam TTS response: {e}",
                        exception=e,
                    )
                    continue

                if not isinstance(message, dict):
                    await self.push_error(error_msg="Invalid Sarvam TTS response structure")
                    continue
                if "data" in message and not isinstance(message["data"], dict):
                    await self.push_error(error_msg="Invalid Sarvam TTS data structure")
                    continue

                await self._handle_message(message)
        except asyncio.CancelledError:
            logger.debug(f"{self}: response processor cancelled")
        except Exception as e:
            await self.push_error(
                error_msg=f"Sarvam SageMaker TTS receive error: {e}",
                exception=e,
            )
        finally:
            logger.debug(f"{self}: response processor stopped")

    async def _handle_message(self, message: dict[str, Any]):
        message_type = message.get("type")
        data = message.get("data") or {}
        context_id = self.get_active_audio_context_id()

        if message_type == "audio":
            try:
                audio = base64.b64decode(data["audio"], validate=True)
            except (KeyError, TypeError, ValueError) as e:
                await self.push_error(
                    error_msg=f"Invalid Sarvam TTS audio response: {e}",
                    exception=e,
                )
                return

            await self.stop_ttfb_metrics()
            await self.append_to_audio_context(
                context_id,
                TTSAudioRawFrame(
                    audio,
                    self.sample_rate or int(self._speech_sample_rate),
                    1,
                    context_id=context_id,
                ),
            )
            return

        if message_type == "event" and data.get("event_type") == "final":
            if context_id and self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id,
                    TTSStoppedFrame(context_id=context_id),
                )
                await self.remove_audio_context(context_id)
            return

        if message_type == "error":
            error_message = _bounded_error_message(
                data.get("message") or "provider returned an error"
            )
            error = ErrorFrame(error=f"TTS Error: {error_message}")
            await self.push_error(error_msg=error.error)
            await self.append_to_audio_context(context_id, error)
            return

        logger.debug(f"{self}: received unknown Sarvam TTS message type: {message_type!r}")
