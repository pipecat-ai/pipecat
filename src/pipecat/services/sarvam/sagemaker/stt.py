#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam speech-to-text service for AWS SageMaker.

This module connects to Sarvam Saaras models deployed behind SageMaker's
HTTP/2 bidirectional streaming endpoint. Audio and control messages use
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

    from pipecat.services.sarvam.stt import (
        MODEL_CONFIGS,
        SarvamSTTSettings,
        language_to_sarvam_language,
    )
except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Sarvam on SageMaker, you need to `uv add "pipecat-ai[sarvam,sagemaker]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.aws.sagemaker.bidi_client import (
    SageMakerBidiClient,
    SageMakerBidiSessionError,
    classify_sagemaker_bidi_error,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import SARVAM_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import assert_given, is_given

_MAX_PROVIDER_PAYLOAD_BYTES = 4 * 1024 * 1024
_MAX_ERROR_MESSAGE_CHARS = 512


def _bounded_error_message(value: object) -> str:
    """Return a bounded, single-line provider error message."""
    return " ".join(str(value).split())[:_MAX_ERROR_MESSAGE_CHARS]


@dataclass
class SarvamSageMakerSTTSettings(SarvamSTTSettings):
    """Settings for :class:`SarvamSageMakerSTTService`.

    Parameters:
        model: Fixed to ``saaras:v3`` for the published Marketplace package.
        language: Input language locale, such as :attr:`Language.EN_IN`.
        vad_signals: Whether Saaras emits provider VAD events. When false,
            Pipecat flushes the stream after its local VAD detects turn end.

    Other inherited Sarvam VAD settings are not part of the published
    SageMaker protocol and are ignored during runtime updates.
    """


class SarvamSageMakerSTTService(STTService):
    """Sarvam Saaras speech-to-text service using SageMaker bidirectional streaming.

    The service uses AWS credentials for authentication and does not require a
    Sarvam API key.

    Example::

        stt = SarvamSageMakerSTTService(
            endpoint_name="sarvam-saaras",
            region="ap-south-1",
            settings=SarvamSageMakerSTTService.Settings(
                model="saaras:v3",
                language=Language.EN_IN,
            ),
        )
    """

    Settings = SarvamSageMakerSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str,
        sample_rate: int | None = None,
        input_audio_codec: str = "wav",
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = SARVAM_TTFS_P99,
        keepalive_timeout: float | None = None,
        keepalive_interval: float = 5.0,
        **kwargs,
    ):
        """Initialize the Sarvam SageMaker STT service.

        Args:
            endpoint_name: Name of the SageMaker endpoint hosting Saaras.
            region: AWS region containing the endpoint.
            sample_rate: Input audio sample rate in Hz. Must be 8000 or 16000.
                Defaults to 16000.
            input_audio_codec: Input audio codec sent to Sarvam. Must be ``wav``.
            settings: Runtime-updatable Saaras settings.
            ttfs_p99_latency: Expected p99 time-to-final-segment latency in seconds.
            keepalive_timeout: Seconds of inactivity before sending silent audio.
            keepalive_interval: Seconds between keepalive checks.
            **kwargs: Arguments forwarded to :class:`STTService`.
        """
        default_settings = self.Settings(
            model="saaras:v3",
            language=None,
            vad_signals=False,
            high_vad_sensitivity=None,
            positive_speech_threshold=None,
            negative_speech_threshold=None,
            min_speech_frames=None,
            first_turn_min_speech_frames=None,
            negative_frames_count=None,
            negative_frames_window=None,
            start_speech_volume_threshold=None,
            interrupt_min_speech_frames=None,
            pre_speech_pad_frames=None,
            num_initial_ignored_frames=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        model = assert_given(default_settings.model)
        if model != "saaras:v3":
            raise ValueError(
                f"Unsupported Sarvam SageMaker model '{model}'. Allowed value: saaras:v3."
            )

        self._config = MODEL_CONFIGS[model]
        if default_settings.language is not None and not self._config.supports_language:
            raise ValueError(f"Model '{model}' does not support language parameter.")
        if input_audio_codec.removeprefix("audio/") != "wav":
            raise ValueError("Sarvam SageMaker STT supports only WAV input audio.")

        effective_sample_rate = sample_rate or 16000
        if effective_sample_rate not in {8000, 16000}:
            raise ValueError("Sarvam SageMaker STT sample_rate must be 8000 or 16000.")
        super().__init__(
            sample_rate=effective_sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=keepalive_timeout,
            keepalive_interval=keepalive_interval,
            settings=default_settings,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region
        self._input_sample_rate = effective_sample_rate

        self._connect_lock = asyncio.Lock()
        self._client: SageMakerBidiClient | None = None
        self._response_task: asyncio.Task | None = None
        self._warned_no_session = False
        self._unmapped_language_codes_warned: set[str] = set()

        self._register_event_handler("on_speech_started")
        self._register_event_handler("on_speech_stopped")
        self._register_event_handler("on_utterance_end")

    def _classify_error(self, exception: Exception) -> ErrorCategory | None:
        """Classify SageMaker session failures."""
        return classify_sagemaker_bidi_error(exception)

    def can_generate_metrics(self) -> bool:
        """Return whether the service can generate usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Pipecat language to Sarvam's locale format."""
        return language_to_sarvam_language(language)

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Describe provider-side turn detection when Sarvam VAD is enabled."""
        frame = super().service_metadata_frame()
        if self._settings.vad_signals:
            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    def _effective_sample_rate(self) -> int:
        return self.sample_rate or self._input_sample_rate

    def _get_language_string(self) -> str | None:
        language = self._settings.language
        if isinstance(language, Language):
            return language_to_sarvam_language(language)
        if isinstance(language, str):
            return language
        return self._config.default_language

    def _build_query_string(self) -> str:
        """Build the URL-encoded model connection query string."""
        params: dict[str, str] = {
            "model": str(assert_given(self._settings.model)),
            "sample_rate": str(self._effective_sample_rate()),
        }

        language = self._get_language_string()
        if language is not None:
            params["language-code"] = language
        if self._settings.vad_signals is not None:
            params["vad_signals"] = str(self._settings.vad_signals).lower()

        return urlencode(params)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect to SageMaker."""
        await super().setup(setup)
        self._input_sample_rate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the SageMaker session."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the SageMaker session."""
        await super().cancel(frame)
        await self._disconnect()

    async def cleanup(self):
        """Release the SageMaker session and background tasks."""
        await super().cleanup()
        await self._disconnect()

    async def _connect(self):
        """Open a SageMaker bidirectional stream and send Sarvam configuration."""
        async with self._connect_lock:
            if self._client and self._client.is_active:
                return
            if not self.is_usable:
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
                self._response_task = self.create_task(self._process_responses())
                self._create_keepalive_task()
                self._warned_no_session = False
                await self._call_event_handler("on_connected")
            except Exception as e:
                if client.is_active:
                    await client.close_session()
                if self._client is client:
                    self._client = None
                await self.push_error(
                    error_msg=f"Failed to connect to Sarvam on SageMaker: {e}",
                    exception=e,
                    force_treat_as_permanent=isinstance(e, SageMakerBidiSessionError),
                )
                await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect(self):
        """Close the SageMaker session. Safe to call multiple times."""
        await self._cancel_keepalive_task()

        task = self._response_task
        self._response_task = None
        if task and task is not asyncio.current_task() and not task.done():
            await self.cancel_task(task)

        client = self._client
        self._client = None
        if client and client.is_active:
            await client.close_session()
            await self._call_event_handler("on_disconnected")

    async def _do_reconnect(self):
        await self._disconnect()
        await self._connect()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply settings and reconnect for supported session parameters."""
        if is_given(delta.model):
            model = delta.model
            if model != "saaras:v3":
                raise ValueError(
                    f"Unsupported Sarvam SageMaker model '{model}'. Allowed value: saaras:v3."
                )

        changed = await super()._update_settings(delta)
        if not changed:
            return changed

        reconnect_fields = {"model", "language", "vad_signals"}
        if changed.keys() & reconnect_fields:
            await self._disconnect()
            await self._connect()

        unhandled = {name for name in changed if name not in reconnect_fields}
        self._warn_unhandled_updated_settings(unhandled)
        return changed

    async def _send_config(self):
        """Send Sarvam's initial configuration message as UTF-8 JSON."""
        if not self._client or not self._client.is_active:
            raise RuntimeError("SageMaker BiDi session not active")
        await self._client.send_json({"type": "config"})

    def _audio_message(self, audio: bytes) -> dict[str, Any]:
        return {
            "audio": {
                "data": base64.b64encode(audio).decode("utf-8"),
                "encoding": "audio/wav",
                "sample_rate": self._effective_sample_rate(),
            },
        }

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Encode and send a raw PCM audio chunk to Saaras.

        Args:
            audio: Raw mono 16-bit PCM bytes at the configured sample rate.

        Yields:
            None after the chunk is sent. Transcripts arrive on the response task.
        """
        if self._client and self._client.is_active:
            try:
                await self._client.send_json(self._audio_message(audio))
            except Exception as e:
                yield ErrorFrame(error=f"Sarvam SageMaker STT error: {e}", exception=e)
        elif not self._warned_no_session:
            self._warned_no_session = True
            logger.warning(f"{self}: no active SageMaker session, audio is being dropped")
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and flush Saaras when local VAD ends an utterance."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame) and not self._settings.vad_signals:
            if self._client and self._client.is_active:
                try:
                    await self._client.send_json({"type": "flush"})
                except Exception as e:
                    await self.push_error(
                        error_msg=f"Error sending flush to Sarvam on SageMaker: {e}",
                        exception=e,
                    )

    def _is_keepalive_ready(self) -> bool:
        return bool(self._client and self._client.is_active)

    async def _send_keepalive(self, silence: bytes):
        if not self._client:
            return
        await self._client.send_json(self._audio_message(silence))

    async def _process_responses(self):
        """Receive and process UTF-8 JSON responses from Saaras."""
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
                        error_msg="Sarvam STT response exceeds the 4 MiB safety limit"
                    )
                    continue
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    await self.push_error(
                        error_msg=f"Invalid Sarvam STT response: {e}",
                        exception=e,
                    )
                    continue

                if not isinstance(message, dict):
                    await self.push_error(error_msg="Invalid Sarvam STT response structure")
                    continue
                if "data" in message and not isinstance(message["data"], dict):
                    await self.push_error(error_msg="Invalid Sarvam STT data structure")
                    continue

                await self._handle_message(message)
        except asyncio.CancelledError:
            logger.debug(f"{self}: response processor cancelled")
        except Exception as e:
            await self.push_error(
                error_msg=f"Sarvam SageMaker STT receive error: {e}",
                exception=e,
            )
        finally:
            logger.debug(f"{self}: response processor stopped")

    async def _handle_message(self, message: dict[str, Any]):
        message_type = message.get("type")
        data = message.get("data") or {}

        if message_type == "events":
            signal = data.get("signal_type")
            if signal == "START_SPEECH":
                await self._call_event_handler("on_speech_started")
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
            elif signal == "END_SPEECH":
                await self._call_event_handler("on_speech_stopped")
                await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
            return

        if message_type == "error":
            error_message = _bounded_error_message(
                data.get("message") or "provider returned an error"
            )
            await self.push_error(error_msg=f"Sarvam STT error: {error_message}")
            return

        if message_type != "data":
            logger.debug(f"{self}: received unknown Sarvam STT message type: {message_type!r}")
            return

        transcript = data.get("transcript", "")
        if not transcript or not transcript.strip():
            return

        language = self._map_language_code_to_enum(
            data.get("language_code") or self._get_language_string()
        )
        await self._call_event_handler("on_utterance_end")
        await self._handle_transcription(transcript, True, language)
        await self.emit_stt_usage_metrics()
        await self.push_frame(
            TranscriptionFrame(
                transcript,
                self._user_id,
                time_now_iso8601(),
                language,
                result=message,
                finalized=True,
            )
        )

    def _map_language_code_to_enum(self, language_code: str | None) -> Language | None:
        if not language_code or language_code == "unknown":
            return None
        if language_code == "od-IN":
            return Language.OR_IN
        try:
            return Language(language_code)
        except ValueError:
            if language_code not in self._unmapped_language_codes_warned:
                self._unmapped_language_codes_warned.add(language_code)
                logger.warning(
                    f"{self} received unmapped Sarvam language code {language_code!r}; "
                    "leaving the transcription language unset"
                )
            return None

    @traced_stt
    async def _handle_transcription(
        self,
        transcript: str,
        is_final: bool,
        language: Language | None = None,
    ):
        """Record a Saaras transcription for tracing."""
        pass
