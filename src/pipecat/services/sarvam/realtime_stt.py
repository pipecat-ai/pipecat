#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam realtime Speech-to-Text service implementation."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given, is_given
from pipecat.services.stt_latency import SARVAM_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

REALTIME_STT_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
REALTIME_MODEL = "saaras:v3-realtime"

SUPPORTED_LANGUAGES = {
    "auto",
    "en-IN",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "or-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "gu-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
}
SUPPORTED_STREAM_TYPES = {"fast", "balanced", "simulated"}
SUPPORTED_ENDPOINTING = {"vad", "manual"}
SUPPORTED_ENCODINGS = {"linear16", "linear32", "mulaw", "alaw"}
SUPPORTED_SAMPLE_RATES = {8000, 16000}
SUPPORTED_MODES = {"transcribe", "translate", "verbatim", "translit", "codemix"}
_BYTES_PER_SAMPLE = {"linear16": 2, "linear32": 4, "mulaw": 1, "alaw": 1}
_SHORT_LANGUAGE_DEFAULTS = {
    language.split("-", maxsplit=1)[0]: language
    for language in SUPPORTED_LANGUAGES
    if language != "auto"
}


def language_to_sarvam_realtime_language(language: Language) -> str:
    """Convert a Language enum to Sarvam realtime's language code."""
    language_map = {
        Language.AS_IN: "as-IN",
        Language.BN_IN: "bn-IN",
        Language.EN_IN: "en-IN",
        Language.GU_IN: "gu-IN",
        Language.HI_IN: "hi-IN",
        Language.KN_IN: "kn-IN",
        Language.KOK_IN: "kok-IN",
        Language.MAI_IN: "mai-IN",
        Language.ML_IN: "ml-IN",
        Language.MR_IN: "mr-IN",
        Language.OR_IN: "or-IN",
        Language.PA_IN: "pa-IN",
        Language.SD_IN: "sd-IN",
        Language.TA_IN: "ta-IN",
        Language.TE_IN: "te-IN",
    }
    return resolve_language(language, language_map, use_base_code=False)


class SarvamRealtimeSTTError(Exception):
    """Error raised for fatal Sarvam realtime STT payloads."""

    def __init__(self, payload: dict[str, Any]):
        """Initialize the error from a raw Sarvam payload.

        Args:
            payload: Raw fatal error payload received from Sarvam.
        """
        self.payload = payload
        code = payload.get("code", "unknown")
        message = payload.get("message", payload)
        status_code = payload.get("status_code")
        if status_code is not None:
            super().__init__(f"{code} ({status_code}): {message}")
        else:
            super().__init__(f"{code}: {message}")


class SarvamRealtimeSTTUsageMetricsData(MetricsData):
    """Sarvam realtime STT audio usage metrics data.

    Parameters:
        value: Audio duration processed by Sarvam realtime STT, in seconds.
    """

    value: float


@dataclass
class SarvamRealtimeSTTSettings(STTSettings):
    """Settings for SarvamRealtimeSTTService.

    Parameters:
        language_code: Sarvam realtime language code or ``auto``.
        stream_type: Streaming cadence: ``fast``, ``balanced``, or ``simulated``.
        endpointing: Turn endpointing mode: ``vad`` or ``manual``.
        encoding: Raw audio encoding sent over the websocket.
        sample_rate: Declared input audio sample rate.
        mode: Realtime STT task mode.
        prompt: Optional decoding prompt.
        return_timestamps: Whether final transcripts should include segment offsets.
        threshold: Optional VAD sensitivity threshold.
        prefix_padding_ms: Optional VAD prefix padding.
        silence_duration_ms: Optional silence duration for end-of-speech.
        min_speech_duration_ms: Optional minimum speech duration.
        lid_gate_seconds: Optional auto-LID gate duration.
        lid_confidence_threshold: Optional auto-LID confidence threshold.
    """

    language_code: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stream_type: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    endpointing: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    encoding: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    sample_rate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    mode: Literal["transcribe", "translate", "verbatim", "translit", "codemix"] | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    return_timestamps: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prefix_padding_ms: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    silence_duration_ms: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_speech_duration_ms: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    lid_gate_seconds: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    lid_confidence_threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SarvamRealtimeSTTService(WebsocketSTTService):
    """Sarvam realtime Speech-to-Text service.

    Streams raw audio bytes to Sarvam's realtime websocket endpoint and maps
    provider VAD and transcript events into Pipecat frames.
    """

    Settings = SarvamRealtimeSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = REALTIME_STT_URL,
        sample_rate: int = 16000,
        settings: Settings | None = None,
        should_interrupt: bool = True,
        session_end_timeout: float = 0.5,
        ttfs_p99_latency: float | None = SARVAM_TTFS_P99,
        **kwargs,
    ):
        """Initialize Sarvam realtime STT.

        Args:
            api_key: Sarvam API key.
            base_url: Realtime STT websocket endpoint.
            sample_rate: Input audio sample rate. Supported values are 8000 and 16000.
            settings: Runtime-updatable realtime settings.
            should_interrupt: Whether provider speech-start events should broadcast interruption.
            session_end_timeout: Seconds to wait for ``session.end`` during clean shutdown.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to :class:`WebsocketSTTService`.
        """
        default_settings = self.Settings(
            model=REALTIME_MODEL,
            language=None,
            language_code="hi-IN",
            stream_type="fast",
            endpointing="vad",
            encoding="linear16",
            sample_rate=sample_rate,
            mode="transcribe",
            prompt=None,
            return_timestamps=False,
            threshold=None,
            prefix_padding_ms=None,
            silence_duration_ms=None,
            min_speech_duration_ms=None,
            lid_gate_seconds=None,
            lid_confidence_threshold=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        if default_settings.language is not None and default_settings.language_code == "hi-IN":
            default_settings.language_code = language_to_sarvam_realtime_language(
                assert_given(default_settings.language)
            )

        self._validate_settings(default_settings)
        resolved_sample_rate = assert_given(default_settings.sample_rate)

        super().__init__(
            sample_rate=resolved_sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._should_interrupt = should_interrupt
        self._session_end_timeout = session_end_timeout
        self._receive_task: asyncio.Task | None = None
        self._audio_buffer = bytearray()
        self._request_id: str | None = None
        self._provider_speech_active = False
        self._speech_end_wall_time: float | None = None
        self._speech_end_audio_position_s: float | None = None
        self._audio_position_s = 0.0
        self._local_audio_duration_s = 0.0
        self._server_usage_reported = False
        self._session_end_event = asyncio.Event()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Language enum to Sarvam realtime's language code."""
        return language_to_sarvam_realtime_language(language)

    async def start(self, frame: StartFrame):
        """Start the service and connect the websocket."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the websocket."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the websocket."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and send manual endpointing boundaries when configured."""
        await super().process_frame(frame, direction)
        if self._settings.endpointing != "manual":
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._send_json({"event": "speech_start"})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._send_json({"event": "speech_end"})

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Buffer and send raw audio bytes to Sarvam as binary websocket frames."""
        if not audio:
            yield None
            return
        if not self._is_websocket_open():
            yield None
            return

        self._audio_buffer.extend(audio)
        chunk_size = self._chunk_size_bytes()
        while len(self._audio_buffer) >= chunk_size:
            chunk = bytes(self._audio_buffer[:chunk_size])
            del self._audio_buffer[:chunk_size]
            try:
                await self._send_audio_chunk(chunk)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Sarvam realtime STT send failed: {e}", exception=e
                )
                break

        yield None

    def _build_ws_url(self) -> str:
        """Build the Sarvam realtime websocket URL."""
        params = self._query_params()
        return f"{self._base_url}?{urlencode(params)}"

    async def _connect(self):
        """Connect to Sarvam realtime and start receive task."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Sarvam realtime and emit local usage fallback if needed."""
        await super()._disconnect()
        if self._websocket and self._websocket.state is State.OPEN:
            await self._flush_audio_buffer()
            try:
                await self._send_json({"event": "end"})
                try:
                    await asyncio.wait_for(
                        self._session_end_event.wait(), timeout=self._session_end_timeout
                    )
                except TimeoutError:
                    logger.debug(f"{self} timed out waiting for Sarvam session.end")
            except Exception as e:
                logger.debug(f"{self} error sending Sarvam end event: {e}")

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()
        if not self._server_usage_reported and self._local_audio_duration_s > 0:
            await self._emit_usage(self._local_audio_duration_s)

    async def _connect_websocket(self):
        """Open the Sarvam realtime websocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = self._build_ws_url()
            headers = {"API-SUBSCRIPTION-KEY": self._api_key}
            self._session_end_event.clear()
            self._server_usage_reported = False
            logger.debug(f"Connecting to Sarvam realtime STT WebSocket: {url}")
            self._websocket = await websocket_connect(
                url,
                additional_headers=headers,
                user_agent_header=sdk_headers()["User-Agent"],
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            logger.exception(f"Failed to connect to Sarvam realtime STT: {self._build_ws_url()}")
            await self.push_error(
                error_msg=f"Unable to connect to Sarvam realtime STT: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect_websocket(self):
        """Close the active websocket."""
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Error closing Sarvam realtime STT websocket: {e}", exception=e
            )
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Receive Sarvam realtime server events."""
        if not self._websocket:
            raise Exception("Websocket not connected")
        async for message in self._websocket:
            if not isinstance(message, str):
                logger.trace(f"{self} ignored non-text Sarvam server message")
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON Sarvam message: {message}")
                continue
            await self._handle_message(data)

    async def _handle_message(self, message: dict[str, Any]):
        """Handle a parsed Sarvam realtime server event."""
        event = message.get("event")
        if event == "session.begin":
            self._request_id = message.get("request_id")
            logger.info(f"{self} Sarvam realtime session.begin request_id={self._request_id}")
        elif event == "vad.speech_start":
            await self._handle_speech_start(message)
        elif event == "vad.speech_end":
            await self._handle_speech_end(message)
        elif event == "transcript.partial":
            await self._handle_partial_transcript(message)
        elif event == "transcript.final":
            await self._handle_final_transcript(message)
        elif event == "session.end":
            await self._handle_session_end(message)
        elif event in {"config.updated", "pong"}:
            logger.trace(f"{self} Sarvam realtime acknowledgement: {message}")
        elif event == "error":
            await self._handle_error(message)
        else:
            logger.trace(f"{self} unhandled Sarvam realtime event: {message}")

    async def update_config(self, **fields: Any):
        """Send a live Sarvam ``config.update`` message without reconnecting."""
        if not fields:
            return
        self._validate_config_update(fields)
        payload = {"event": "config.update", **fields}
        await self._send_json(payload)

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply runtime settings and send supported fields via ``config.update``."""
        proposed = self._settings.copy()
        proposed.apply_update(delta)
        self._validate_settings(proposed)
        changed = await super()._update_settings(delta)
        if not changed:
            return changed

        config_fields = {
            "language_code",
            "stream_type",
            "endpointing",
            "mode",
            "prompt",
            "threshold",
            "silence_duration_ms",
            "min_speech_duration_ms",
            "lid_gate_seconds",
            "lid_confidence_threshold",
        }
        unsupported = set(changed) - config_fields
        if unsupported:
            self._warn_unhandled_updated_settings({key: changed[key] for key in unsupported})

        payload = {
            key: getattr(self._settings, key)
            for key in changed
            if key in config_fields and is_given(getattr(self._settings, key))
        }
        if payload:
            await self.update_config(**payload)
        return changed

    async def _handle_speech_start(self, message: dict[str, Any]):
        if self._provider_speech_active:
            return
        self._provider_speech_active = True
        self._speech_end_wall_time = None
        self._speech_end_audio_position_s = None
        await self.start_processing_metrics()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()

    async def _handle_speech_end(self, message: dict[str, Any]):
        if not self._provider_speech_active:
            return
        self._provider_speech_active = False
        self._speech_end_wall_time = time.time()
        self._speech_end_audio_position_s = self._audio_position_s
        await self.broadcast_frame(UserStoppedSpeakingFrame)
        await self.start_ttfb_metrics(start_time=self._speech_end_wall_time)

    async def _handle_partial_transcript(self, message: dict[str, Any]):
        text = (message.get("text") or "").strip()
        if not text:
            return
        result = self._result_payload(message)
        result.setdefault("confidence", 0.0)
        language = self._language_for_frame(message.get("language"))
        await self.push_frame(
            InterimTranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=result,
            )
        )

    async def _handle_final_transcript(self, message: dict[str, Any]):
        text = (message.get("text") or "").strip()
        if not text:
            return
        language = self._language_for_frame(message.get("language"))
        result = self._result_payload(message)
        result["speech_end_audio_position_s"] = self._speech_end_audio_position_s
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=result,
                finalized=True,
            )
        )
        await self._trace_transcription(text, True, language)
        await self.stop_processing_metrics()

    async def _handle_session_end(self, message: dict[str, Any]):
        if message.get("request_id"):
            self._request_id = message.get("request_id")
        audio_duration_s = message.get("audio_duration_s")
        if audio_duration_s is not None:
            await self._emit_usage(float(audio_duration_s))
            self._server_usage_reported = True
        self._session_end_event.set()

    async def _handle_error(self, message: dict[str, Any]):
        code = message.get("code", "unknown")
        is_fatal = bool(message.get("is_fatal"))
        logger.warning(
            f"{self} Sarvam realtime error code={code} "
            f"is_fatal={is_fatal} request_id={self._request_id}"
        )
        logger.debug(f"{self} Sarvam realtime error payload: {message}")
        if not is_fatal:
            return
        error = SarvamRealtimeSTTError(message)
        await self.push_error(
            error_msg=f"Sarvam realtime STT fatal error: {error}", exception=error, fatal=True
        )
        await self.stop_all_metrics()
        raise error

    async def _send_audio_chunk(self, chunk: bytes):
        if not self._websocket:
            raise RuntimeError("WebSocket not connected")
        await self._websocket.send(chunk)
        duration = self._duration_for_bytes(len(chunk))
        self._local_audio_duration_s += duration
        self._audio_position_s += duration

    async def _flush_audio_buffer(self):
        if not self._audio_buffer:
            return
        chunk = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        await self._send_audio_chunk(chunk)

    async def _send_json(self, payload: dict[str, Any]):
        if not self._is_websocket_open():
            return
        assert self._websocket is not None
        await self._websocket.send(json.dumps(payload))

    async def _emit_usage(self, audio_duration_s: float):
        if not self.usage_metrics_enabled:
            return
        frame = MetricsFrame(
            data=[
                SarvamRealtimeSTTUsageMetricsData(
                    processor=self.name,
                    model=self._settings.model,
                    value=audio_duration_s,
                )
            ]
        )
        logger.debug(f"{self} usage audio seconds: {audio_duration_s}")
        await self.push_frame(frame)

    def _query_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "language_code": self._settings.language_code,
            "stream_type": self._settings.stream_type,
            "endpointing": self._settings.endpointing,
            "encoding": self._settings.encoding,
            "sample_rate": self._settings.sample_rate,
            "model": self._settings.model,
            "mode": self._settings.mode,
            "return_timestamps": str(self._settings.return_timestamps).lower(),
        }
        optional = {
            "prompt": self._settings.prompt,
            "threshold": self._settings.threshold,
            "prefix_padding_ms": self._settings.prefix_padding_ms,
            "silence_duration_ms": self._settings.silence_duration_ms,
            "min_speech_duration_ms": self._settings.min_speech_duration_ms,
            "lid_gate_seconds": self._settings.lid_gate_seconds,
            "lid_confidence_threshold": self._settings.lid_confidence_threshold,
        }
        params.update({key: value for key, value in optional.items() if value is not None})
        return params

    def _chunk_size_bytes(self) -> int:
        stream_type = assert_given(self._settings.stream_type)
        chunk_ms = 500 if stream_type == "fast" else 1000
        return int(
            assert_given(self._settings.sample_rate)
            * (chunk_ms / 1000)
            * _BYTES_PER_SAMPLE[assert_given(self._settings.encoding)]
        )

    def _duration_for_bytes(self, byte_count: int) -> float:
        return byte_count / (
            assert_given(self._settings.sample_rate)
            * _BYTES_PER_SAMPLE[assert_given(self._settings.encoding)]
        )

    def _result_payload(self, message: dict[str, Any]) -> dict[str, Any]:
        payload = dict(message)
        if self._request_id is not None:
            payload.setdefault("request_id", self._request_id)
        return payload

    def _language_for_frame(self, raw_language: str | None = None) -> Language | None:
        language_code = self._normalize_language_code(raw_language or self._settings.language_code)
        if language_code == "auto":
            return None
        try:
            return Language(language_code)
        except ValueError:
            return None

    def _normalize_language_code(self, language_code: str | None) -> str:
        if not language_code:
            return assert_given(self._settings.language_code)
        if "-" not in language_code and language_code != "auto":
            configured = assert_given(self._settings.language_code)
            if configured != "auto" and configured.startswith(f"{language_code}-"):
                return configured
            return _SHORT_LANGUAGE_DEFAULTS.get(language_code, language_code)
        return language_code

    def _is_websocket_open(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    def _validate_config_update(self, fields: dict[str, Any]):
        if "stream_type" in fields:
            current = assert_given(self._settings.stream_type)
            requested = fields["stream_type"]
            if current == "simulated" or requested == "simulated":
                raise ValueError("Changing to or from stream_type='simulated' is not supported.")
        proposed = self._settings.copy()
        proposed.apply_update(self.Settings(**fields))
        self._validate_settings(proposed)

    @staticmethod
    def _validate_settings(settings: Settings):
        model = assert_given(settings.model)
        if model != REALTIME_MODEL:
            raise ValueError(f"Unsupported model '{model}'. Only '{REALTIME_MODEL}' is supported.")

        language_code = assert_given(settings.language_code)
        if language_code not in SUPPORTED_LANGUAGES:
            allowed = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(
                f"Unsupported language_code '{language_code}'. Allowed values: {allowed}."
            )

        stream_type = assert_given(settings.stream_type)
        if stream_type not in SUPPORTED_STREAM_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_STREAM_TYPES))
            raise ValueError(f"Unsupported stream_type '{stream_type}'. Allowed values: {allowed}.")

        endpointing = assert_given(settings.endpointing)
        if endpointing not in SUPPORTED_ENDPOINTING:
            allowed = ", ".join(sorted(SUPPORTED_ENDPOINTING))
            raise ValueError(f"Unsupported endpointing '{endpointing}'. Allowed values: {allowed}.")

        encoding = assert_given(settings.encoding)
        if encoding not in SUPPORTED_ENCODINGS:
            allowed = ", ".join(sorted(SUPPORTED_ENCODINGS))
            raise ValueError(f"Unsupported encoding '{encoding}'. Allowed values: {allowed}.")

        sample_rate = assert_given(settings.sample_rate)
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            allowed = ", ".join(str(rate) for rate in sorted(SUPPORTED_SAMPLE_RATES))
            raise ValueError(f"Unsupported sample_rate '{sample_rate}'. Allowed values: {allowed}.")

        mode = assert_given(settings.mode)
        if mode not in SUPPORTED_MODES:
            allowed = ", ".join(sorted(SUPPORTED_MODES))
            raise ValueError(f"Unsupported mode '{mode}'. Allowed values: {allowed}.")

        _validate_optional_range("threshold", settings.threshold, minimum=0.0, maximum=1.0)
        _validate_optional_minimum("prefix_padding_ms", settings.prefix_padding_ms, minimum=0)
        _validate_optional_minimum("silence_duration_ms", settings.silence_duration_ms, minimum=0)
        _validate_optional_minimum(
            "min_speech_duration_ms", settings.min_speech_duration_ms, minimum=0
        )
        _validate_optional_minimum("lid_gate_seconds", settings.lid_gate_seconds, minimum=0.0)
        _validate_optional_range(
            "lid_confidence_threshold",
            settings.lid_confidence_threshold,
            minimum=0.0,
            maximum=1.0,
        )

    @traced_stt
    async def _trace_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record transcription event for tracing."""
        pass


def _validate_optional_range(
    field_name: str,
    value: float | int | None | _NotGiven,
    *,
    minimum: float,
    maximum: float,
):
    if not is_given(value) or value is None:
        return
    if value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")


def _validate_optional_minimum(
    field_name: str,
    value: float | int | None | _NotGiven,
    *,
    minimum: float,
):
    if not is_given(value) or value is None:
        return
    if value < minimum:
        raise ValueError(f"{field_name} must be greater than or equal to {minimum}.")
