#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam realtime Speech-to-Text service implementation."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.sarvam._utils import PeriodicCollector
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given, is_given
from pipecat.services.stt_latency import SARVAM_REALTIME_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sarvam realtime STT, install the websockets package.")
    raise ImportError(f"Missing module: {e}") from e


SARVAM_STT_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
REALTIME_MODEL = "saaras:v3-realtime"

RealtimeStreamType = Literal["fast", "balanced", "simulated"]
RealtimeEndpointing = Literal["vad", "manual"]
RealtimeEncoding = Literal["linear16"]
RealtimeMode = Literal["transcribe", "translate", "indic-en", "verbatim", "translit", "codemix"]

SUPPORTED_SAMPLE_RATES = {8000, 16000}
SUPPORTED_STREAM_TYPES = {"fast", "balanced", "simulated"}
SUPPORTED_ENDPOINTING = {"vad", "manual"}
SUPPORTED_ENCODINGS = {"linear16"}
SUPPORTED_MODES = {"transcribe", "translate", "indic-en", "verbatim", "translit", "codemix"}
STREAM_TYPE_CHUNK_MS = {"fast": 500, "balanced": 1000, "simulated": 1000}
EOS_FALLBACK_TIMEOUT = 2.0
SESSION_END_DRAIN_TIMEOUT = 2.0

SUPPORTED_LANGUAGES = {
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
    "auto",
}


class SarvamRealtimeSTTError(RuntimeError):
    """Sarvam realtime STT API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        body: Any | None = None,
        retryable: bool = False,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error message.
            status_code: Optional API or WebSocket status code.
            request_id: Sarvam request ID, when known.
            body: Raw error payload.
            retryable: Whether retrying the operation may succeed.
        """
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.body = body
        self.retryable = retryable


def language_to_sarvam_realtime_language(language: Language) -> str:
    """Convert a Pipecat language enum to Sarvam realtime BCP-47 language code."""
    language_map = {
        Language.EN_IN: "en-IN",
        Language.HI_IN: "hi-IN",
        Language.BN_IN: "bn-IN",
        Language.KN_IN: "kn-IN",
        Language.ML_IN: "ml-IN",
        Language.MR_IN: "mr-IN",
        Language.OR_IN: "or-IN",
        Language.PA_IN: "pa-IN",
        Language.TA_IN: "ta-IN",
        Language.TE_IN: "te-IN",
        Language.GU_IN: "gu-IN",
        Language.AS_IN: "as-IN",
        Language.UR_IN: "ur-IN",
        Language.KOK_IN: "kok-IN",
        Language.SD_IN: "sd-IN",
        Language.MAI_IN: "mai-IN",
    }
    return resolve_language(language, language_map, use_base_code=False)


def _coerce_realtime_language(language: Language | str | None | _NotGiven) -> str | _NotGiven:
    if not is_given(language):
        return NOT_GIVEN
    if isinstance(language, Language):
        return language_to_sarvam_realtime_language(language)
    if language is None:
        return "hi-IN"
    return language


def _build_realtime_ws_url(
    base_url: str,
    settings: SarvamRealtimeSTTSettings,
    sample_rate: int,
) -> str:
    """Build a Sarvam realtime STT WebSocket URL."""
    language = str(settings.language if is_given(settings.language) else "hi-IN")
    stream_type = str(settings.stream_type if is_given(settings.stream_type) else "fast")
    endpointing = str(settings.endpointing if is_given(settings.endpointing) else "vad")
    encoding = str(settings.encoding if is_given(settings.encoding) else "linear16")
    model = str(settings.model if is_given(settings.model) else REALTIME_MODEL)
    mode = str(settings.mode if is_given(settings.mode) else "transcribe")

    params: dict[str, str] = {
        "language-code": language,
        "stream-type": stream_type,
        "endpointing": endpointing,
        "encoding": encoding,
        "sample_rate": str(sample_rate),
        "model": model,
    }

    if stream_type == "simulated":
        params["mode"] = mode

    if endpointing == "vad":
        if is_given(settings.vad_sot_threshold) and settings.vad_sot_threshold is not None:
            params["vad_sot_threshold"] = str(settings.vad_sot_threshold)
        if is_given(settings.vad_min_speech_ms) and settings.vad_min_speech_ms is not None:
            params["vad_min_speech_ms"] = str(settings.vad_min_speech_ms)
        if is_given(settings.vad_min_silence_ms) and settings.vad_min_silence_ms is not None:
            params["vad_min_silence_ms"] = str(settings.vad_min_silence_ms)
        if is_given(settings.vad_smoothing_alpha) and settings.vad_smoothing_alpha is not None:
            params["vad_smoothing_alpha"] = str(settings.vad_smoothing_alpha)

    return f"{base_url}?{urlencode(params)}"


@dataclass
class SarvamRealtimeSTTSettings(STTSettings):
    """Settings for Sarvam realtime STT.

    Parameters:
        language: BCP-47 language code or Pipecat language enum.
        stream_type: Realtime streaming mode: fast, balanced, or simulated.
        mode: Transcription/translation mode, honored only with simulated stream type.
        endpointing: Server VAD or manual endpointing.
        encoding: Audio encoding. Only linear16 is supported.
        vad_sot_threshold: Optional VAD speech-onset threshold.
        vad_min_speech_ms: Optional minimum speech duration in milliseconds.
        vad_min_silence_ms: Optional minimum silence duration in milliseconds.
        vad_smoothing_alpha: Optional VAD smoothing alpha.
    """

    language: Language | str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stream_type: RealtimeStreamType | str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    mode: RealtimeMode | str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    endpointing: RealtimeEndpointing | str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    encoding: RealtimeEncoding | str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_sot_threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_min_speech_ms: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_min_silence_ms: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_smoothing_alpha: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    def __post_init__(self) -> None:
        if is_given(self.language):
            self.language = _coerce_realtime_language(self.language)
        if is_given(self.model) and self.model != REALTIME_MODEL:
            raise ValueError(f"model must be {REALTIME_MODEL}")
        if is_given(self.language) and self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language {self.language} is not supported")
        if is_given(self.stream_type) and self.stream_type not in SUPPORTED_STREAM_TYPES:
            allowed = ", ".join(sorted(SUPPORTED_STREAM_TYPES))
            raise ValueError(f"stream_type must be one of {allowed}")
        if (
            is_given(self.language)
            and self.language == "auto"
            and is_given(self.stream_type)
            and self.stream_type != "simulated"
        ):
            raise ValueError("language auto is only supported when stream_type is simulated")
        if is_given(self.mode) and self.mode not in SUPPORTED_MODES:
            allowed = ", ".join(sorted(SUPPORTED_MODES))
            raise ValueError(f"mode must be one of {allowed}")
        if is_given(self.endpointing) and self.endpointing not in SUPPORTED_ENDPOINTING:
            allowed = ", ".join(sorted(SUPPORTED_ENDPOINTING))
            raise ValueError(f"endpointing must be one of {allowed}")
        if is_given(self.encoding) and self.encoding not in SUPPORTED_ENCODINGS:
            allowed = ", ".join(sorted(SUPPORTED_ENCODINGS))
            raise ValueError(f"encoding must be one of {allowed}")
        if (
            is_given(self.vad_sot_threshold)
            and self.vad_sot_threshold is not None
            and not 0.0 <= self.vad_sot_threshold <= 1.0
        ):
            raise ValueError("vad_sot_threshold must be between 0.0 and 1.0")
        if (
            is_given(self.vad_smoothing_alpha)
            and self.vad_smoothing_alpha is not None
            and not 0.0 < self.vad_smoothing_alpha <= 1.0
        ):
            raise ValueError("vad_smoothing_alpha must be greater than 0.0 and at most 1.0")
        if (
            is_given(self.vad_min_speech_ms)
            and self.vad_min_speech_ms is not None
            and self.vad_min_speech_ms < 0
        ):
            raise ValueError("vad_min_speech_ms must be greater than or equal to 0")
        if (
            is_given(self.vad_min_silence_ms)
            and self.vad_min_silence_ms is not None
            and self.vad_min_silence_ms < 0
        ):
            raise ValueError("vad_min_silence_ms must be greater than or equal to 0")


class SarvamRealtimeSTTService(WebsocketSTTService):
    """Speech-to-text service for Sarvam's realtime ``saaras:v3-realtime`` API."""

    Settings = SarvamRealtimeSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = SARVAM_STT_REALTIME_URL,
        sample_rate: int | None = None,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = SARVAM_REALTIME_TTFS_P99,
        reconnect_on_error: bool = False,
        keepalive_timeout: float | None = None,
        keepalive_interval: float = 5.0,
        **kwargs,
    ):
        """Initialize the Sarvam realtime STT service.

        Args:
            api_key: Sarvam API key. Defaults to ``SARVAM_API_KEY``.
            base_url: Sarvam realtime STT WebSocket URL.
            sample_rate: Audio sample rate, either 8000 or 16000.
            should_interrupt: Whether server VAD start should interrupt bot speech.
            settings: Runtime-updatable realtime settings.
            ttfs_p99_latency: P99 speech-end to final transcript latency.
            reconnect_on_error: Whether Pipecat should reconnect on WebSocket errors.
            keepalive_timeout: Seconds of no audio before sending silence.
            keepalive_interval: Seconds between keepalive checks.
            **kwargs: Additional arguments passed to ``WebsocketSTTService``.
        """
        api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not api_key:
            raise ValueError(
                "Sarvam API key is required. "
                "Provide it directly or set SARVAM_API_KEY environment variable."
            )
        if sample_rate is not None and sample_rate not in SUPPORTED_SAMPLE_RATES:
            allowed = ", ".join(str(r) for r in sorted(SUPPORTED_SAMPLE_RATES))
            raise ValueError(f"sample_rate must be one of {allowed}")

        default_settings = self.Settings(
            model=REALTIME_MODEL,
            language="hi-IN",
            stream_type="fast",
            mode="transcribe",
            endpointing="vad",
            encoding="linear16",
            vad_sot_threshold=None,
            vad_min_speech_ms=None,
            vad_min_silence_ms=None,
            vad_smoothing_alpha=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)
            default_settings.__post_init__()

        super().__init__(
            sample_rate=sample_rate or 16000,
            reconnect_on_error=reconnect_on_error,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=keepalive_timeout,
            keepalive_interval=keepalive_interval,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._should_interrupt = should_interrupt
        self._sdk_headers = sdk_headers()

        self._receive_task: asyncio.Task | None = None
        self._audio_buffer = bytearray()
        self._chunk_size_bytes = self._calculate_chunk_size_bytes()
        self._manual_speech_started = False

        self._request_id = ""
        self._session_id = ""
        self._session_ended = False
        self._session_end_event = asyncio.Event()
        self._pending_eos = False
        self._pending_eos_time: float | None = None
        self._pending_final_data: dict[str, Any] | None = None
        self._utterance_start_audio_pos = 0.0
        self._utterance_speech_end_audio_pos: float | None = None
        self._utterance_speech_end_wall: float | None = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False
        self._eos_fallback_task: asyncio.Task | None = None
        self._audio_position = 0.0
        self._total_reported_audio_duration = 0.0
        self._audio_duration_collector = PeriodicCollector(
            callback=self._schedule_audio_duration_report,
            duration=5.0,
        )

        self._register_event_handler("on_usage")

    @property
    def provider(self) -> str:
        """Return the STT provider name."""
        return "Sarvam"

    @property
    def supports_ttfs(self) -> bool:
        """Sarvam realtime exposes server speech-end and final transcript timing."""
        return True

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert Pipecat language enum to Sarvam realtime language code."""
        return language_to_sarvam_realtime_language(language)

    async def start(self, frame: StartFrame):
        """Start the service and connect the WebSocket."""
        await super().start(frame)
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            allowed = ", ".join(str(r) for r in sorted(SUPPORTED_SAMPLE_RATES))
            raise ValueError(f"sample_rate must be one of {allowed}")
        self._chunk_size_bytes = self._calculate_chunk_size_bytes()
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service gracefully."""
        await super().stop(frame)
        await self._flush_audio_buffer()
        await self._send_manual_speech_end()
        await self._send_json({"event": "end"})
        self._audio_duration_collector.flush()
        await self._drain_session_end()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service immediately."""
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        changed = await super()._update_settings(delta)
        if not changed:
            return changed

        self._settings.__post_init__()
        self._chunk_size_bytes = self._calculate_chunk_size_bytes()
        if self._websocket:
            await self._disconnect()
            await self._connect()
        return changed

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error),
                name="sarvam_realtime_receive",
            )

    async def _disconnect(self):
        await super()._disconnect()
        self._cancel_eos_fallback()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    def _calculate_chunk_size_bytes(self) -> int:
        chunk_ms = STREAM_TYPE_CHUNK_MS[assert_given(self._settings.stream_type)]
        return int(self._effective_sample_rate() * 2 * chunk_ms / 1000)

    def _effective_sample_rate(self) -> int:
        return self.sample_rate or self._init_sample_rate or 16000

    def _websocket_url(self) -> str:
        return _build_realtime_ws_url(self._base_url, self._settings, self._effective_sample_rate())

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._reset_connection_state()
            headers = {
                **self._sdk_headers,
                "API-SUBSCRIPTION-KEY": self._api_key,
            }
            logger.info(
                "Connecting to Sarvam realtime STT WebSocket",
                **self._build_log_context(),
            )
            self._websocket = await websocket_connect(
                self._websocket_url(),
                additional_headers=headers,
                ping_interval=30,
            )
            await self._call_event_handler("on_connected")
            logger.info("Connected to Sarvam realtime STT WebSocket", **self._build_log_context())
        except Exception as e:
            self._websocket = None
            await self.push_error(
                error_msg=f"Unable to connect to Sarvam realtime STT: {e}", exception=e
            )
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
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

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Buffer and send raw PCM audio to Sarvam."""
        self._audio_buffer.extend(audio)
        try:
            await self._flush_audio_buffer(only_full_chunks=True)
            yield None
        except SarvamRealtimeSTTError as e:
            yield ErrorFrame(error=str(e))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process Pipecat frames and map manual endpointing VAD boundaries."""
        await super().process_frame(frame, direction)

        if assert_given(self._settings.endpointing) != "manual":
            return

        if isinstance(frame, VADUserStartedSpeakingFrame) and not self._manual_speech_started:
            await self._send_json({"event": "speech_start"})
            self._manual_speech_started = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._send_manual_speech_end()

    async def _flush_audio_buffer(self, *, only_full_chunks: bool = False) -> None:
        while self._audio_buffer and (
            len(self._audio_buffer) >= self._chunk_size_bytes or not only_full_chunks
        ):
            if only_full_chunks:
                chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
                del self._audio_buffer[: self._chunk_size_bytes]
            else:
                chunk = bytes(self._audio_buffer)
                self._audio_buffer.clear()
            if not chunk:
                return
            await self._send_audio_chunk(chunk)

    async def _send_audio_chunk(self, chunk: bytes) -> None:
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        if assert_given(self._settings.endpointing) == "manual" and not self._manual_speech_started:
            await self._send_json({"event": "speech_start"})
            self._manual_speech_started = True

        try:
            if self._reconnect_on_error:
                await self.send_with_retry(chunk, self._report_error)
            else:
                await self._websocket.send(chunk)
        except Exception as e:
            raise SarvamRealtimeSTTError(
                f"Sarvam realtime STT send error: {e}",
                request_id=self._request_id or None,
            ) from e

        duration = len(chunk) / (self._effective_sample_rate() * 2)
        self._audio_position += duration
        self._audio_duration_collector.push(duration)

    async def _send_keepalive(self, silence: bytes):
        await self._send_audio_chunk(silence)

    async def _send_manual_speech_end(self) -> None:
        if assert_given(self._settings.endpointing) == "manual" and self._manual_speech_started:
            await self._send_json({"event": "speech_end"})
            self._manual_speech_started = False

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._websocket.send(json.dumps(payload))
        except (ConnectionClosed, ConnectionError):
            logger.debug("Sarvam realtime STT WebSocket closed before send completed")

    async def _receive_messages(self):
        try:
            async for message in self._get_websocket():
                if not isinstance(message, str):
                    logger.warning(
                        f"Received non-text Sarvam realtime STT message: {type(message)}"
                    )
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    if self._looks_like_error_text(message):
                        raise SarvamRealtimeSTTError(
                            f"Sarvam realtime STT non-JSON error message: {message}",
                            request_id=self._request_id or None,
                            body={"raw_message": message},
                        ) from e
                    logger.warning(
                        "Invalid JSON received from Sarvam realtime STT",
                        raw_data=message,
                        **self._build_log_context(),
                    )
                    continue
                logger.debug(f"Sarvam realtime STT raw server message: raw_data={data}")
                await self._handle_message(data)
        except ConnectionClosedError as e:
            raise self._exception_from_connection_closed(e) from e

    async def _receive_task_handler(self, report_error):
        while True:
            self._last_connect_time = time.monotonic()
            try:
                await self._receive_messages()
                if self._session_ended:
                    break
                message = f"{self} connection closed by server"
                should_continue = await self._maybe_try_reconnect(message, report_error)
                if not should_continue:
                    break
            except ConnectionClosedOK:
                break
            except SarvamRealtimeSTTError as e:
                await report_error(ErrorFrame(str(e)))
                if e.retryable and self._reconnect_on_error:
                    if await self._try_reconnect(report_error=report_error):
                        continue
                break
            except ConnectionClosedError as e:
                error = self._exception_from_connection_closed(e)
                await report_error(ErrorFrame(str(error)))
                if error.retryable and self._reconnect_on_error:
                    if await self._try_reconnect(report_error=report_error):
                        continue
                break
            except Exception as e:
                message = f"{self} error receiving messages: {e}"
                should_continue = await self._maybe_try_reconnect(message, report_error, e)
                if not should_continue:
                    break

    @staticmethod
    def _looks_like_error_text(text: str | None) -> bool:
        if not text:
            return False
        lower = text.lower()
        return any(token in lower for token in ("error", "unauthorized", "forbidden", "quota"))

    async def _handle_message(self, data: dict[str, Any]) -> None:
        event = data.get("event")
        self._capture_server_ids(data)
        self._log_stt_event(event, data)

        if event == "session.begin":
            return
        if event == "vad.speech_start":
            await self._handle_speech_start()
        elif event == "vad.speech_end":
            await self._handle_speech_end(data)
        elif event == "transcript.partial":
            await self._send_transcript_frame(data, final=False)
        elif event == "transcript.final":
            if assert_given(self._settings.endpointing) == "vad":
                if self._is_valid_transcript(data):
                    self._pending_final_data = data
                    self._final_received_for_utterance = True
                    await self._try_commit_utterance()
            else:
                await self._send_transcript_frame(data, final=True)
                self._final_received_for_utterance = True
        elif event == "session.end":
            await self._handle_session_end(data)
        elif event == "error":
            await self._handle_error_event(data)
        elif event == "pong":
            return
        else:
            logger.debug("Unknown Sarvam realtime STT event", event=event, data=data)

    async def _handle_speech_start(self) -> None:
        self._reset_utterance_state()
        await self.start_processing_metrics()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()

    async def _handle_speech_end(self, data: dict[str, Any] | None = None) -> None:
        self._utterance_speech_end_audio_pos = self._audio_position
        self._utterance_speech_end_wall = time.time()

        if assert_given(self._settings.endpointing) != "vad":
            await self._emit_end_of_speech()
            return

        if self._eos_emitted_for_utterance:
            return

        if self._final_received_for_utterance:
            await self._try_commit_utterance()
            return

        self._pending_eos = True
        self._pending_eos_time = self._utterance_speech_end_wall
        if self._eos_fallback_task is None or self._eos_fallback_task.done():
            self._eos_fallback_task = self._create_service_task(
                self._emit_pending_eos_after_timeout(),
                name="sarvam_realtime_eos_fallback",
            )

    def _create_service_task(self, coroutine, *, name: str) -> asyncio.Task:
        if self._task_manager is not None:
            return self.create_task(coroutine, name=name)
        return asyncio.create_task(coroutine, name=name)

    async def _emit_pending_eos_after_timeout(self, timeout: float = EOS_FALLBACK_TIMEOUT) -> None:
        try:
            if timeout > 0:
                await asyncio.sleep(timeout)
            if self._pending_eos and not self._eos_emitted_for_utterance:
                await self._emit_end_of_speech()
                await self.stop_processing_metrics()
        except asyncio.CancelledError:
            raise

    async def _try_commit_utterance(self) -> None:
        if (
            self._pending_final_data is None
            or self._utterance_speech_end_audio_pos is None
            or self._eos_emitted_for_utterance
        ):
            return

        await self.start_ttfb_metrics(start_time=self._utterance_speech_end_wall)
        if await self._send_transcript_frame(self._pending_final_data, final=True):
            logger.info(
                "Sarvam realtime STT utterance committed",
                end_time=self._utterance_speech_end_audio_pos,
                speech_end_wall_time=self._utterance_speech_end_wall,
                **self._build_log_context(),
            )
            await self._emit_end_of_speech()
            await self.stop_processing_metrics()
            self._pending_final_data = None

    async def _emit_end_of_speech(self) -> None:
        current_task = asyncio.current_task()
        fallback_task = self._eos_fallback_task
        self._eos_fallback_task = None
        if fallback_task and fallback_task is not current_task and not fallback_task.done():
            fallback_task.cancel()

        if self._eos_emitted_for_utterance:
            return

        await self.broadcast_frame(UserStoppedSpeakingFrame)
        self._pending_eos = False
        self._pending_eos_time = None
        self._eos_emitted_for_utterance = True

    async def _send_transcript_frame(self, data: dict[str, Any], *, final: bool) -> bool:
        text = data.get("text")
        if not isinstance(text, str) or not text:
            return False

        language = self._map_language_code_to_enum(data.get("language") or self._settings.language)
        confidence = data.get("language_confidence", data.get("confidence", 0.0))
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = 0.0

        result = self._transcript_result(data, final=final, confidence=float(confidence))
        if final:
            frame = TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=language,
                result=result,
                finalized=True,
            )
        else:
            frame = InterimTranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=language,
                result=result,
            )
        await self.push_frame(frame)
        await self._handle_transcription(text, is_final=final, language=language)
        return True

    def _transcript_result(
        self,
        data: dict[str, Any],
        *,
        final: bool,
        confidence: float,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "raw_data": data,
            "request_id": self._request_id,
            "session_id": self._session_id,
            "utterance_idx": data.get("utterance_idx"),
            "confidence": confidence,
        }
        if final:
            result["end_time"] = (
                self._utterance_speech_end_audio_pos
                if self._utterance_speech_end_audio_pos is not None
                else self._audio_position
            )
            if self._utterance_speech_end_wall is not None:
                result["speech_end_wall_time"] = self._utterance_speech_end_wall
        return result

    async def _handle_session_end(self, data: dict[str, Any]) -> None:
        self._capture_server_ids(data)
        audio_duration = data.get("audio_duration_s")
        if isinstance(audio_duration, (int, float)) and not isinstance(audio_duration, bool):
            delta = max(float(audio_duration) - self._total_reported_audio_duration, 0.0)
            if delta:
                await self._emit_usage(delta)
        self._session_ended = True
        self._session_end_event.set()

    async def _drain_session_end(self) -> None:
        if self._session_ended:
            return
        try:
            await asyncio.wait_for(
                self._session_end_event.wait(),
                timeout=SESSION_END_DRAIN_TIMEOUT,
            )
        except TimeoutError:
            logger.debug("Timed out waiting for Sarvam realtime STT session.end")

    async def _handle_error_event(self, data: dict[str, Any]) -> None:
        if not data.get("is_fatal", False):
            logger.warning(
                "Non-fatal Sarvam realtime STT error", error=data, **self._build_log_context()
            )
            return

        code = data.get("code", "unknown")
        status_code = data.get("status_code")
        if not isinstance(status_code, int):
            status_code = None
        raise SarvamRealtimeSTTError(
            f"Sarvam realtime STT error: {data.get('message', code)}",
            status_code=status_code,
            request_id=self._request_id or None,
            body=data,
            retryable=code == "model_unavailable",
        )

    def _exception_from_close_code(
        self, close_code: object, close_reason: object
    ) -> SarvamRealtimeSTTError:
        status_code = int(close_code) if isinstance(close_code, int) else None
        retryable = close_code == 1013
        message = f"Sarvam realtime STT WebSocket closed unexpectedly: {close_reason}"
        if close_code == 1003:
            message = "Sarvam realtime STT authentication, quota, or rate limit error"
        elif close_code == 1008:
            message = "Sarvam realtime STT session timed out or exceeded the maximum duration"
        elif close_code == 1013:
            message = "Sarvam realtime STT backend temporarily unavailable"
        elif close_code == 4000:
            message = f"Sarvam realtime STT rejected the session: {close_reason}"
        return SarvamRealtimeSTTError(
            message,
            status_code=status_code,
            request_id=self._request_id or None,
            body={"close_code": close_code, "close_reason": close_reason},
            retryable=retryable,
        )

    def _exception_from_connection_closed(
        self, error: ConnectionClosedError
    ) -> SarvamRealtimeSTTError:
        close = error.rcvd or error.sent
        close_code = close.code if close else getattr(error, "code", None)
        close_reason = close.reason if close else getattr(error, "reason", "")
        return self._exception_from_close_code(close_code, close_reason)

    def _schedule_audio_duration_report(self, duration: float) -> None:
        task = self._on_audio_duration_report(duration)
        if self._task_manager is not None:
            self.create_task(task, name="sarvam_realtime_usage_report")
        else:
            asyncio.create_task(task, name="sarvam_realtime_usage_report")

    async def _on_audio_duration_report(self, duration: float) -> None:
        await self._emit_usage(duration)

    async def _emit_usage(self, duration: float) -> None:
        self._total_reported_audio_duration += duration
        await self._call_event_handler("on_usage", self._request_id, duration)

    def _reset_utterance_state(self) -> None:
        self._cancel_eos_fallback()
        self._pending_eos = False
        self._pending_eos_time = None
        self._pending_final_data = None
        self._utterance_start_audio_pos = self._audio_position
        self._utterance_speech_end_audio_pos = None
        self._utterance_speech_end_wall = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False

    def _reset_connection_state(self) -> None:
        self._cancel_eos_fallback()
        self._request_id = ""
        self._session_id = ""
        self._session_ended = False
        self._session_end_event.clear()
        self._manual_speech_started = False
        self._pending_eos = False
        self._pending_eos_time = None
        self._pending_final_data = None
        self._utterance_start_audio_pos = 0.0
        self._utterance_speech_end_audio_pos = None
        self._utterance_speech_end_wall = None
        self._final_received_for_utterance = False
        self._eos_emitted_for_utterance = False
        self._audio_position = 0.0
        self._total_reported_audio_duration = 0.0

    def _cancel_eos_fallback(self) -> None:
        if self._eos_fallback_task and not self._eos_fallback_task.done():
            self._eos_fallback_task.cancel()
        self._eos_fallback_task = None

    @staticmethod
    def _extract_request_id(data: dict[str, Any]) -> str | None:
        request_id = data.get("request_id")
        if request_id is None:
            nested = data.get("data")
            if isinstance(nested, dict):
                request_id = nested.get("request_id")
        if request_id is None:
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                request_id = metadata.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
        return None

    @staticmethod
    def _extract_session_id(data: dict[str, Any]) -> str | None:
        session_id = data.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
        return None

    def _capture_server_ids(self, data: dict[str, Any]) -> None:
        session_id = self._extract_session_id(data)
        if session_id is not None:
            self._session_id = session_id
        if not self._request_id:
            request_id = self._extract_request_id(data)
            if request_id is not None:
                self._request_id = request_id

    def _build_log_context(self) -> dict[str, Any]:
        return {
            "request_id": self._request_id,
            "session_id": self._session_id,
            "model": assert_given(self._settings.model),
            "language": assert_given(self._settings.language),
            "stream_type": assert_given(self._settings.stream_type),
            "endpointing": assert_given(self._settings.endpointing),
        }

    def _log_stt_event(self, event: object, data: dict[str, Any]) -> None:
        if event == "pong":
            return
        extra: dict[str, Any] = {
            **self._build_log_context(),
            "event": event,
            "utterance_idx": data.get("utterance_idx"),
            "raw_data": data,
        }
        if event in {"transcript.partial", "transcript.final"}:
            text = data.get("text")
            if isinstance(text, str):
                extra["text"] = text[:200]
                extra["text_length"] = len(text)
            extra["language"] = data.get("language") or self._settings.language
            extra["confidence"] = data.get("language_confidence", data.get("confidence"))
        elif event in {"vad.speech_start", "vad.speech_end"}:
            extra["audio_position"] = self._audio_position
        elif event == "session.end":
            extra["audio_duration_s"] = data.get("audio_duration_s")
        elif event != "session.begin":
            return
        logger.info(f"Sarvam realtime STT {event}", **extra)

    @staticmethod
    def _is_valid_transcript(data: dict[str, Any]) -> bool:
        text = data.get("text")
        return isinstance(text, str) and bool(text)

    def _map_language_code_to_enum(self, language_code: object) -> Language | None:
        if not isinstance(language_code, str):
            return None
        mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "or-IN": Language.OR_IN,
            "pa-IN": Language.PA_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
            "ur-IN": Language.UR_IN,
            "kok-IN": Language.KOK_IN,
            "sd-IN": Language.SD_IN,
            "mai-IN": Language.MAI_IN,
        }
        return mapping.get(language_code)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass
