#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Meta speech-to-text service implementation.

This module provides integration with Meta's Muse Voice realtime transcription
WebSocket API documented at
https://dev.meta.ai/docs/api-reference/voice/realtime.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import META_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

META_STT_WS_URL = "wss://api.meta.ai/v1/asr/realtime"
META_STT_MODEL = "muse-voice-transcribe-1.0"

# The server drops the connection if the handshake frame doesn't arrive within
# ten seconds of the socket opening; it answers well inside that window.
HANDSHAKE_TIMEOUT = 10.0

# The engine's native rate. Anything else is resampled to 16 kHz, the only
# other rate the API accepts.
META_NATIVE_SAMPLE_RATE = 24000
META_FALLBACK_SAMPLE_RATE = 16000


_LANGUAGE_MAP = {
    Language.AR: "Arabic",
    Language.BN: "Bengali",
    Language.DE: "German",
    Language.EN: "English",
    Language.ES: "Spanish",
    Language.FR: "French",
    Language.HE: "Hebrew",
    Language.HI: "Hindi",
    Language.ID: "Indonesian",
    Language.IT: "Italian",
    Language.JA: "Japanese",
    Language.KN: "Kannada",
    Language.KO: "Korean",
    Language.MR: "Marathi",
    Language.MS: "Malay",
    Language.NL: "Dutch",
    Language.PL: "Polish",
    Language.PT: "Portuguese",
    Language.TA: "Tamil",
    Language.TE: "Telugu",
    Language.TH: "Thai",
    Language.TL: "Tagalog",
    Language.TR: "Turkish",
    Language.VI: "Vietnamese",
    Language.ZH: "Mandarin Chinese",
}

# Settings store the service's language name; frames carry the enum it came from.
_LANGUAGE_NAMES = {name: language for language, name in _LANGUAGE_MAP.items()}


def language_to_meta_language(language: Language) -> str:
    """Convert a Language enum to the Meta STT language name.

    Meta biases recognition on English language *names* rather than codes, e.g.
    ``"English"`` or ``"Mandarin Chinese"``.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language name. If ``language`` is not in the
        verified mapping, falls back to the base language code (e.g., ``en``
        from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``). Meta ignores a bias it
        doesn't recognize, so an unmapped language degrades to auto-detection.
    """
    return resolve_language(language, _LANGUAGE_MAP, use_base_code=True)


@dataclass
class MetaSTTSettings(STTSettings):
    """Settings for MetaSTTService.

    Parameters:
        language_bias: Languages to bias recognition toward, for speech that
            switches between them. When set, it replaces the single ``language``
            bias. Leave both unset to let the model detect the language.
        mode: Segmentation mode. ``"ENDPOINTING"`` has the model mark turn
            boundaries; ``"PUSH_TO_TALK"`` leaves them to the client;
            ``"DIARIZATION"`` attributes turns to speakers.
        keywords: Terms to bias recognition toward. Biasing raises the odds a
            term is recognized but doesn't guarantee its spelling.
        emit_audio_progress: Whether the server reports progress for every
            processed audio chunk.
        zdr_override: Forces metadata-only logging when True, or allows content
            retention when False. Leave unset to use the account's setting.
    """

    language_bias: list[Language] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    mode: str | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keywords: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emit_audio_progress: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    zdr_override: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class MetaSTTService(WebsocketSTTService):
    """Meta Muse Voice real-time speech-to-text service.

    Streams audio to Meta's realtime transcription WebSocket and emits interim
    and final transcription frames. The API key travels in the handshake frame
    rather than an HTTP header, and the session's configuration is fixed once
    the server accepts that handshake, so a settings change reconnects.

    In the default ``ENDPOINTING`` mode the server marks turn boundaries
    itself, emitting ``speechStart``, a run of cumulative partial transcripts,
    ``speechEnd``, and finally ``speechComplete`` with the post-processed text
    that becomes the :class:`~pipecat.frames.frames.TranscriptionFrame`. Those
    boundaries segment transcripts only: Pipecat's own VAD and turn strategies
    still decide when the user's turn ends.

    Sessions are capped at 60 minutes. The server closes the socket when a
    session reaches that limit, and the base class reconnects into a fresh one.
    """

    Settings = MetaSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = META_STT_WS_URL,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = META_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Meta STT service.

        Args:
            api_key: Meta Model API key, sent as a Bearer token in the handshake.
            url: WebSocket endpoint URL. Defaults to
                ``wss://api.meta.ai/v1/asr/realtime``.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate. Audio at anything other than 24000 is resampled to
                16000, the only other rate the API accepts.
            settings: Runtime-updatable settings overriding defaults.
            ttfs_p99_latency: P99 latency from speech end to final transcript in
                seconds. Override for your deployment. See
                https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        default_settings = self.Settings(
            model=META_STT_MODEL,
            language=Language.EN,
            language_bias=None,
            mode="ENDPOINTING",
            keywords=None,
            emit_audio_progress=False,
            zdr_override=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url

        self._receive_task: asyncio.Task | None = None
        self._session_ready = asyncio.Event()
        self._session_id: str | None = None
        self._turn_id: int | None = None
        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to the Meta STT language name."""
        return language_to_meta_language(language)

    @property
    def _send_sample_rate(self) -> int:
        """The rate audio is sent at, which the API limits to 24 kHz or 16 kHz."""
        if self.sample_rate == META_NATIVE_SAMPLE_RATE:
            return META_NATIVE_SAMPLE_RATE
        return META_FALLBACK_SAMPLE_RATE

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta and reconnect to apply changes.

        Meta fixes a session's configuration when it accepts the handshake, so
        any change needs a fresh connection.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        await self._request_reconnect()
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the speech-to-text service."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech-to-text service."""
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Forward raw audio bytes to the Meta STT WebSocket.

        Transcription frames are pushed from the receive task, not yielded from
        this coroutine.
        """
        if self.sample_rate != self._send_sample_rate:
            audio = await self._resampler.resample(audio, self.sample_rate, self._send_sample_rate)

        if (
            audio
            and self._websocket
            and self._websocket.state is State.OPEN
            and self._session_ready.is_set()
        ):
            try:
                await self._websocket.send(audio)
            except Exception as e:
                await self.push_error(error_msg=f"Meta STT send failed: {e}", exception=e)
        yield None

    def _build_handshake(self) -> dict[str, Any]:
        """Build the handshake frame that opens and configures the session."""
        s = self._settings

        handshake: dict[str, Any] = {
            "authorization": {"accessToken": f"Bearer {self._api_key}"},
            "audioEncoding": f"PCM_{self._send_sample_rate // 1000}KHZ",
            "model": assert_given(s.model),
            "mode": assert_given(s.mode),
            # Every partial carries the complete current hypothesis, which is
            # what the interim frames below replace each other with. DELTA
            # would make each partial a fragment instead.
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": assert_given(s.emit_audio_progress),
        }

        language_bias = assert_given(s.language_bias)
        if language_bias:
            handshake["languageBias"] = [language_to_meta_language(lang) for lang in language_bias]
        elif s.language is not None:
            handshake["languageBias"] = [assert_given(s.language)]

        keywords = assert_given(s.keywords)
        if keywords:
            handshake["keywords"] = keywords

        zdr_override = assert_given(s.zdr_override)
        if zdr_override is not None:
            handshake["zdrOverride"] = zdr_override

        return handshake

    async def _connect(self):
        """Establish the WebSocket connection and start the receive task."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Tear down the WebSocket connection and cancel the receive task."""
        await super()._disconnect()
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                # Half-closes the input so the server flushes what it still owes
                # us before the socket goes away.
                await self._websocket.send(json.dumps({"type": "endStream"}))
        except Exception as e:
            logger.debug(f"{self} error sending endStream during disconnect: {e}")

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open a WebSocket connection and complete the Meta STT handshake."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Meta STT WebSocket")
            self._session_ready.clear()
            self._session_id = None

            if self.sample_rate != self._send_sample_rate:
                logger.debug(
                    f"{self} resampling audio from {self.sample_rate} to "
                    f"{self._send_sample_rate} for Meta STT"
                )

            websocket = await self._websocket_connect(
                f"{self._url}?sessionId=stream-{uuid.uuid4()}"
            )
            self._websocket = websocket

            # The credential travels in the handshake frame, so a rejected key
            # surfaces here rather than as a refused HTTP upgrade. Reading the
            # acknowledgement before the receive task starts is what turns that
            # rejection into an error instead of a silently idle connection.
            await websocket.send(json.dumps(self._build_handshake()))
            ack = json.loads(await asyncio.wait_for(websocket.recv(), HANDSHAKE_TIMEOUT))

            if "sessionId" not in ack:
                message = ack.get("message", ack)
                await websocket.close()
                self._websocket = None
                await self.push_error(
                    error_msg=f"Meta STT rejected the session: {message}",
                    exception=Exception(message),
                    category=ErrorCategory.AUTHENTICATION,
                    force_treat_as_permanent=True,
                )
                return

            self._session_id = ack["sessionId"]
            self._session_ready.set()
            await self._call_event_handler("on_connected")
            logger.debug(f"{self} connected to Meta STT WebSocket (session {self._session_id})")
        except Exception as e:
            self._websocket = None
            self._session_ready.clear()
            await self.push_error(error_msg=f"Unable to connect to Meta STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from Meta STT WebSocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Meta STT websocket: {e}", exception=e)
        finally:
            self._websocket = None
            self._session_ready.clear()
            self._session_id = None
            self._turn_id = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Receive and dispatch Meta STT WebSocket messages."""
        if not self._websocket:
            raise Exception("Websocket not connected")
        async for message in self._websocket:
            if isinstance(message, bytes):
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message: {message}")
                continue
            await self._handle_message(data)

    async def _handle_message(self, message: dict[str, Any]):
        """Branch on Meta STT event type."""
        msg_type = message.get("type")

        if msg_type == "transcript":
            await self._handle_transcript(message)
        elif msg_type == "speechComplete":
            self._turn_id = message.get("turnId")
            await self._push_final_transcript(message)
        elif msg_type in ("speechStart", "speechEnd"):
            self._turn_id = message.get("turnId")
            logger.trace(f"{self} Meta STT {msg_type} (turn {self._turn_id})")
        elif msg_type == "speaker":
            # Diarization marks a possible new speaker mid-turn rather than a
            # boundary, so the label only rides along on the transcript result.
            logger.trace(f"{self} Meta STT speaker {message.get('label')}")
        elif msg_type == "audioProgress":
            # Only arrives when emit_audio_progress is on, and says nothing the
            # transcript events don't already carry.
            pass
        elif msg_type == "error":
            await self.push_error(
                error_msg=f"Meta STT error: {message.get('message', message)}",
                exception=Exception(message),
            )
        else:
            logger.debug(f"{self} unhandled Meta STT message: {message}")

    async def _handle_transcript(self, message: dict[str, Any]):
        """Push a partial hypothesis, or a final in the client-delimited modes."""
        text = message.get("transcript", "")
        if not text:
            return

        # In ENDPOINTING mode the model closes the turn itself and the
        # punctuated text arrives on speechComplete, so every transcript here is
        # still a hypothesis. The other modes have no such event.
        if message.get("final") and self._settings.mode != "ENDPOINTING":
            await self._push_final_transcript(message, text=text)
            return

        await self.push_frame(
            InterimTranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                self._language_for_frame(),
                result=message,
            )
        )

    async def _push_final_transcript(self, message: dict[str, Any], *, text: str | None = None):
        """Push the transcript that closes a turn."""
        text = text if text is not None else message.get("transcript", "")
        if not text:
            return

        # Report usage before the transcription frame so tracing can attach it
        # to the STT span the frame closes.
        await self.emit_stt_usage_metrics()
        language = self._language_for_frame()
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=message,
                finalized=True,
            )
        )
        await self._trace_transcription(text, True, language)

    def _language_for_frame(self) -> Language | None:
        """Return a Language enum suitable for transcription frames.

        Meta reports no language of its own, so frames carry whichever language
        the session was biased toward, or None when it was left to detection.
        """
        bias = assert_given(self._settings.language_bias)
        if bias:
            return bias[0] if len(bias) == 1 else None

        lang = self._settings.language
        if isinstance(lang, Language):
            return lang
        if isinstance(lang, str):
            return _LANGUAGE_NAMES.get(lang)
        return None

    @traced_stt
    async def _trace_transcription(
        self, transcript: str, is_final: bool, language: Language | None
    ):
        """Record transcription event for tracing."""
        pass
