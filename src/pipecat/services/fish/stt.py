#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio realtime speech recognition with server-side turn detection."""

import asyncio
import json
import re
from collections import deque
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field
from websockets.exceptions import ConnectionClosed, InvalidStatus

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserBackchannelFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin
from pipecat.turns.user_stop.external_user_turn_stop_strategy import ExternalUserTurnStopStrategy
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import ormsgpack
except ModuleNotFoundError as e:
    logger.error('In order to use Fish Audio, you need to `uv add "pipecat-ai[fish]"`.')
    raise ImportError(f"Missing module: {e}") from e


def _describe_refusal(exception: Exception) -> str | None:
    """Read Fish's explanation and request ID from a refused handshake."""
    if not isinstance(exception, InvalidStatus):
        return None
    response = exception.response
    try:
        reason = json.loads(response.body).get("message")
    except (TypeError, ValueError, AttributeError):
        reason = None
    message = f"Fish refused the connection (HTTP {response.status_code})"
    if reason:
        message += f": {reason}"
    request_id = response.headers.get("x-request-id")
    if request_id:
        message += f" (request_id={request_id})"
    return message


class FishAudioContextMessage(BaseModel):
    """A prior conversation message supplied to Fish Audio.

    Parameters:
        role: The participant who spoke.
        text: Spoken text, up to 2,000 characters.
    """

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=2000)


class FishAudioSTTService(EagerEndOfTurnSTTServiceMixin, WebsocketSTTService):
    """Fish Audio realtime ASR in automatic turn-detection mode.

    Send continuous mono PCM, including silence. Fish decides when the user
    takes the turn. Omit local VAD for server-controlled turn boundaries;
    Pipecat holds turn-end decisions while local VAD still detects speech.
    Committed acknowledgments are emitted as :class:`UserBackchannelFrame`
    rather than transcriptions. Only ``user_turn_taken`` starts a user turn.

    Bot speaking frames report playback to Fish. This covers actual playback,
    but not the preceding synthesis wait or gaps between playback runs.
    Completed assistant replies are supplied as conversation context. A new
    connection restores the most recent 32 user and assistant context items;
    audio received while disconnected is dropped.

    The ``on_server_event(service, message)`` event exposes decoded Fish
    messages, including early backchannel detection and raw sound annotations.
    These notifications do not themselves modify the conversation.
    """

    Settings = STTSettings
    _CONNECTION_TIMEOUT = 10.0
    _SEND_TIMEOUT = 5.0
    _DRAIN_TIMEOUT = 2.0
    _SEND_QUEUE_SIZE = 128
    _SAMPLE_RATES = {8000, 12000, 16000, 22050, 24000, 32000, 44100, 48000}

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.fish.audio/v1/asr/realtime",
        sample_rate: int | None = None,
        settings: STTSettings | None = None,
        terms: list[str] | None = None,
        context_history: list[FishAudioContextMessage] | None = None,
        drafts: bool = False,
        silence_timeout_ms: int | None = None,
        should_interrupt: bool = True,
        enable_eager_end_of_turn: bool = False,
        transport_destination: str | None = None,
        **kwargs,
    ):
        """Initialize the realtime ASR service.

        Args:
            api_key: Fish Audio API key.
            url: Realtime ASR WebSocket endpoint.
            sample_rate: PCM sample rate, or the pipeline input rate.
            settings: Initial model settings. Language is detected automatically.
                Runtime settings updates are not supported.
            terms: Up to 100 recognition hints of at most 32 characters each.
            context_history: Prior conversation, oldest first, up to 32 items.
            drafts: Request interim transcripts. Only drafts from an accepted
                user turn are forwarded to the pipeline.
            silence_timeout_ms: Server silence safety timeout, 3,000–60,000 ms.
            should_interrupt: Let the recommended turn strategy interrupt on
                ``user_turn_taken``.
            enable_eager_end_of_turn: Prepare replies to predicted turn ends
                using Pipecat's speculative response handling.
            transport_destination: Output destination whose speaking frames
                describe the agent heard by this user.
            **kwargs: Additional arguments for :class:`WebsocketSTTService`.
        """
        default_settings = self.Settings(model="transcribe-stream-1", language=None)
        if settings is not None:
            default_settings.apply_update(settings)
        if default_settings.language is not None:
            raise ValueError("Fish realtime ASR detects language automatically")
        if terms and (len(terms) > 100 or any(not 1 <= len(term) <= 32 for term in terms)):
            raise ValueError("Fish accepts up to 100 terms of 1–32 characters each")
        if context_history and len(context_history) > 32:
            raise ValueError("Fish accepts up to 32 context history items")
        if silence_timeout_ms is not None and not 3000 <= silence_timeout_ms <= 60000:
            raise ValueError("silence_timeout_ms must be between 3000 and 60000 in auto mode")

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            enable_eager_end_of_turn=enable_eager_end_of_turn,
            **kwargs,
        )
        self._api_key = api_key
        self._url = url
        self._terms = list(terms or [])
        self._history: deque[dict[str, str]] = deque(
            (item.model_dump() for item in context_history or []), maxlen=32
        )
        self._drafts = drafts
        self._silence_timeout_ms = silence_timeout_ms
        self._should_interrupt = should_interrupt
        self._transport_destination = transport_destination
        self._agent_speaking = False
        self._language: Language | None = None
        self._request_id: str | None = None
        self._user_turn_index: int | None = None
        self._last_committed_turn_index = -1
        self._ready = asyncio.Event()
        self._session_done = asyncio.Event()
        self._receive_task: asyncio.Task | None = None
        self._send_task: asyncio.Task | None = None
        self._send_queue: asyncio.Queue[tuple[Any, dict[str, Any]]] = asyncio.Queue(
            maxsize=self._SEND_QUEUE_SIZE
        )
        self._pending_context: list[dict[str, str]] = []
        self._graceful_close = False
        self._register_event_handler("on_server_event")

    @property
    def supports_ttfs(self) -> bool:
        """Turn boundaries are provided by the server."""
        return False

    def can_generate_metrics(self) -> bool:
        """Report transcription latency and submitted audio usage."""
        return True

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend strategies driven by Fish's explicit turn decisions."""
        frame = super().service_metadata_frame()
        strategies = self.recommended_user_turn_strategies(
            enable_interruptions=self._should_interrupt
        )
        # A commit is terminal even when recognition produced no words.
        for strategy in strategies.stop or []:
            assert isinstance(strategy, ExternalUserTurnStopStrategy)
            strategy.wait_for_transcript = False
        frame.user_turn_strategies = strategies
        return frame

    async def setup(self, setup: FrameProcessorSetup):
        """Validate the input rate and open the ASR connection."""
        await super().setup(setup)
        if self.sample_rate not in self._SAMPLE_RATES:
            raise ValueError(f"Unsupported Fish realtime ASR sample rate: {self.sample_rate}")
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Drain final server results before ending the session."""
        self._graceful_close = True
        try:
            await super().stop(frame)
        finally:
            self._graceful_close = False

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Report runtime changes as unsupported and retain session settings."""
        changed = deepcopy(self._settings).apply_update(delta)
        self._warn_unhandled_updated_settings(changed.keys())
        return {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Observe playback notifications and forward frames normally."""
        await super().process_frame(frame, direction)
        if (
            direction != FrameDirection.UPSTREAM
            or frame.transport_destination != self._transport_destination
        ):
            return
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            speaking = isinstance(frame, BotStartedSpeakingFrame)
            if self._agent_speaking != speaking:
                self._agent_speaking = speaking
                await self._queue_message({"type": "agent_speaking", "speaking": speaking})

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection):
        """Require PCM matching the mono stream declared in the session config."""
        if not self.is_usable:
            return
        if frame.num_channels != 1 or frame.sample_rate != self.sample_rate:
            await self.push_error_frame(
                ErrorFrame(
                    f"Fish realtime ASR requires mono PCM at {self.sample_rate} Hz; "
                    f"received {frame.num_channels} channels at {frame.sample_rate} Hz"
                ),
                force_treat_as_permanent=True,
            )
            return
        if not self._ready.is_set() or self._disconnecting:
            return
        await super().process_audio_frame(frame, direction)

    def _append_history(self, role: str, text: str) -> list[dict[str, str]]:
        items = [{"role": role, "text": text[i : i + 2000]} for i in range(0, len(text), 2000)]
        self._history.extend(items)
        return items

    async def _process_assistant_turn(self, text: str) -> None:
        """Supply completed spoken replies as recognition context."""
        items = self._append_history("assistant", text)
        if not self._ready.is_set():
            self._pending_context.extend(items)
            self._pending_context = self._pending_context[-32:]
            return
        for offset in range(0, len(items), 32):
            await self._queue_message({"type": "context_add", "items": items[offset : offset + 32]})

    def _session_config(self) -> dict[str, Any]:
        turn_detection: dict[str, Any] = {
            "mode": "auto",
            "eager": self.eager_end_of_turn_enabled,
        }
        if self._silence_timeout_ms is not None:
            turn_detection["silence_timeout_ms"] = self._silence_timeout_ms
        return {
            "type": "config",
            "audio": {"format": "pcm_s16le", "sample_rate": self.sample_rate, "channels": 1},
            "turn_detection": turn_detection,
            "results": {"drafts": self._drafts, "diarize": False},
            "context": {"history": list(self._history), "terms": self._terms},
        }

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if not self._send_task:
            self._send_task = self.create_task(self._send_messages(), name="fish-asr-send")
        if not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error), name="fish-asr-receive"
            )

    async def _connect_websocket(self):
        try:
            self._ready.clear()
            self._session_done.clear()
            self._last_committed_turn_index = -1
            self._request_id = None
            websocket = await self._websocket_connect(
                self._url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "model": self._settings.model,
                },
            )
            self._websocket = websocket
            if websocket.response:
                self._request_id = websocket.response.headers.get("x-request-id")
            config = self._session_config()
            self._pending_context.clear()
            await websocket.send(ormsgpack.packb(config))
        except Exception as e:
            await self._report_error(
                ErrorFrame(
                    _describe_refusal(e) or f"Fish realtime ASR connection failed: {e}", exception=e
                ),
            )
            if self._websocket:
                await self._websocket.close()
            self._websocket = None

    async def _disconnect(self):
        await super()._disconnect()
        if self._graceful_close and self._ready.is_set() and self._websocket:
            try:
                async with asyncio.timeout(self._DRAIN_TIMEOUT):
                    await self._send_queue.join()
                    await self._websocket.send(ormsgpack.packb({"type": "audio_done"}))
                    await self._session_done.wait()
            except (TimeoutError, ConnectionClosed, ConnectionError):
                logger.debug("Fish realtime ASR final results did not finish before shutdown")
        for name in ("_send_task", "_receive_task"):
            task = getattr(self, name)
            if task:
                await self.cancel_task(task)
                setattr(self, name, None)
        await self._disconnect_websocket()

    def _clear_send_queue(self):
        while not self._send_queue.empty():
            self._send_queue.get_nowait()
            self._send_queue.task_done()

    async def _disconnect_websocket(self):
        self._ready.clear()
        self._clear_send_queue()
        websocket, self._websocket = self._websocket, None
        if websocket:
            await websocket.close()
            await self._call_event_handler("on_disconnected")

    async def _queue_message(self, message: dict[str, Any]) -> None:
        if not self._ready.is_set() or self._disconnecting:
            return
        try:
            self._send_queue.put_nowait((self._websocket, message))
        except asyncio.QueueFull:
            # A partial audio stream cannot be repaired by replaying old chunks.
            self._ready.clear()
            await self.push_error("Fish realtime ASR send queue filled; reconnecting")
            if self._websocket:
                self.create_task(self._websocket.close(), name="fish-asr-close-backpressure")

    async def _send_messages(self):
        while True:
            websocket, message = await self._send_queue.get()
            try:
                if websocket is not self._websocket or not self._ready.is_set():
                    continue
                async with asyncio.timeout(self._SEND_TIMEOUT):
                    await websocket.send(ormsgpack.packb(message))
            except Exception as e:
                if websocket is self._websocket:
                    self._ready.clear()
                    await self.push_error(f"Fish realtime ASR send failed: {e}", exception=e)
                await websocket.close()
            finally:
                self._send_queue.task_done()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Queue a PCM chunk without waiting for connection or network I/O."""
        await self._queue_message({"type": "audio", "data": audio})
        yield None

    async def _send_keepalive(self, silence: bytes):
        """Keep an idle connection alive without synthesizing input audio."""
        await self._queue_message({"type": "keepalive"})

    async def _receive_messages(self):
        if self._websocket is None:
            raise ConnectionError("Fish realtime ASR is not connected")
        messages = aiter(self._websocket)
        try:
            if not await self._receive_until_confirmed(
                messages, self._ready, self._process_message, self._CONNECTION_TIMEOUT
            ):
                return
            async for message in messages:
                await self._process_message(message)
        finally:
            self._ready.clear()
            if not self._disconnecting:
                await self._cancel_eager_end_of_turn()
                await self._end_user_turn()

    async def _process_message(self, message: str | bytes):
        if not isinstance(message, bytes):
            raise ValueError("Fish realtime ASR messages must use binary MessagePack frames")
        await self._handle_message(ormsgpack.unpackb(message))

    @staticmethod
    def _spoken_text(text: str) -> str:
        return " ".join(re.sub(r"\[[^\[\]]*\]", "", text).split())

    def _message_language(self, data: dict[str, Any]) -> Language | None:
        code = data.get("language_code")
        if code:
            try:
                self._language = Language(code)
            except ValueError:
                logger.debug(f"Unknown Fish transcription language: {code}")
        return self._language

    async def _end_user_turn(self):
        if self._user_turn_index is not None:
            self._user_turn_index = None
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _handle_message(self, data: dict[str, Any]):
        if not isinstance(data, dict) or not isinstance(data.get("type"), str):
            raise ValueError("Expected a Fish realtime ASR message map with a type")
        kind = data["type"]
        if kind == "session_started":
            self._request_id = data.get("request_id") or self._request_id
            logger.info(f"Fish realtime ASR connected (request_id={self._request_id})")
            self._ready.set()
            if self._agent_speaking:
                await self._queue_message({"type": "agent_speaking", "speaking": True})
            if self._pending_context:
                await self._queue_message({"type": "context_add", "items": self._pending_context})
                self._pending_context = []
            await self._call_event_handler("on_connected")
        elif kind == "user_turn_taken":
            if self._user_turn_index != data["turn_index"]:
                self._user_turn_index = data["turn_index"]
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif kind == "draft":
            if self._user_turn_index == data["turn_index"]:
                text = self._spoken_text(data.get("text", ""))
                if text:
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text,
                            self._user_id,
                            time_now_iso8601(),
                            self._message_language(data),
                            result=data,
                        )
                    )
        elif kind == "eager_end_of_turn":
            text = self._spoken_text(data.get("text", ""))
            if text and self._user_turn_index == data["turn_index"]:
                await self._push_eager_end_of_turn(
                    text, user_id=self._user_id, language=self._message_language(data), result=data
                )
        elif kind == "turn_resumed":
            await self._cancel_eager_end_of_turn()
        elif kind == "committed":
            await self._handle_committed(data)
        elif kind == "session_done":
            self._session_done.set()
        elif kind == "error":
            # recoverable describes the current session. Only a rejected
            # configuration makes a fresh session fail for the same reason.
            error_type = data.get("error_type")
            category = (
                ErrorCategory.INVALID_REQUEST
                if error_type in ("invalid_config", "unsupported_audio")
                else ErrorCategory.UNKNOWN
            )
            request_id = data.get("request_id") or self._request_id
            await self.push_error_frame(
                ErrorFrame(
                    f"Fish realtime ASR {error_type} ({data.get('code')}): "
                    f"{data.get('message')} (request_id={request_id})",
                    category=category,
                ),
            )
        await self._call_event_handler("on_server_event", data)

    async def _handle_committed(self, data: dict[str, Any]):
        turn_index = data["turn_index"]
        if turn_index <= self._last_committed_turn_index:
            return
        self._last_committed_turn_index = turn_index
        await self.emit_stt_usage_metrics()
        text = self._spoken_text(data.get("text", ""))
        if data.get("user_turn", True):
            self._append_history("user", data.get("text", ""))
            if text:
                self._clear_eager_end_of_turn()
                await self.push_frame(
                    TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        self._message_language(data),
                        result=data,
                        finalized=True,
                    )
                )
                await self._handle_transcription(text, True, self._message_language(data))
            else:
                await self._cancel_eager_end_of_turn()
            await self._end_user_turn()
        elif data.get("turn_kind") == "backchannel":
            await self.push_frame(UserBackchannelFrame(text, self._user_id, time_now_iso8601()))

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record a completed transcription in the STT trace."""
        pass
