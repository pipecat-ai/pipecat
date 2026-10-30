#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp streaming text-to-speech service implementation.

Krisp's TTS WebSocket synthesizes one stream per connection: audio arrives as
untagged binary frames, in order, and the server answers every client
``end_of_sentence`` with its own ``end_of_sentence`` once that sentence's audio
has been sent. The service keeps a FIFO of sentences in flight to attribute
audio to pipecat audio contexts.
"""

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.krisp.api import (
    KRISP_REST_BASE_URL,
    KrispServiceError,
    create_session_token,
    open_session,
    session_headers,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

KRISP_TTS_URL = "wss://streaming.krisp.ai/tts"

# The API streams 24 kHz mono 16-bit PCM.
KRISP_TTS_SAMPLE_RATE = 24000

# Upper bound on how long audio is discarded after an ``interrupt`` while
# waiting for the server's acknowledging ``end_of_sentence``.
_INTERRUPT_DISCARD_TIMEOUT = 1.5

# How long to wait for the server's ``Done`` message after ``end``.
_END_OF_STREAM_TIMEOUT = 3.0


def language_to_krisp_tts_language(language: Language) -> str:
    """Convert a Language enum to a Krisp TTS language code.

    Args:
        language: The language to convert.

    Returns:
        The Krisp language code, e.g. ``en-US``.
    """
    LANGUAGE_MAP = {
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        Language.FR: "fr-FR",
        Language.FR_FR: "fr-FR",
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


@dataclass
class KrispTTSSettings(TTSSettings):
    """Settings for KrispTTSService.

    Parameters:
        speed: Playback speed: 0.9, 0.95, 1.0, 1.05 or 1.1. None uses the
            voice's configured speed.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _Sentence:
    """A sentence sent for synthesis whose audio has not fully arrived."""

    context_id: str
    text: str
    # Offset of the sentence's first audio within its context, in seconds.
    start_s: float | None = None


class KrispTTSService(WebsocketTTSService):
    """Krisp streaming text-to-speech service.

    Pushes ``TTSAudioRawFrame`` for synthesized audio and word-level
    ``TTSTextFrame``. Word times are estimated by spreading each sentence's
    spoken text across that sentence's audio. On interruption the server is
    sent ``interrupt``, and audio still in flight is discarded until the server
    acknowledges it.

    Voices are listed by :func:`pipecat.services.krisp.api.list_tts_voices`;
    pass a voice's ``name`` as the ``voice`` setting.
    """

    Settings = KrispTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = KRISP_TTS_URL,
        rest_base_url: str = KRISP_REST_BASE_URL,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Krisp TTS service.

        Args:
            api_key: Krisp master API key. It is only used to create a session
                token before each connection and never sent on the WebSocket.
            url: Krisp TTS WebSocket endpoint.
            rest_base_url: Krisp REST API base URL used to create session tokens.
            settings: Runtime-updatable settings (``voice``, ``language``,
                ``speed``, ``model``). Changing any of them reconnects, since
                they are fixed by the session handshake.
            **kwargs: Additional arguments passed to :class:`WebsocketTTSService`.
        """
        default_settings = self.Settings(
            model="krisp-tts-v1", voice="Amy", language="en-US", speed=None
        )
        if settings is not None:
            default_settings.apply_update(settings)
        if isinstance(default_settings.language, Language):
            default_settings.language = language_to_krisp_tts_language(default_settings.language)

        super().__init__(
            push_text_frames=False,
            push_start_frame=True,
            sample_rate=KRISP_TTS_SAMPLE_RATE,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._rest_base_url = rest_base_url

        self._receive_task = None
        self._connect_lock = asyncio.Lock()
        self._done_received = asyncio.Event()

        self._sentences: deque[_Sentence] = deque()
        self._context_audio_s: dict[str, float] = {}
        self._flushed_contexts: set[str] = set()
        self._awaiting_interrupt_ack = False
        self._interrupt_deadline = 0.0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Krisp TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Krisp TTS language code.

        Args:
            language: The language to convert.

        Returns:
            The Krisp language code.
        """
        return language_to_krisp_tts_language(language)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect with a new handshake.

        Args:
            delta: A :class:`TTSSettings` (or ``KrispTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed and self._websocket:
            await self._disconnect()
            await self._connect()
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """End the text stream gracefully, then disconnect.

        Sends ``end`` and waits for the server's ``Done`` message so remaining
        audio is delivered before the pipeline ends.

        Args:
            frame: The end frame.
        """
        await self._end_text_stream()
        await super().stop(frame)

    async def _end_text_stream(self):
        websocket = self._websocket
        if not websocket or websocket.state is not State.OPEN:
            return
        self._disconnecting = True
        self._done_received.clear()
        try:
            await websocket.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(self._done_received.wait(), _END_OF_STREAM_TIMEOUT)
        except TimeoutError:
            logger.warning(f"{self}: no 'Done' message within {_END_OF_STREAM_TIMEOUT}s")
        except Exception as e:
            logger.warning(f"{self}: failed to end the text stream: {e}")

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        # A receive loop that ended (e.g. the server closed the socket) can't
        # serve the new connection.
        if self._receive_task and self._receive_task.done():
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    def _build_handshake(self) -> dict[str, Any]:
        handshake: dict[str, Any] = {
            "model": self._settings.model,
            "voice": self._settings.voice,
            "word_label_exporting": False,
        }
        if self._settings.language:
            handshake["language"] = self._settings.language
        if self._settings.speed is not None:
            handshake["speed"] = self._settings.speed
        return handshake

    async def _connect_websocket(self):
        # run_tts and the receive loop's reconnect can both connect; the lock
        # makes the second caller reuse the first one's connection.
        async with self._connect_lock:
            await self._open_session()

    async def _open_session(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"Connecting to Krisp TTS ({self._url}, tokens: {self._rest_base_url})")
            self._reset_stream_state()
            token = await create_session_token(self._api_key, "tts", base_url=self._rest_base_url)
            websocket = await self._websocket_connect(
                self._url, additional_headers=session_headers(token)
            )
            try:
                ready = await open_session(websocket, self._build_handshake())
            except Exception:
                await websocket.close()
                raise
            # Exposed only once ready: no text may be sent before that.
            self._websocket = websocket
            if ready.get("warning"):
                logger.warning(f"{self}: {ready['warning']}")
            logger.debug(
                f"Connected to Krisp TTS (request_id={ready.get('request_id')}, "
                f"voice={ready.get('voice')}, language={ready.get('language')})"
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Krisp TTS: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Krisp TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            self._reset_stream_state()
            await self._call_event_handler("on_disconnected")

    def _reset_stream_state(self):
        self._sentences.clear()
        self._context_audio_s.clear()
        self._flushed_contexts.clear()
        self._awaiting_interrupt_ack = False

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send_json(self, msg: dict):
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.warning(f"{self}: failed to send {msg.get('type')}: {e}")

    async def flush_audio(self, context_id: str | None = None):
        """Mark a context complete once its in-flight sentences finish.

        Args:
            context_id: The context to flush. If None, uses the active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id:
            return
        self._flushed_contexts.add(flush_id)
        await self._maybe_finish_context(flush_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Stop generation on the server and discard audio already in flight.

        Args:
            context_id: The interrupted context.
        """
        await self.stop_all_metrics()
        if self._sentences:
            await self._send_json({"type": "interrupt"})
            # The server acknowledges with one end_of_sentence covering
            # everything that was pending.
            self._awaiting_interrupt_ack = True
            self._interrupt_deadline = time.monotonic() + _INTERRUPT_DISCARD_TIMEOUT
        self._sentences.clear()
        self._flushed_contexts.discard(context_id)
        self._context_audio_s.pop(context_id, None)
        await super().on_audio_context_interrupted(context_id)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                await self._on_audio(message)
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON message: {message}")
                continue

            msg_type = msg.get("type")
            if msg_type == "end_of_sentence":
                await self._on_end_of_sentence(msg)
            elif msg_type == "Done":
                self._done_received.set()
            elif msg_type == "error":
                error = KrispServiceError.from_message(msg)
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Krisp TTS error: {error}", exception=error)
            # "info" and "all_processed" carry nothing the service needs.

    def _discarding_interrupted_audio(self) -> bool:
        if self._awaiting_interrupt_ack and time.monotonic() >= self._interrupt_deadline:
            logger.warning(f"{self}: no interrupt acknowledgement, resuming audio")
            self._awaiting_interrupt_ack = False
        return self._awaiting_interrupt_ack

    async def _on_audio(self, audio: bytes):
        if self._discarding_interrupted_audio() or not self._sentences:
            return
        sentence = self._sentences[0]
        ctx_id = sentence.context_id
        if not self.audio_context_available(ctx_id):
            return

        offset = self._context_audio_s.get(ctx_id, 0.0)
        if sentence.start_s is None:
            sentence.start_s = offset
            await self.stop_ttfb_metrics()
        self._context_audio_s[ctx_id] = offset + len(audio) / (2 * self.sample_rate)

        frame = TTSAudioRawFrame(
            audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=ctx_id
        )
        await self.append_to_audio_context(ctx_id, frame)

    async def _on_end_of_sentence(self, msg: dict):
        if self._awaiting_interrupt_ack:
            self._awaiting_interrupt_ack = False
            return
        if not self._sentences:
            return

        sentence = self._sentences.popleft()
        ctx_id = sentence.context_id
        if self.audio_context_available(ctx_id) and sentence.start_s is not None:
            end_s = self._context_audio_s.get(ctx_id, sentence.start_s)
            text = msg.get("spoken_text") or sentence.text
            word_times = _spread_words(text, sentence.start_s, end_s)
            if word_times:
                await self.add_word_timestamps(word_times, ctx_id)
        await self._maybe_finish_context(ctx_id)

    async def _maybe_finish_context(self, ctx_id: str):
        if ctx_id not in self._flushed_contexts:
            return
        if any(s.context_id == ctx_id for s in self._sentences):
            return
        self._flushed_contexts.discard(ctx_id)
        self._context_audio_s.pop(ctx_id, None)
        if self.audio_context_available(ctx_id):
            await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
            await self.remove_audio_context(ctx_id)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send a sentence to Krisp for synthesis.

        Args:
            text: The text to synthesize.
            context_id: The audio context the resulting audio belongs to.

        Yields:
            None on success (audio arrives on the receive task), or an
            ``ErrorFrame`` and ``TTSStoppedFrame`` on failure.
        """
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        try:
            websocket = self._get_websocket()
            await websocket.send(json.dumps({"type": "text", "text": text}))
            await websocket.send(json.dumps({"type": "end_of_sentence"}))
            self._sentences.append(_Sentence(context_id=context_id, text=text))
            await self.start_tts_usage_metrics(text)
        except Exception as e:
            yield ErrorFrame(error=f"Krisp TTS send failed: {e}")
            yield TTSStoppedFrame(context_id=context_id)
            await self._disconnect()
            await self._connect()
            return
        yield None


def _spread_words(text: str, start_s: float, end_s: float) -> list[tuple[str, float]]:
    """Estimate word start times by spreading words over a span by length."""
    words = text.split()
    total_chars = sum(len(w) for w in words)
    if not total_chars:
        return []
    duration = max(end_s - start_s, 0.0)
    word_times = []
    chars_before = 0
    for word in words:
        word_times.append((word, start_s + duration * chars_before / total_chars))
        chars_before += len(word)
    return word_times
