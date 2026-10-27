#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AssemblyAI text-to-speech service implementation.

This module provides integration with AssemblyAI's Streaming TTS WebSocket API,
which synthesizes text into streamed PCM audio with per-word timings.
"""

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import is_given

# Audrey, a voice from AssemblyAI's public voice library.
_DEFAULT_VOICE_ID = "49c443a4-8a1d-5f7e-b70b-98bdbd9823c3"

# Longest text one Generate message accepts, when Begin doesn't report a limit.
_MAX_GENERATE_CHARS = 2000

# How long before a session's expires_at to replace it between requests, at
# most half the session's length. The server stops reading text at expires_at,
# which would cut off a reply that straddles it.
_SESSION_RENEWAL_MARGIN_S = 120.0

# How often to send KeepAlive. Every session closes after 120 s without a frame
# either way, and a bot can be silent that long while the user speaks.
_KEEPALIVE_INTERVAL_S = 30.0

# How long to wait, after a turn's last FlushDone, for the WordBoundaries frame
# that trails it. It is best-effort, so the turn closes without it after this.
_WORD_BOUNDARIES_GRACE_S = 1.0

# Retryable errors that end a session in the normal course of things.
_ROUTINE_ERROR_TYPES = frozenset({"inactivity_timeout", "session_expired", "service_restart"})


def language_to_assemblyai_tts_language(language: Language) -> str:
    """Convert a Language enum to an AssemblyAI Streaming TTS language.

    The TTS API takes the full English name of the language (``"spanish"``),
    unlike the STT APIs, which take short codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding AssemblyAI TTS language name.
    """
    LANGUAGE_MAP = {
        Language.DE: "german",
        Language.EN: "english",
        Language.ES: "spanish",
        Language.FR: "french",
        Language.IT: "italian",
        Language.PT: "portuguese",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class AssemblyAITTSSettings(TTSSettings):
    """Settings for AssemblyAITTSService.

    ``voice`` is the API's ``voice_id``: any voice by its ID, an AssemblyAI
    voice by its library slug, such as ``aai-jane``, or a cloned voice by the
    slug the AssemblyAI dashboard shows. It defaults to Audrey
    (``_DEFAULT_VOICE_ID``); ``None`` leaves the voice to the server.
    ``language`` is the language the session speaks, and ``model`` names the
    model, such as ``assemblyai-tts-1``. Unset, the session speaks the voice's
    own language and runs on that language's default model.
    """

    pass


@dataclass
class _ContextState:
    """Bookkeeping for one audio context.

    Parameters:
        audio_ms: Playback length of the context's finished requests, where
            the next request's audio starts within the context.
        turn_done: Whether the turn has ended, so no more text will be sent.
        awaiting_words: Flush IDs whose WordBoundaries frame is still due,
            each with where its request's audio starts within the context.
        grace_task: Task that closes the context if WordBoundaries never come.
    """

    audio_ms: float = 0.0
    turn_done: bool = False
    awaiting_words: dict[int, float] = field(default_factory=dict)
    grace_task: asyncio.Task | None = None


class AssemblyAITTSService(WebsocketTTSService):
    """AssemblyAI Streaming TTS service.

    Streams text to AssemblyAI over a single WebSocket session and plays the
    audio as it arrives. Each sentence is sent as its own request (a
    ``Generate`` followed by a ``Flush``). With ``TextAggregationMode.TOKEN``
    the turn's text streams as it is generated and is flushed when the turn
    ends. Interruptions send ``Cancel``, which keeps the session open, and
    per-word timings drive word-level ``TTSTextFrame`` output.

    The session is configured at connect time, so changing the voice,
    language or model reconnects. A session lasts at most an hour; the service
    replaces it between requests before it expires.

    Example::

        # Speaks in Audrey, the default voice.
        tts = AssemblyAITTSService(api_key=os.getenv("ASSEMBLYAI_API_KEY"))

        # Another voice, by its ID: Keegan.
        tts = AssemblyAITTSService(
            api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            settings=AssemblyAITTSService.Settings(voice="987db62e-a2c2-569a-9c4d-82460cdea1bd"),
        )
    """

    Settings = AssemblyAITTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://tts.us.assemblyai.com/v1/ws",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the AssemblyAI TTS service.

        Args:
            api_key: AssemblyAI API key. A temporary token opens a single
                session, so it can't carry the service's reconnects and
                session renewals.
            url: Streaming TTS WebSocket URL. Use
                ``wss://tts.eu.assemblyai.com/v1/ws`` to process text and audio
                in the EU. A cloned voice works only on the host of the zone it
                was created in.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline's
                output rate.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent WebsocketTTSService.
        """
        default_settings = self.Settings(model=None, voice=_DEFAULT_VOICE_ID, language=None)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._receive_task: asyncio.Task | None = None

        # Session state reported by Begin. A session ID means Begin arrived.
        self._session_id: str | None = None
        self._renew_at: float | None = None
        self._max_generate_chars = _MAX_GENERATE_CHARS
        self._word_boundaries = False
        # The wait an Error asked for before the next connection attempt.
        self._reconnect_delay = 0.0

        # The server numbers requests from 0 per session, and each Flush and
        # Cancel takes the next number. This is the number of the request text
        # is being written to now.
        self._next_flush_id = 0
        # Flush ID -> context, for requests whose FlushDone is still due.
        self._requests: dict[int, str] = {}
        self._contexts: dict[str, _ContextState] = {}
        # Flush ID -> context, for the WordBoundaries that follows a FlushDone.
        self._word_contexts: dict[int, str] = {}

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as AssemblyAI TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to an AssemblyAI TTS language name.

        Args:
            language: The language to convert.

        Returns:
            The AssemblyAI TTS language name.
        """
        return language_to_assemblyai_tts_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Connecting here, rather than on the first sentence, keeps the
        connection setup off the first response's latency.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting if the session's configuration changed.

        Args:
            delta: A :class:`TTSSettings` (or ``AssemblyAITTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed.keys() & {"voice", "language", "model"}:
            await self._disconnect()
            await self._connect()
        else:
            self._warn_unhandled_updated_settings(changed)
        return changed

    def _build_websocket_url(self) -> str:
        """Build the connect URL, which carries the whole session configuration."""
        settings = self._settings
        params: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "encoding": "pcm_s16le",
            "word_boundaries": "true",
        }
        # Unset values are left to the server's defaults.
        if settings.voice:
            params["voice_id"] = settings.voice
        if is_given(settings.language) and settings.language:
            params["language"] = settings.language
        if is_given(settings.model) and settings.model:
            params["model_id"] = settings.model
        return f"{self._url}?{urlencode(params)}"

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to AssemblyAI and start the receive and keepalive tasks."""
        await super()._connect()
        self._reconnect_delay = 0.0
        await self._open_session()
        self._create_keepalive_task(_KEEPALIVE_INTERVAL_S)

    async def _disconnect(self):
        """Disconnect from AssemblyAI and stop the receive and keepalive tasks."""
        await super()._disconnect()
        await self.stop_all_metrics()
        await self._cancel_keepalive_task()
        await self._close_session()

    async def _open_session(self):
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _close_session(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _renew_session_if_expiring(self):
        """Replace the session ahead of its expiry, while nothing is in flight.

        Audio contexts stay open, so a turn in progress carries on in the new
        session.
        """
        if self._renew_at is None or time.time() < self._renew_at:
            return
        if self._requests or self._word_contexts:
            return
        logger.debug(f"{self}: session {self._session_id} expires soon; opening a new one")
        await self._close_session()
        await self._open_session()

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect, first waiting as long as the last Error asked."""
        if self._reconnect_delay:
            delay, self._reconnect_delay = self._reconnect_delay, 0.0
            logger.debug(f"{self}: waiting {delay} s before reconnecting, as the server asked")
            await asyncio.sleep(delay)
        return await super()._reconnect_websocket(attempt_number)

    async def _connect_websocket(self):
        """Open the WebSocket session."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = self._build_websocket_url()
            logger.debug(f"{self}: connecting to {url}")
            headers = {
                "Authorization": self._api_key,
                "User-Agent": f"AssemblyAI/1.0 (integration=Pipecat/{pipecat_version()})",
            }
            self._websocket = await self._websocket_connect(url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to AssemblyAI TTS: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket session and forget its state.

        Closing discards synthesis still in flight, which is what a disconnect
        wants; by the end of a pipeline every reply has already been played.
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug(f"{self}: disconnecting")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            self._websocket = None
            self._reset_session_state()
            await self._call_event_handler("on_disconnected")

    def _reset_session_state(self):
        """Forget per-session state, which a new session doesn't carry over."""
        self._next_flush_id = 0
        self._requests.clear()
        self._word_contexts.clear()
        self._session_id = None
        self._renew_at = None
        self._max_generate_chars = _MAX_GENERATE_CHARS
        self._word_boundaries = False

    async def _close_connection_contexts(self):
        """Forget the bookkeeping of the contexts a replaced connection closes."""
        for context_id in list(self._contexts):
            await self._forget_context(context_id)
        await super()._close_connection_contexts()

    def _get_websocket(self):
        """Return the active WebSocket connection or raise if disconnected."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send(self, message: dict):
        await self._get_websocket().send(json.dumps(message))

    async def _send_keepalive(self):
        """Send KeepAlive so an idle session stays open."""
        await self._send({"type": "KeepAlive"})

    # ------------------------------------------------------------------
    # Audio contexts
    # ------------------------------------------------------------------

    def _context_state(self, context_id: str) -> _ContextState:
        return self._contexts.setdefault(context_id, _ContextState())

    async def _maybe_close_context(self, context_id: str):
        """Close a context once its turn has ended and the server owes it nothing.

        That is: every Flush answered with FlushDone, and every WordBoundaries
        frame due arrived, or the grace period for one ran out.
        """
        state = self._contexts.get(context_id)
        if not state or not state.turn_done:
            return
        if context_id in self._requests.values():
            return
        if state.awaiting_words:
            if not state.grace_task:
                state.grace_task = self.create_task(
                    self._word_boundaries_grace(context_id), f"{self}::word_boundaries_grace"
                )
            return
        await self._close_context(context_id)

    async def _word_boundaries_grace(self, context_id: str):
        await asyncio.sleep(_WORD_BOUNDARIES_GRACE_S)
        state = self._contexts.get(context_id)
        if state:
            logger.debug(f"{self}: no WordBoundaries for flushes {state.awaiting_words}")
            state.grace_task = None
            await self._close_context(context_id)

    async def _forget_context(self, context_id: str):
        """Drop a context's bookkeeping, including WordBoundaries still due for it."""
        state = self._contexts.pop(context_id, None)
        if state and state.grace_task:
            await self.cancel_task(state.grace_task)
        self._word_contexts = {
            flush_id: owner
            for flush_id, owner in self._word_contexts.items()
            if owner != context_id
        }

    async def _close_context(self, context_id: str):
        await self._forget_context(context_id)
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def _send_flush(self):
        self._next_flush_id += 1
        await self._send({"type": "Flush"})

    async def flush_audio(self, context_id: str | None = None):
        """Conclude the turn's text so the context can close once it is spoken.

        In sentence aggregation each sentence is already flushed by
        :meth:`run_tts`. When streaming tokens, the turn's text is flushed here.

        Args:
            context_id: The context whose turn has ended.
        """
        context_id = context_id or self.get_active_audio_context_id()
        if not context_id:
            return
        if self._requests.get(self._next_flush_id) == context_id:
            try:
                await self._send_flush()
            except Exception as e:
                logger.warning(f"{self}: error sending Flush: {e}")
        self._context_state(context_id).turn_done = True
        await self._maybe_close_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the server's in-flight synthesis when the bot is interrupted.

        ``Cancel`` discards everything the server has not yet delivered and
        keeps the session open, so the next response needs no reconnect.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        if self._requests:
            # Audio still in transit for these requests is dropped by flush_id.
            self._requests.clear()
            if self._websocket and self._websocket.state is State.OPEN:
                self._next_flush_id += 1
                try:
                    await self._send({"type": "Cancel"})
                except Exception as e:
                    logger.warning(f"{self}: error sending Cancel: {e}")
        await self._forget_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Forget a context's bookkeeping once it has finished playing.

        A context that goes quiet mid-turn ends on the base class's timeout and
        is recreated when more audio arrives, with word times measured from its
        new first audio, so its offset starts again from zero.

        Args:
            context_id: The ID of the audio context that finished playing.
        """
        await self._forget_context(context_id)
        await super().on_audio_context_completed(context_id)

    # ------------------------------------------------------------------
    # Server messages
    # ------------------------------------------------------------------

    async def _receive_messages(self):
        """Receive and dispatch server frames.

        Called by ``WebsocketService._receive_task_handler``, which reconnects
        when the session closes.
        """
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: invalid JSON message: {message}")
                continue

            msg_type = msg.get("type")
            if msg_type == "Audio":
                await self._handle_audio(msg)
            elif msg_type == "FlushDone":
                await self._handle_flush_done(msg)
            elif msg_type == "WordBoundaries":
                await self._handle_word_boundaries(msg)
            elif msg_type == "Begin":
                self._handle_begin(msg)
            elif msg_type == "Error":
                await self._handle_error(msg)
            else:
                # The protocol may add frame types, which are safe to ignore.
                logger.trace(f"{self}: unhandled message: {msg}")

    def _handle_begin(self, msg: dict):
        configuration = msg.get("configuration") or {}
        limits = configuration.get("limits") or {}
        self._session_id = msg.get("id")
        expires_at = msg.get("expires_at")
        if expires_at:
            margin = min(_SESSION_RENEWAL_MARGIN_S, (expires_at - time.time()) / 2)
            self._renew_at = expires_at - margin
        self._max_generate_chars = limits.get("max_generate_text_length") or _MAX_GENERATE_CHARS
        self._word_boundaries = configuration.get("word_boundaries") is True
        logger.debug(f"{self}: session {self._session_id} started: {configuration}")

    async def _handle_audio(self, msg: dict):
        context_id = self._requests.get(msg["flush_id"])
        if not context_id or not self.audio_context_available(context_id):
            return
        frame = TTSAudioRawFrame(
            audio=base64.b64decode(msg["audio"]),
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )
        await self.append_to_audio_context(context_id, frame)

    async def _handle_flush_done(self, msg: dict):
        flush_id = msg["flush_id"]
        context_id = self._requests.pop(flush_id, None)
        if not context_id:
            return
        state = self._context_state(context_id)
        duration_ms = msg.get("audio_duration_ms") or 0
        if self._word_boundaries and duration_ms:
            self._word_contexts[flush_id] = context_id
            state.awaiting_words[flush_id] = state.audio_ms
        state.audio_ms += duration_ms
        await self._maybe_close_context(context_id)

    async def _handle_word_boundaries(self, msg: dict):
        flush_id = msg["flush_id"]
        context_id = self._word_contexts.pop(flush_id, None)
        state = self._contexts.get(context_id) if context_id else None
        if not context_id or not state or flush_id not in state.awaiting_words:
            return

        # Word times are on the session's timeline; audio_start_ms is where
        # this request's audio starts on it.
        offset_ms = state.awaiting_words.pop(flush_id) - msg["audio_start_ms"]
        word_times = [
            (word["word"], (offset_ms + word["start"]) / 1000.0) for word in msg.get("words", [])
        ]
        if word_times:
            await self.add_word_timestamps(word_times, context_id)
        await self._maybe_close_context(context_id)

    async def _handle_error(self, msg: dict):
        """Report an Error frame. The server closes the session right after it.

        A refusal before ``Begin`` that isn't retryable is in the session's own
        configuration (the credential, voice, model or a parameter), so it
        leaves the service unusable. Retryable errors are retried by the
        receive loop's reconnect, after any wait the server asks for, and
        other errors after ``Begin`` concern the text sent.
        """
        code = msg.get("error_code")
        error_type = msg.get("error_type")
        label = f"{code} {error_type}" if error_type else f"{code}"
        text = f"AssemblyAI TTS error {label}: {msg.get('error', '')}"

        if msg.get("retryable"):
            self._reconnect_delay = float(msg.get("retry_after_seconds") or 0)
            if error_type in _ROUTINE_ERROR_TYPES:
                logger.debug(f"{self}: {text}")
                return
            category = ErrorCategory.RATE_LIMIT if code == 3009 else ErrorCategory.SERVER
        elif self._session_id is None:
            category = (
                ErrorCategory.AUTHENTICATION if code == 1008 else ErrorCategory.INVALID_REQUEST
            )
        else:
            category = ErrorCategory.UNKNOWN
        await self.push_error(error_msg=text, category=category)

    # ------------------------------------------------------------------
    # TTS generation
    # ------------------------------------------------------------------

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to AssemblyAI for synthesis.

        Audio arrives on the receive task. In sentence aggregation each call is
        flushed as its own request, since only a flush guarantees the server
        speaks all of the text.

        Args:
            text: The text to synthesize.
            context_id: The audio context the text belongs to.

        Yields:
            Frame: None, as audio arrives via the WebSocket receive task.
        """
        if self._reconnect_in_progress:
            # The receive loop owns the connection until it reconnects.
            yield ErrorFrame(error="AssemblyAI TTS is reconnecting; text not spoken")
            yield TTSStoppedFrame(context_id=context_id)
            return

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()
            else:
                await self._renew_session_if_expiring()

            # Recorded before sending, so audio that arrives while the messages
            # are still being sent is attributed to this context.
            self._requests[self._next_flush_id] = context_id
            # Consecutive Generate texts are joined exactly as sent.
            size = self._max_generate_chars
            for start in range(0, len(text), size):
                await self._send({"type": "Generate", "text": text[start : start + size]})
            if not self._is_streaming_tokens:
                await self._send_flush()

            await self.start_tts_usage_metrics(text)
        except Exception as e:
            yield ErrorFrame(error=f"AssemblyAI TTS error: {e}")
            yield TTSStoppedFrame(context_id=context_id)
            await self._disconnect()
            await self._connect()
            return

        yield None
