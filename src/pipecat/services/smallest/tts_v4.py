#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI Lightning v4 text-to-speech service implementation.

This module provides a WebSocket-based integration with Smallest AI's Lightning
v4 live session API (``/waves/v1/lightning-v4/live``), a dedicated endpoint held
separately from the Lightning v3.1 family's ``/waves/v1/tts/live`` route and
protocol implemented in :mod:`pipecat.services.smallest.tts`.

Lightning v4 is in beta: access is gated per account, English-only (``en`` or
``auto``), does not support word timestamps, and its live session is turn-based
(``speak``/``turn_start``/``turn_end``) rather than the continuation-context
model (``context_id``/``continue``) Lightning v3.1 uses. Because the wire
protocols differ this fundamentally, Lightning v4 is implemented as its own
service rather than a model option on :class:`~pipecat.services.smallest.tts.SmallestTTSService`.
"""

import asyncio
import base64
import io
import json
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

# Sample rates the live session will negotiate; anything else is rejected at
# connect time with close code 1008.
_ALLOWED_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)
_DEFAULT_SAMPLE_RATE = 48000

_DEFAULT_VOICE = "brannock"

# The `speed` connect param's accepted range; outside it, the connection is
# refused at the handshake. Validated at construction so a bad value fails
# immediately instead of on the next connect.
_SPEED_MIN = 0.5
_SPEED_MAX = 2.0

_READY_TIMEOUT_SECONDS = 10.0


def language_to_smallest_lightning_v4_language(language: Language) -> str:
    """Convert a Language enum to a Lightning v4 language string.

    Lightning v4 is English-only and accepts only ``en`` or ``auto``; any other
    language is rejected at connect time. Non-English languages are mapped to
    ``auto`` (server-side detection) rather than left to fail.

    Args:
        language: The Language enum value to convert.

    Returns:
        ``"en"`` for English variants, ``"auto"`` otherwise.
    """
    base_code = language.value.split("-")[0].lower()
    return "en" if base_code == "en" else "auto"


def _resolve_sample_rate(sample_rate: int) -> int:
    """Pick the rate Lightning v4 will render at.

    Args:
        sample_rate: The pipeline's output rate.

    Returns:
        The rate Lightning v4 will synthesize at.
    """
    if sample_rate in _ALLOWED_SAMPLE_RATES:
        return sample_rate
    logger.warning(
        f"Smallest Lightning v4 cannot render {sample_rate} Hz "
        f"(supports {list(_ALLOWED_SAMPLE_RATES)}); synthesizing at "
        f"{_DEFAULT_SAMPLE_RATE} Hz and resampling on output"
    )
    return _DEFAULT_SAMPLE_RATE


def _error_detail(msg: dict) -> str:
    """Unwrap either of Lightning v4's two error shapes to a readable string."""
    error = msg.get("error")
    if isinstance(error, dict):
        return f"{error.get('code')}: {error.get('message', msg)}"
    return str(msg.get("message", msg))


@dataclass
class SmallestLightningV4TTSSettings(TTSSettings):
    """Settings for SmallestLightningV4TTSService.

    ``voice``, ``language``, ``speed``, ``content_filter`` and
    ``content_filter_action`` are all negotiated in the connection's query
    string, so changing any of them updates the running session by
    reconnecting rather than taking effect on the next turn.

    Parameters:
        speed: Playback speed multiplier (0.5-2.0). If None, the API default
            (1.0) applies.
        content_filter: Whether to screen each turn's text before synthesis.
            If None, the API default (off) applies.
        content_filter_action: ``"reject"`` drops a turn that matches the
            filter; ``"flag"`` synthesizes it and records the match. If None,
            the API default (``"reject"``) applies.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    content_filter: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    content_filter_action: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SmallestLightningV4TTSService(WebsocketTTSService):
    """Smallest AI Lightning v4 real-time text-to-speech service.

    Lightning v4 is beta and gated per account: an account without access gets
    a ``model_access_denied`` error and the connection is closed.

    Unlike :class:`~pipecat.services.smallest.tts.SmallestTTSService`, Lightning
    v4's live session has no continuation model — each turn is synthesized and
    delivered independently (``turn_start``, audio, ``turn_end``), with no
    ``context_id``/``continue`` fragment buffering. This service reflects that
    by not reusing a context ID across a turn's text fragments
    (``reuse_context_id_within_turn=False``): each aggregated piece of text
    becomes its own Lightning v4 turn.

    Word timestamps are not supported on this endpoint. Only ``en`` and
    ``auto`` are valid languages; other languages resolve to ``auto``.

    Defaults to the ``brannock`` voice; pass ``settings.voice`` to use a
    different one from the Lightning v4 ``get_voices`` catalogue.

    The session also conditions each turn on the *caller's* side of the
    conversation, but only if told, and only with audio: this service
    captures the caller's raw microphone audio between
    ``UserStartedSpeakingFrame`` and ``UserStoppedSpeakingFrame`` on its own,
    and pairs it with the caller's transcript when the application calls
    :meth:`add_user_turn` (e.g. from an LLM user aggregator's
    ``on_user_turn_stopped`` handler) after each user turn.

    Example::

        tts = SmallestLightningV4TTSService(
            api_key="your-api-key",
            settings=SmallestLightningV4TTSService.Settings(
                voice="rhodes",
                language=Language.EN,
            ),
        )
    """

    Settings = SmallestLightningV4TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.smallest.ai",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Smallest Lightning v4 WebSocket TTS service.

        Args:
            api_key: Smallest AI API key for authentication.
            base_url: Base WebSocket URL for the Smallest API.
            sample_rate: Audio sample rate in Hz. Must be one of 8000, 16000,
                24000, 44100 or 48000; any other value falls back to 48000
                with a warning. If None, uses the pipeline default.
            settings: Runtime-updatable settings for the TTS service. Defaults
                to the ``brannock`` voice; pass ``voice`` to use another one.
            **kwargs: Additional arguments passed to parent WebsocketTTSService.
        """
        default_settings = self.Settings(
            model="lightning_v4",
            voice=_DEFAULT_VOICE,
            language=Language.EN,
            speed=None,
            content_filter=None,
            content_filter_action=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        if (
            is_given(default_settings.speed)
            and default_settings.speed is not None
            and not (_SPEED_MIN <= default_settings.speed <= _SPEED_MAX)
        ):
            raise ValueError(
                f"speed must be within {_SPEED_MIN}-{_SPEED_MAX}, got {default_settings.speed}"
            )

        super().__init__(
            push_stop_frames=False,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            reuse_context_id_within_turn=False,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._v4_sample_rate = _DEFAULT_SAMPLE_RATE
        self._receive_task = None
        # Binary audio frames carry no turn id, so they're attributed to the
        # turn most recently confirmed by `turn_start`.
        self._turn_context_id: str | None = None

        # Caller audio for the turn currently being captured (between
        # UserStartedSpeakingFrame and UserStoppedSpeakingFrame), and the most
        # recently finished turn's captured audio, held until its transcript
        # arrives via add_user_turn().
        self._capturing_caller_audio = False
        self._caller_audio = bytearray()
        self._caller_audio_sample_rate = 0
        self._pending_caller_audio: bytes | None = None
        self._pending_caller_audio_sample_rate = 0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Lightning v4 service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Lightning v4 service language format.

        Args:
            language: The language to convert.

        Returns:
            ``"en"`` or ``"auto"``.
        """
        return language_to_smallest_lightning_v4_language(language)

    def _build_websocket_url(self) -> str:
        """Build the Lightning v4 live-session WebSocket URL with connect-time params."""
        params: dict[str, Any] = {
            "voice_id": self._settings.voice,
            "language": self._settings.language,
            "sample_rate": self._v4_sample_rate,
            "output_format": "pcm",
        }
        if self._settings.speed is not None:
            params["speed"] = self._settings.speed
        if self._settings.content_filter is not None:
            params["content_filter"] = str(self._settings.content_filter).lower()
        if self._settings.content_filter_action is not None:
            params["content_filter_action"] = self._settings.content_filter_action

        return f"{self._base_url}/waves/v1/lightning-v4/live?{urlencode(params)}"

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._v4_sample_rate = _resolve_sample_rate(self.sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Smallest Lightning v4 TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Smallest Lightning v4 TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Capture the caller's raw audio for the turn add_user_turn() will send.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._capturing_caller_audio = True
            self._caller_audio = bytearray()
        elif isinstance(frame, InputAudioRawFrame) and self._capturing_caller_audio:
            self._caller_audio_sample_rate = frame.sample_rate
            self._caller_audio.extend(frame.audio)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._capturing_caller_audio = False
            if self._caller_audio:
                self._pending_caller_audio = bytes(self._caller_audio)
                self._pending_caller_audio_sample_rate = self._caller_audio_sample_rate
            self._caller_audio = bytearray()

    async def add_user_turn(self, text: str):
        """Feed the caller's turn into Lightning v4's server-held context.

        The session conditions each `speak` on what came before it, including
        what the caller said — but only via a `user_audio` frame that carries
        BOTH the transcript and the caller's own captured audio. A text-only
        entry is not a weaker version of this: the server discards context
        entries with no audio outright, so sending text alone looks like it
        works while conditioning nothing. If no audio was captured for this
        turn (e.g. it arrived before the first `UserStoppedSpeakingFrame`),
        this drops the turn and logs a warning rather than sending one.

        Nothing in the base pipeline calls this automatically — a
        `TranscriptionFrame` is consumed by the user aggregator and never
        reaches the TTS service — so the application must call it itself,
        typically from an `on_user_turn_stopped` handler on the LLM's user
        aggregator, right after this service's own `process_frame` has seen
        that turn's `UserStoppedSpeakingFrame`.

        Args:
            text: The caller's turn, e.g. the content of a
                :class:`~pipecat.processors.aggregators.llm_response_universal.UserTurnStoppedMessage`.
        """
        if not text or not self._websocket:
            return

        audio = self._pending_caller_audio
        sample_rate = self._pending_caller_audio_sample_rate
        self._pending_caller_audio = None

        if not audio:
            logger.warning(
                f"{self}: dropping caller turn with no captured audio ({text[:40]!r}); "
                "Lightning v4 discards text-only context entries"
            )
            return

        msg = {
            "event": "user_audio",
            "text": text,
            "audio": self._wav_base64(audio, sample_rate),
        }
        try:
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.warning(f"{self} error sending user turn context: {e}")

    @staticmethod
    def _wav_base64(pcm: bytes, sample_rate: int) -> str:
        """Wrap 16-bit mono PCM in a WAV container, base64 encoded."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting if a connect-time field changed.

        ``voice``, ``language``, ``speed``, ``content_filter`` and
        ``content_filter_action`` are only negotiated when the connection is
        opened, so changing any of them requires a fresh connection.
        """
        new_speed = getattr(delta, "speed", NOT_GIVEN)
        if (
            is_given(new_speed)
            and new_speed is not None
            and not (_SPEED_MIN <= new_speed <= _SPEED_MAX)
        ):
            raise ValueError(f"speed must be within {_SPEED_MIN}-{_SPEED_MAX}, got {new_speed}")

        changed = await super()._update_settings(delta)

        if changed.keys() & {
            "voice",
            "language",
            "speed",
            "content_filter",
            "content_filter_action",
        }:
            await self._disconnect()
            await self._connect()

        return changed

    async def _connect(self):
        """Connect to the Lightning v4 live session and start the receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the Lightning v4 live session and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open the socket and wait for the session to be fully attached.

        The server sends `ready` twice: first to confirm admission (carrying
        `session_id`), then once a voice server is attached (carrying the
        negotiated `sample_rate`). `speak` is only valid after the second.
        """
        websocket = None
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Smallest Lightning v4")

            websocket = await self._websocket_connect(
                self._build_websocket_url(),
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Source": "pipecat",
                    "X-Pipecat-Version": pipecat_version(),
                },
            )

            dispatched = json.loads(
                await asyncio.wait_for(websocket.recv(), timeout=_READY_TIMEOUT_SECONDS)
            )
            if dispatched.get("event") != "ready":
                raise Exception(f"Lightning v4 rejected the session: {_error_detail(dispatched)}")

            attached = json.loads(
                await asyncio.wait_for(websocket.recv(), timeout=_READY_TIMEOUT_SECONDS)
            )
            if attached.get("event") != "ready":
                raise Exception(f"Lightning v4 rejected the session: {_error_detail(attached)}")

            negotiated_rate = attached.get("sample_rate")
            if negotiated_rate and negotiated_rate != self._v4_sample_rate:
                logger.warning(
                    f"Lightning v4 negotiated {negotiated_rate} Hz, expected "
                    f"{self._v4_sample_rate} Hz; using the negotiated rate"
                )
                self._v4_sample_rate = negotiated_rate

            logger.debug(f"{self}: session ready (session_id: {dispatched.get('session_id')})")
            self._websocket = websocket
            self._turn_context_id = None
            await self._call_event_handler("on_connected")
        except Exception as e:
            if websocket is not None:
                try:
                    await websocket.close()
                except Exception:
                    pass
            await self.push_error(
                error_msg=f"Smallest Lightning v4 connection error: {e}", exception=e
            )
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Smallest Lightning v4")
                if self._websocket.state is State.OPEN:
                    await self._websocket.send(json.dumps({"event": "end"}))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Smallest Lightning v4 error closing websocket: {e}", exception=e
            )
        finally:
            self._websocket = None
            self._turn_context_id = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the WebSocket connection if available.

        Returns:
            The active WebSocket connection.

        Raises:
            Exception: If no WebSocket connection is available.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Barge in on the turn currently speaking instead of reconnecting.

        A single `interrupt` frame stops the turn in progress and retires any
        turn queued behind it server-side, so the session (and its
        accumulated context) survives an interruption.
        """
        await self.stop_all_metrics()
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"event": "interrupt"}))
            except Exception as e:
                logger.error(f"{self} error sending interrupt message: {e}")
        if self._turn_context_id == context_id:
            self._turn_context_id = None
        await super().on_audio_context_interrupted(context_id)

    async def _receive_messages(self):
        """Receive and process messages from the Lightning v4 live session."""
        try:
            async for message in self._get_websocket():
                if isinstance(message, bytes):
                    context_id = self._turn_context_id or self.get_active_audio_context_id()
                    await self.stop_ttfb_metrics()
                    await self.append_to_audio_context(
                        context_id,
                        TTSAudioRawFrame(message, self._v4_sample_rate, 1, context_id=context_id),
                    )
                    continue

                msg = json.loads(message)
                event = msg.get("event")

                if event == "ready":
                    logger.debug(f"{self}: received a late `ready` frame: {msg}")
                elif event == "turn_start":
                    self._turn_context_id = msg.get("turn_id")
                elif event == "turn_end":
                    await self._end_turn(msg.get("turn_id"))
                    await self.stop_all_metrics()
                elif event == "interrupted":
                    await self._end_turn(msg.get("turn_id"))
                elif event == "error":
                    # Session-level errors (e.g. `bad_frame`, `bad_user_audio`) that
                    # leave the connection open.
                    await self.push_error(
                        error_msg=f"Smallest Lightning v4 error: {msg.get('code')}: "
                        f"{msg.get('message', msg)}"
                    )
                elif msg.get("status") == "error":
                    await self.push_error(
                        error_msg=f"Smallest Lightning v4 error: {_error_detail(msg)}"
                    )
                else:
                    logger.warning(f"{self} unknown message: {msg}")
        finally:
            # The socket is gone; a turn still in flight is over too, since turn
            # state lives in the session and the reconnect the base class is
            # about to perform starts a new one.
            lost = self._turn_context_id
            if lost is not None:
                await self.push_error(
                    error_msg=f"{self} lost the connection mid-turn; turn {lost} was dropped"
                )
                await self._end_turn(lost)

    async def _end_turn(self, context_id: str | None):
        """Close out a turn's audio context and clear the in-flight turn marker."""
        if self._turn_context_id == context_id:
            self._turn_context_id = None
        if context_id and self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Synthesize a turn using Lightning v4's turn-based live session.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this turn. Sent as the Lightning
                v4 `turn_id` so `turn_start`/`turn_end`/`interrupted` can be
                matched back to it.

        Yields:
            Frame: Audio arrives via the WebSocket receive task.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                msg = {"event": "speak", "turn_id": context_id, "text": text}
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Smallest Lightning v4 send error: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Smallest Lightning v4 error: {e}")
