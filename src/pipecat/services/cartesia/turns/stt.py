#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cartesia Ink-2 Streaming ASR (v2 turn-based) speech-to-text service."""

import asyncio
import json
import time
import urllib.parse
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.cartesia.stt import _prepare_keyterms
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.eager_end_of_turn_mixin import EagerEndOfTurnSTTServiceMixin
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

# Connection-bound turn detection query parameters, named as Cartesia documents
# them. A change to any of them is applied by reconnecting.
_TURN_DETECTION_FIELDS = (
    "turn_start_threshold",
    "turn_eager_end_threshold",
    "turn_end_threshold",
    "turn_end_timeout_ms",
)


class CartesiaTurnsRequestError(Exception):
    """Cartesia Ink-2 ASR rejected a connection's settings.

    Carries the HTTP status code the server reports with the error, so the
    error is classified like a rejected HTTP request.
    """

    def __init__(self, message: str, status_code: int | None = None):
        """Initialize the error.

        Args:
            message: The error message.
            status_code: The HTTP status code the server reported, if any.
        """
        super().__init__(message)
        self.status_code = status_code


@dataclass
class CartesiaTurnsSTTSettings(STTSettings):
    """Settings for CartesiaTurnsSTTService.

    The ink-2 model family is English-only and does not support runtime model
    or language switching.

    The four ``turn_*`` fields tune server-side turn detection. Cartesia binds
    them to a connection, so updating any of them at runtime triggers a
    reconnect, and requires the three thresholds to stay in order:
    ``turn_start_threshold`` > ``turn_eager_end_threshold`` >
    ``turn_end_threshold``. See https://docs.cartesia.ai/use-the-api/stt/turns
    for the Balanced, Responsive and Patient starting profiles.

    Parameters:
        keyterm: Key terms or phrases to bias transcription towards, sent as
            repeated ``keyterm`` query parameters on the connection URL.
            Cartesia binds keyterms to a connection, so updating this setting
            at runtime triggers a reconnect. See
            https://docs.cartesia.ai/use-the-api/stt/keyterms.
        turn_start_threshold: Likelihood above which the server emits
            ``turn.start``. 0.5–0.9; Cartesia defaults to 0.8.
        turn_eager_end_threshold: Likelihood below which the server emits
            ``turn.eager_end``. 0.3–0.8; Cartesia defaults to 0.6.
        turn_end_threshold: Likelihood below which the server emits
            ``turn.end``. 0.05–0.5; Cartesia defaults to 0.3.
        turn_end_timeout_ms: Milliseconds to wait after the user stops speaking
            before emitting ``turn.end``. 640–11200; Cartesia defaults to 5600.
    """

    keyterm: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_start_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_eager_end_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_end_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_end_timeout_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class CartesiaTurnsSTTService(EagerEndOfTurnSTTServiceMixin, WebsocketSTTService):
    """Speech-to-text service using the Cartesia Streaming ASR v2 (Ink-2) API.

    Speaks the v2 turn-based wire protocol exposed by
    ``/stt/turns/websocket``. The server drives the conversation::

        connected -> turn.start -> turn.update* -> (turn.eager_end -> turn.resume?)*
                                -> turn.end -> ...

    Transcripts are cumulative per turn; there is no ``is_final`` flag and no
    ``finalize`` command — closing the socket ends the session.

    Each ``turn.start`` pushes a :class:`ProposedUserStartedSpeakingFrame`; each
    ``turn.update`` pushes an :class:`InterimTranscriptionFrame`; ``turn.end``
    pushes a final :class:`TranscriptionFrame` followed by a
    :class:`ProposedUserStoppedSpeakingFrame`. ``turn.eager_end`` pushes an
    :class:`EagerTranscriptionFrame` and ``turn.resume`` an
    :class:`EagerEndOfTurnCancelFrame`, which
    :class:`~pipecat.turns.user_turn_strategies.EagerUserTurnStrategies` uses to
    answer a predicted end of turn ahead of the committed one.

    Event handlers available (in addition to the base
    ``on_connected`` / ``on_disconnected`` / ``on_connection_error``):

    - on_turn_start(service, transcript): server detected start of a turn
    - on_turn_update(service, transcript): incremental transcript update
    - on_turn_eager_end(service, transcript): server eagerly predicted end of turn
    - on_turn_resume(service): user resumed speaking after an eager end
    - on_turn_end(service, transcript): final transcript for the completed turn

    Example::

        @stt.event_handler("on_turn_end")
        async def on_turn_end(service, transcript):
            ...
    """

    Settings = CartesiaTurnsSTTSettings
    _settings: Settings

    # How long to wait for the server's ``connected`` frame after the
    # handshake before treating the connection as failed.
    _CONNECTION_TIMEOUT = 10.0

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.cartesia.ai/stt/turns/websocket",
        sample_rate: int | None = None,
        should_interrupt: bool = True,
        watchdog_min_timeout: float = 0.5,
        enable_eager_end_of_turn: bool = False,
        extra_headers: dict[str, str] | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Cartesia Ink-2 STT service.

        Args:
            api_key: Cartesia API key.
            url: WebSocket URL for the Cartesia Streaming ASR v2 endpoint.
            sample_rate: Audio sample rate in Hz. If ``None``, uses the pipeline
                sample rate.
            should_interrupt: Whether to interrupt the bot when the server
                signals the start of a new turn. Passed along to the user turn
                strategies this service recommends, which own the interruption;
                a user-supplied ``user_turn_strategies`` overrides the
                recommendation and this setting with it.
            watchdog_min_timeout: Minimum idle timeout before sending silence to
                prevent dangling turns. The actual threshold is
                ``max(chunk_duration * 2, watchdog_min_timeout)``. Defaults to 0.5.
            enable_eager_end_of_turn: Whether to answer the server's predicted
                end of turn (``turn.eager_end``) ahead of the committed one, so
                the gap between the two is spent generating a response rather
                than waiting. The response is discarded if the user resumes
                speaking or the committed transcript differs from the predicted
                one. Off by default: it spends an inference on every prediction,
                including the ones the server withdraws.
            extra_headers: Optional additional HTTP headers to send with the
                WebSocket handshake.
            settings: Runtime-updatable settings. The ink-2 family does not
                support runtime model or language switching; attempts to update
                either field will be reported as unhandled.
            **kwargs: Additional arguments passed to the parent
                :class:`WebsocketSTTService`.
        """
        # ink-2 is English-only at launch.
        default_settings = self.Settings(
            model="ink-2",
            language=None,
            keyterm=None,
            # Left unset so the server applies its own turn detection defaults.
            turn_start_threshold=None,
            turn_eager_end_threshold=None,
            turn_end_threshold=None,
            turn_end_timeout_ms=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            enable_eager_end_of_turn=enable_eager_end_of_turn,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._should_interrupt = should_interrupt
        self._watchdog_min_timeout = watchdog_min_timeout
        self._extra_headers = dict(extra_headers) if extra_headers else {}
        # ink-2 is English-only at launch; language on emitted frames is fixed.
        self._language = Language.EN

        self._request_id: str | None = None
        self._receive_task: asyncio.Task | None = None
        self._connection_established_event = asyncio.Event()

        # Watchdog state — see _watchdog_task_handler for details.
        self._last_stt_time: float | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._user_is_speaking = False
        self._last_audio_chunk_duration: float = 0.0
        # The transcript of the turn in progress, pushed if its connection
        # closes before turn.end.
        self._turn_transcript = ""

        self._register_event_handler("on_turn_start")
        self._register_event_handler("on_turn_update")
        self._register_event_handler("on_turn_eager_end")
        self._register_event_handler("on_turn_resume")
        self._register_event_handler("on_turn_end")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Cartesia Ink-2 service supports metrics generation.
        """
        return True

    @property
    def supports_ttfs(self) -> bool:
        """TTFS doesn't apply: the server defines turn boundaries directly."""
        return False

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend turn strategies that leave turn detection to the server.

        Cartesia's turn-detection STT defines turn boundaries on the server and
        emits ``ProposedUserStarted/StoppedSpeakingFrame``, so the user aggregator
        resolves those rather than running local VAD/smart-turn. With
        ``enable_eager_end_of_turn``, the recommendation also answers the
        server's predicted end of turn. Applied unless the user passed their own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        frame.user_turn_strategies = self.recommended_user_turn_strategies(
            enable_interruptions=self._should_interrupt,
        )
        return frame

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close the WebSocket connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close the WebSocket connection.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Keyterms and the turn detection thresholds are bound to a connection,
        so a change to any of them is applied by reconnecting. Ink-2 does not
        support runtime model or language switching, so those changes are
        reported as unhandled.

        Args:
            delta: A :class:`STTSettings` (or
                :class:`CartesiaTurnsSTTSettings`) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        connection_bound = {"keyterm", *_TURN_DETECTION_FIELDS}
        self._warn_unhandled_updated_settings(changed.keys() - connection_bound)

        if changed.keys() & connection_bound:
            await self._request_reconnect()

        return changed

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to the WebSocket and start the receive loop.

        The receive loop starts even when the connection fails: with no socket
        it goes straight to its reconnect path, which retries the connect. It
        owns reconnection for the session, so audio is never held up waiting
        on a connection.
        """
        await super()._connect()
        await self._connect_websocket()
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close the session, letting the server flush pending turn responses first."""
        # Marks the disconnect as intentional, so the receive loop doesn't
        # reconnect as it winds down.
        await super()._disconnect()
        try:
            # Cancel the watchdog before sending close so it doesn't keep
            # injecting silence during the drain.
            if self._watchdog_task:
                await self.cancel_task(self._watchdog_task, timeout=1.0)
                self._watchdog_task = None

            # Send close so the server can flush any pending turn responses
            # before it tears the socket down from its side.
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Sending close command to Cartesia Ink-2 ASR")
                try:
                    await self._websocket.send(json.dumps({"type": "close"}))
                except Exception as e:
                    logger.warning(f"Failed to send close command: {e}")

            # Wait for the receive loop to exit as the server drains responses
            # and closes the socket. Fall back to cancellation if it doesn't
            # complete in time.
            if self._receive_task:
                try:
                    await asyncio.wait_for(asyncio.shield(self._receive_task), timeout=2.0)
                except TimeoutError:
                    logger.debug("Timed out waiting for server to close; cancelling receive task")
                    await self.cancel_task(self._receive_task, timeout=1.0)
                except Exception:
                    # Receive task already errored; cancel_task is a no-op on done tasks.
                    await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

        await self._disconnect_websocket()

    def _websocket_url(self) -> str:
        # Pipecat pipes 16-bit signed little-endian PCM through the pipeline,
        # so the wire encoding is fixed.
        params = [
            ("model", self._settings.model),
            ("encoding", "pcm_s16le"),
            ("sample_rate", str(self.sample_rate)),
        ]
        params.extend(
            (name, str(value))
            for name in _TURN_DETECTION_FIELDS
            if is_given(value := getattr(self._settings, name)) and value is not None
        )
        params.extend(("keyterm", term) for term in _prepare_keyterms(self._settings.keyterm))
        # Cartesia expects spaces inside a keyterm as %20, which urlencode only
        # emits with quote_via=quote.
        return f"{self._url}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"

    async def _connect_websocket(self):
        """Open the v2 WebSocket.

        The server confirms the connection with a ``connected`` frame, which
        the receive loop waits for (see ``_receive_messages``).
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._connection_established_event.clear()
            self._request_id = None
            self._user_is_speaking = False
            self._last_stt_time = None

            headers = {
                "X-API-Key": self._api_key,
                "Cartesia-Version": "2026-03-01",
                **self._extra_headers,
            }
            logger.debug(f"Connecting to Cartesia Ink-2 ASR: {self._websocket_url()}")
            self._websocket = await self._websocket_connect(
                self._websocket_url(), additional_headers=headers
            )
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket and reset connection state.

        Called on its own by the receive loop when it reconnects, so it leaves
        the receive loop and watchdog running.
        """
        try:
            self._connection_established_event.clear()
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug(f"{self}: Disconnecting from Cartesia Ink-2 ASR")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    # ------------------------------------------------------------------
    # Audio send / receive
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Forward raw PCM audio to the server.

        Transcription results are delivered asynchronously via the receive
        task and are not yielded from this method.

        Audio that arrives while the connection is down is dropped; the
        receive loop reconnects in the background. The server takes audio as
        soon as the socket opens, ahead of its ``connected`` frame.

        Args:
            audio: Raw 16-bit signed little-endian PCM audio bytes.

        Yields:
            Frame: ``None`` (transcription results are pushed by the receive
                task), or :class:`ErrorFrame` on send failure.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                self._last_stt_time = time.monotonic()
                self._last_audio_chunk_duration = len(audio) / (self.sample_rate * 2)
                await self._websocket.send(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Cartesia Ink-2 ASR send error: {e}")
                return
        yield None

    async def _send_silence(self, duration_secs: float = 0.5):
        """Send a block of 16-bit PCM mono silence of the specified duration."""
        sample_width = 2  # bytes per sample for 16-bit PCM
        num_channels = 1  # mono
        num_samples = int(self.sample_rate * duration_secs)
        silence = b"\x00" * (num_samples * sample_width * num_channels)
        await self._get_websocket().send(silence)
        # Watchdog silence is real audio submitted to the service, so it
        # counts toward usage.
        self._record_stt_audio_usage(silence)

    async def _watchdog_task_handler(self):
        """Prevent dangling turns by sending silence when audio stops flowing.

        If we stop sending audio after a ``turn.start``, the server never
        emits ``turn.end`` unless we resume sending audio.
        """
        while self._websocket and self._websocket.state is State.OPEN:
            now = time.monotonic()
            # Send silence if we go more than watchdog_min_timeout (or twice
            # the chunk size, whichever is larger) without sending new audio.
            threshold = max(self._last_audio_chunk_duration * 2, self._watchdog_min_timeout)
            if (
                self._user_is_speaking
                and self._last_stt_time
                and (elapsed := now - self._last_stt_time) > threshold
            ):
                logger.warning(
                    f"No audio received for {elapsed * 1000:.0f} ms. "
                    "Sending silence to Cartesia to prevent a dangling turn"
                )
                try:
                    await self._send_silence(elapsed)
                except Exception as e:
                    logger.warning(f"Failed to send silence: {e}")
                self._last_stt_time = time.monotonic()
            await asyncio.sleep(0.1)

    async def _receive_messages(self):
        """Receive and process messages from the WebSocket.

        Waits for the server to confirm the connection with ``connected``, then
        processes turn events and errors.

        Raises:
            ConnectionError: If ``connected`` doesn't arrive within
                ``_CONNECTION_TIMEOUT``.
            CartesiaTurnsRequestError: If the server rejects the connection's
                settings.
        """
        messages = aiter(self._get_websocket())
        if not await self._receive_until_confirmed(
            messages,
            self._connection_established_event,
            self._process_message,
            self._CONNECTION_TIMEOUT,
        ):
            # Closed before confirming; the receive loop treats it as a drop.
            return

        await self._call_event_handler("on_connected")
        # The watchdog stops when its connection closes, so each confirmed
        # connection gets a new one.
        if not self._watchdog_task or self._watchdog_task.done():
            self._watchdog_task = self.create_task(self._watchdog_task_handler())

        try:
            async for message in messages:
                await self._process_message(message)
        finally:
            # A turn lives in one connection, so it ends when the connection
            # does, unless the service is stopping.
            if not self._disconnecting:
                await self._end_turn_cut_short()

    async def _process_message(self, message: str | bytes):
        """Decode and handle one message from the WebSocket."""
        if not isinstance(message, str):
            logger.warning(f"Received non-text message: {type(message)}")
            return
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON message")
            return
        try:
            await self._handle_message(data)
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            # Re-raise so the receive loop handles it: it reconnects, unless
            # the error left the service unusable.
            raise

    async def _end_turn_cut_short(self):
        """End a turn whose connection closed before the server sent turn.end.

        A turn lives in one connection, so a new connection never ends the old
        one's turn. The transcript received so far is pushed as the turn's
        transcription, unfinalized since the server never committed it, and the
        turn stop is proposed. If the user keeps talking on a new connection,
        the rest is transcribed as a new turn.
        """
        if not self._user_is_speaking:
            return
        logger.debug(f"{self}: connection closed mid-turn, ending the turn")
        self._user_is_speaking = False
        self._clear_eager_end_of_turn()
        transcript, self._turn_transcript = self._turn_transcript, ""
        if transcript:
            await self.emit_stt_usage_metrics()
            await self.push_frame(
                TranscriptionFrame(transcript, self._user_id, time_now_iso8601(), self._language)
            )
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, data: Any):
        if not isinstance(data, dict):
            logger.warning("Received non-dict message")
            return

        message_type = data.get("type")
        if not message_type:
            logger.warning("Message missing 'type' field")
            return

        if message_type == "connected":
            await self._handle_connected(data)
        elif message_type == "turn.start":
            await self._handle_turn_start(data)
        elif message_type == "turn.update":
            await self._handle_turn_update(data)
        elif message_type == "turn.eager_end":
            await self._handle_turn_eager_end(data)
        elif message_type == "turn.resume":
            await self._handle_turn_resume(data)
        elif message_type == "turn.end":
            await self._handle_turn_end(data)
        elif message_type == "error":
            await self._handle_server_error(data)
        else:
            logger.debug(f"Unhandled message type: {message_type}")

    async def _handle_connected(self, data: dict):
        self._request_id = data.get("request_id")
        logger.info(f"Cartesia Ink-2 ASR connected (request_id={self._request_id})")
        self._connection_established_event.set()

    async def _handle_turn_start(self, data: dict):
        transcript = data.get("transcript", "")
        logger.debug("Cartesia Ink-2 ASR turn.start")
        self._user_is_speaking = True
        self._turn_transcript = transcript
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        await self._call_event_handler("on_turn_start", transcript)

    async def _handle_turn_update(self, data: dict):
        transcript = data.get("transcript", "")
        if transcript:
            logger.trace(f"Cartesia Ink-2 ASR turn.update: {transcript}")
            self._turn_transcript = transcript
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    self._language,
                    result=data,
                )
            )
        await self._call_event_handler("on_turn_update", transcript)

    async def _handle_turn_eager_end(self, data: dict):
        """Handle an eagerly predicted end of turn.

        The prediction may be withdrawn by ``turn.resume``, and the transcript
        committed by ``turn.end`` may differ from this one. Pair the service with
        :class:`~pipecat.turns.user_turn_strategies.EagerUserTurnStrategies` to
        have a response generated here and discarded if either happens.
        """
        transcript = data.get("transcript", "")
        await self._push_eager_end_of_turn(
            transcript, user_id=self._user_id, language=self._language, result=data
        )
        await self._call_event_handler("on_turn_eager_end", transcript)

    async def _handle_turn_resume(self, data: dict):
        """Handle the user resuming a turn that was eagerly predicted to have ended."""
        logger.trace("Cartesia Ink-2 ASR turn.resume")
        await self._cancel_eager_end_of_turn()
        await self._call_event_handler("on_turn_resume")

    async def _handle_turn_end(self, data: dict):
        transcript = data.get("transcript", "")
        logger.debug(f"Cartesia Ink-2 ASR turn.end: {transcript}")
        self._user_is_speaking = False
        self._turn_transcript = ""
        # The turn is committed, so any eager prediction it followed is resolved.
        self._clear_eager_end_of_turn()
        # The watchdog injects silence to force turn.end when audio stops
        # mid-turn, so a turn that captured only silence/noise can end with
        # an empty transcript. Skip the TranscriptionFrame in that case to
        # avoid an empty user message downstream; the lifecycle frames below
        # still fire so the turn closes cleanly.
        if transcript:
            # Report usage before the transcription frame so tracing can
            # attach it to the STT span the frame closes.
            await self.emit_stt_usage_metrics()
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    self._language,
                    result=data,
                    finalized=True,
                )
            )
            await self._handle_transcription(transcript, True, self._language)
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        await self._call_event_handler("on_turn_end", transcript)

    async def _handle_server_error(self, data: dict):
        message = data.get("message", "Unknown error")
        error_code = data.get("error_code", "unknown")
        error_msg = f"Cartesia Ink-2 ASR error [{error_code}]: {message}"
        if not self._connection_established_event.is_set():
            # The server validates a connection's settings after the
            # handshake, sending this error in place of ``connected`` and then
            # closing the connection.
            raise CartesiaTurnsRequestError(error_msg, status_code=data.get("status_code"))
        await self.push_error(error_msg=error_msg)
