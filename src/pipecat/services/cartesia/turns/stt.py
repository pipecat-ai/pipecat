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
from dataclasses import dataclass
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

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
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


@dataclass
class CartesiaTurnsSTTSettings(STTSettings):
    """Settings for CartesiaTurnsSTTService.

    The ink-2 model family is English-only and does not support runtime model or language switching,
    so no fields are added beyond the inherited :class:`STTSettings`.
    """

    pass


class CartesiaTurnsSTTService(WebsocketSTTService):
    """Speech-to-text service using the Cartesia Streaming ASR v2 (Ink-2) API.

    Speaks the v2 turn-based wire protocol exposed by
    ``/stt/turns/websocket``. The server drives the conversation::

        connected -> turn.start -> turn.update* -> (turn.eager_end -> turn.resume?)*
                                -> turn.end -> ...

    Transcripts are cumulative per turn; there is no ``is_final`` flag and no
    ``finalize`` command — closing the socket ends the session.

    Each ``turn.start`` pushes a :class:`UserStartedSpeakingFrame`; each
    ``turn.update`` pushes an :class:`InterimTranscriptionFrame`; ``turn.end``
    pushes a final :class:`TranscriptionFrame` followed by a
    :class:`UserStoppedSpeakingFrame`. ``turn.eager_end`` and ``turn.resume``
    are surfaced only via their respective event handlers.

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

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.cartesia.ai/stt/turns/websocket",
        sample_rate: int | None = None,
        should_interrupt: bool = True,
        watchdog_min_timeout: float = 0.5,
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
            should_interrupt: Whether to broadcast an interruption when the
                server signals the start of a new turn.
            watchdog_min_timeout: Minimum idle timeout before sending silence to
                prevent dangling turns. The actual threshold is
                ``max(chunk_duration * 2, watchdog_min_timeout)``. Defaults to 0.5.
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
        )

        if settings is not None:
            default_settings.apply_update(settings)

        # reconnect_on_error=False: we want the server's "connected" frame
        # before declaring the socket ready; send_with_retry handles
        # reconnection on demand.
        super().__init__(
            sample_rate=sample_rate,
            reconnect_on_error=False,
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        """Start the STT service and establish the WebSocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
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

        Ink-2 does not support runtime model or language switching, so any
        changed fields are reported as unhandled.

        Args:
            delta: A :class:`STTSettings` (or
                :class:`CartesiaTurnsSTTSettings`) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        self._warn_unhandled_updated_settings(changed.keys())
        return changed

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()

    async def _disconnect(self):
        await super()._disconnect()
        try:
            await self._disconnect_websocket()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None

    def _websocket_url(self) -> str:
        # Pipecat pipes 16-bit signed little-endian PCM through the pipeline,
        # so the wire encoding is fixed.
        params = {
            "model": self._settings.model,
            "encoding": "pcm_s16le",
            "sample_rate": str(self.sample_rate),
        }
        return f"{self._url}?{urllib.parse.urlencode(params)}"

    async def _connect_websocket(self):
        """Connect to the v2 WebSocket and wait for the server's ``connected`` frame."""
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
            self._websocket = await websocket_connect(
                self._websocket_url(), additional_headers=headers
            )

            if not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error)
                )

            if not self._watchdog_task:
                self._watchdog_task = self.create_task(self._watchdog_task_handler())

            logger.debug("WebSocket connected, waiting for server confirmation...")
            await self._connection_established_event.wait()
            logger.debug(f"Connected to Cartesia Ink-2 ASR (request_id={self._request_id})")
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Send a ``close`` command, drain pending responses, then close the WebSocket."""
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

            # Wait for the receive task to exit naturally as the server drains
            # responses and closes the socket. Fall back to cancellation if it
            # doesn't complete in time.
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

            self._connection_established_event.clear()
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Cartesia Ink-2 ASR")
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

        Args:
            audio: Raw 16-bit signed little-endian PCM audio bytes.

        Yields:
            Frame: ``None`` (transcription results are pushed by the receive
                task), or :class:`ErrorFrame` on send failure.
        """
        if not self._websocket:
            return
        try:
            self._last_stt_time = time.monotonic()
            self._last_audio_chunk_duration = len(audio) / (self.sample_rate * 2)
            await self.send_with_retry(audio, self._report_error)
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
        await self.send_with_retry(silence, self._report_error)

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
        """Receive and process messages from the WebSocket."""
        async for message in self._get_websocket():
            if not isinstance(message, str):
                logger.warning(f"Received non-text message: {type(message)}")
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.exception("Failed to decode JSON message")
                continue
            try:
                await self._handle_message(data)
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
                # Re-raise so WebsocketService._receive_task_handler tears down
                # the receive loop. With reconnect_on_error=False, it reports
                # the error and exits — no reconnect happens here.
                raise

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
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()
        await self.start_processing_metrics()
        await self._call_event_handler("on_turn_start", transcript)

    async def _handle_turn_update(self, data: dict):
        transcript = data.get("transcript", "")
        if transcript:
            logger.trace(f"Cartesia Ink-2 ASR turn.update: {transcript}")
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
        transcript = data.get("transcript", "")
        logger.trace(f"Cartesia Ink-2 ASR turn.eager_end: {transcript}")
        await self._call_event_handler("on_turn_eager_end", transcript)

    async def _handle_turn_resume(self, data: dict):
        logger.trace("Cartesia Ink-2 ASR turn.resume")
        await self._call_event_handler("on_turn_resume")

    async def _handle_turn_end(self, data: dict):
        transcript = data.get("transcript", "")
        logger.debug(f"Cartesia Ink-2 ASR turn.end: {transcript}")
        self._user_is_speaking = False
        # The watchdog injects silence to force turn.end when audio stops
        # mid-turn, so a turn that captured only silence/noise can end with
        # an empty transcript. Skip the TranscriptionFrame in that case to
        # avoid an empty user message downstream; the lifecycle frames below
        # still fire so the turn closes cleanly.
        if transcript:
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
        await self.stop_processing_metrics()
        await self.broadcast_frame(UserStoppedSpeakingFrame)
        await self._call_event_handler("on_turn_end", transcript)

    async def _handle_server_error(self, data: dict):
        message = data.get("message", "Unknown error")
        error_code = data.get("error_code", "unknown")
        await self.push_error(error_msg=f"Cartesia Ink-2 ASR error [{error_code}]: {message}")
