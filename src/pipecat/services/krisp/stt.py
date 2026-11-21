#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp real-time speech-to-text service implementation.

Streams 16 kHz PCM audio to Krisp's STT WebSocket API. Krisp predicts where
each user turn ends; finals for a turn are accumulated and pushed as one
``TranscriptionFrame`` when the server sends its ``turn`` message.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.krisp.api import (
    KRISP_REST_BASE_URL,
    KrispServiceError,
    create_session_token,
    open_session,
    session_headers,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_start_strategies,
)
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven

KRISP_STT_URL = "wss://streaming.krisp.ai/stt"

# The API accepts 16 kHz mono PCM only, ideally in 80 ms chunks.
KRISP_STT_SAMPLE_RATE = 16000
_CHUNK_BYTES = KRISP_STT_SAMPLE_RATE * 2 * 80 // 1000

# Audio that can't be sent yet (e.g. during a reconnect) is kept and sent once
# the connection is ready; only the newest 5 seconds are kept.
_MAX_PENDING_AUDIO_BYTES = KRISP_STT_SAMPLE_RATE * 2 * 5

# How long to wait for the server's ``end`` message after ``end_of_audio``.
_END_OF_STREAM_TIMEOUT = 3.0


@dataclass
class KrispSTTSettings(STTSettings):
    """Settings for KrispSTTService.

    ``language`` is sent as a language hint: it restricts the scripts the
    transcript may use rather than selecting a language.

    Parameters:
        vocabulary: Custom vocabulary terms.
        turn_prediction: Whether the server predicts end of turn. When True,
            the server's turn events end pipecat's user turns; when False, the
            service asks the server to finalize whenever local VAD detects the
            user stopped speaking. Turn start and bot interruption are always
            left to pipecat's start strategies.
        turn_sensitivity: End-of-turn sensitivity, 0 (more sensitive) to 1.
        turn_max_silence_ms: Silence in ms after which the turn ends (400-4000).
        turn_silence_validation_ms: Silence in ms used to validate the
            turn model's decision (10-1000).
    """

    vocabulary: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_prediction: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_sensitivity: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_max_silence_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_silence_validation_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class KrispSTTService(WebsocketSTTService):
    """Krisp real-time speech-to-text service.

    Pushes ``InterimTranscriptionFrame`` with the turn's finals followed by the
    latest partial hypothesis, and one ``TranscriptionFrame`` per user turn.
    With ``turn_prediction`` enabled (the default) the service recommends user
    turn strategies that keep pipecat's default start strategies (which also
    interrupt the bot) and end turns on Krisp's end-of-turn events.
    Reconnection and silence keepalive (enabled with ``keepalive_timeout``)
    come from :class:`WebsocketSTTService`.

    Audio at other sample rates is resampled to 16 kHz.
    """

    Settings = KrispSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = KRISP_STT_URL,
        rest_base_url: str = KRISP_REST_BASE_URL,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Krisp STT service.

        Args:
            api_key: Krisp master API key. It is only used to create a session
                token before each connection and never sent on the WebSocket.
            url: Krisp STT WebSocket endpoint.
            rest_base_url: Krisp REST API base URL used to create session tokens.
            settings: Runtime-updatable settings. Changing any of them
                reconnects, since they are fixed by the session handshake.
            **kwargs: Additional arguments passed to :class:`WebsocketSTTService`
                (e.g. ``keepalive_timeout``, ``ttfs_p99_latency``).
        """
        default_settings = self.Settings(
            model="krisp-realtime-stt-agents-v1",
            language=None,
            vocabulary=None,
            turn_prediction=True,
            turn_sensitivity=0.9,
            turn_max_silence_ms=2000,
            turn_silence_validation_ms=100,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._api_key = api_key
        self._url = url
        self._rest_base_url = rest_base_url

        self._receive_task = None
        self._input_resampler = create_stream_resampler()
        self._audio_buffer = bytearray()
        self._send_lock = asyncio.Lock()
        # Set once end_of_audio is sent; no audio may follow it.
        self._audio_ended = False
        self._end_received = asyncio.Event()

        # Transcript state for the current user turn.
        self._final_text = ""
        self._partial_text = ""
        self._detected_language: Language | None = None
        self._in_turn = False

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Krisp STT supports metrics generation.
        """
        return True

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend ending user turns on Krisp's end-of-turn events.

        Turn start (and with it bot interruption) stays with pipecat's default
        start strategies. Applied unless the application passes its own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        if self._settings.turn_prediction:
            frame.user_turn_strategies = UserTurnStrategies(
                start=default_user_turn_start_strategies(),
                stop=[ExternalUserTurnStopStrategy()],
            )
        return frame

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect with a new handshake.

        Args:
            delta: A :class:`STTSettings` (or ``KrispSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
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
        """End the audio stream gracefully, then disconnect.

        Sends ``end_of_audio`` and waits for the server's ``end`` message so the
        last turn's transcript is delivered before the pipeline ends.

        Args:
            frame: The end frame.
        """
        await self._end_audio_stream()
        await super().stop(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, finalizing on VAD stop when turn prediction is off.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame) and not self._settings.turn_prediction:
            await self._send_json({"type": "finalize"})

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to Krisp for transcription.

        Args:
            audio: Raw 16-bit mono PCM audio at the pipeline sample rate.

        Yields:
            None, since transcriptions arrive on the receive task.
        """
        await self._send_audio(audio)
        yield None

    async def _send_keepalive(self, silence: bytes):
        await self._send_audio(silence)
        self._record_stt_audio_usage(silence)

    async def _send_audio(self, audio: bytes):
        if self.sample_rate != KRISP_STT_SAMPLE_RATE:
            audio = await self._input_resampler.resample(
                audio, self.sample_rate, KRISP_STT_SAMPLE_RATE
            )
        # run_stt and the keepalive task both send; the lock keeps a chunk from
        # being sent twice.
        async with self._send_lock:
            if self._audio_ended:
                return
            self._audio_buffer.extend(audio)
            excess = len(self._audio_buffer) - _MAX_PENDING_AUDIO_BYTES
            if excess > 0:
                del self._audio_buffer[: excess + excess % 2]
            while len(self._audio_buffer) >= _CHUNK_BYTES:
                if not self._websocket or self._websocket.state is not State.OPEN:
                    break
                try:
                    await self._websocket.send(bytes(self._audio_buffer[:_CHUNK_BYTES]))
                except Exception as e:
                    logger.warning(f"{self}: send failed: {e}")
                    break
                del self._audio_buffer[:_CHUNK_BYTES]

    async def _send_json(self, msg: dict):
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.warning(f"{self}: failed to send {msg.get('type')}: {e}")

    async def _end_audio_stream(self):
        websocket = self._websocket
        if not websocket or websocket.state is not State.OPEN:
            return
        # The server closes the socket after "end"; that close is expected.
        self._disconnecting = True
        self._end_received.clear()
        try:
            async with self._send_lock:
                self._audio_ended = True
                while self._audio_buffer:
                    await websocket.send(bytes(self._audio_buffer[:_CHUNK_BYTES]))
                    del self._audio_buffer[:_CHUNK_BYTES]
                await websocket.send(json.dumps({"type": "end_of_audio"}))
            await asyncio.wait_for(self._end_received.wait(), _END_OF_STREAM_TIMEOUT)
        except TimeoutError:
            logger.warning(f"{self}: no 'end' message within {_END_OF_STREAM_TIMEOUT}s")
        except Exception as e:
            logger.warning(f"{self}: failed to end the audio stream: {e}")

    def _build_handshake(self) -> dict[str, Any]:
        turn: dict[str, Any] = {"prediction": self._settings.turn_prediction}
        optional_turn_fields = {
            "sensitivity": self._settings.turn_sensitivity,
            "max_silence": self._settings.turn_max_silence_ms,
            "silence_validation": self._settings.turn_silence_validation_ms,
        }
        turn.update({k: v for k, v in optional_turn_fields.items() if v is not None})

        handshake: dict[str, Any] = {
            "model": self._settings.model,
            "sample_rate": KRISP_STT_SAMPLE_RATE,
            "channels": 1,
            "audio_format": "pcm_s16le",
            "turn": turn,
        }
        if self._settings.vocabulary:
            handshake["vocabulary"] = self._settings.vocabulary
        if self._settings.language:
            handshake["language_hints"] = [str(self._settings.language)]
        return handshake

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
        self._reset_stream_state()

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Push the dropped connection's finals, then reconnect.

        The server's transcript state goes with the connection, so the turn's
        finals received so far are pushed as a transcription. An open turn
        stays open for the new connection's turn event to end, and unsent audio
        stays buffered until the new connection is ready.

        Args:
            attempt_number: Current retry attempt number for logging.

        Returns:
            True if reconnection and verification succeeded.
        """
        in_turn = self._in_turn
        await self._push_turn_transcription({})
        self._in_turn = in_turn
        self._partial_text = ""
        return await super()._reconnect_websocket(attempt_number)

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"Connecting to Krisp STT ({self._url}, tokens: {self._rest_base_url})")
            token = await create_session_token(self._api_key, "stt", base_url=self._rest_base_url)
            websocket = await self._websocket_connect(
                self._url, additional_headers=session_headers(token)
            )
            try:
                ready = await open_session(websocket, self._build_handshake())
            except Exception:
                await websocket.close()
                raise
            # Exposed only once ready: no audio may be sent before that.
            self._audio_ended = False
            self._websocket = websocket
            logger.debug(f"Connected to Krisp STT (request_id={ready.get('request_id')})")
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Krisp STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug("Disconnecting from Krisp STT")
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            if self._websocket is ws:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _reset_stream_state(self):
        self._audio_buffer.clear()
        self._final_text = ""
        self._partial_text = ""
        self._detected_language = None
        self._in_turn = False

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON message: {message}")
                continue

            msg_type = msg.get("type")
            if msg_type == "transcript":
                await self._on_transcript(msg.get("transcript") or {})
            elif msg_type == "turn":
                await self._on_turn(msg)
            elif msg_type == "end":
                await self._push_turn_transcription(msg)
                self._end_received.set()
            elif msg_type == "error":
                error = KrispServiceError.from_message(msg)
                await self.push_error(error_msg=f"Krisp STT error: {error}", exception=error)

    async def _on_transcript(self, transcript: dict):
        words = transcript.get("words") or []
        # Word texts carry their leading space, so concatenation is the text.
        text = "".join(w.get("text", "") for w in words)

        if transcript.get("isFinal"):
            self._final_text += text
            for word in words:
                language = _parse_language(word.get("language"))
                if language:
                    self._detected_language = language
        else:
            # Each partial replaces the previous one.
            self._partial_text = text

        current = (self._final_text + self._partial_text).strip()
        if not current:
            return

        self._in_turn = True

        # A final shrinks the partial region, so the stale partial could repeat
        # the final's words; the next partial frame updates the interim text.
        if not transcript.get("isFinal"):
            await self.push_frame(
                InterimTranscriptionFrame(
                    current, self._user_id, time_now_iso8601(), self._language(), result=transcript
                )
            )

    async def _on_turn(self, msg: dict):
        was_in_turn = self._in_turn
        await self._push_turn_transcription(msg)
        if was_in_turn and self._settings.turn_prediction:
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _push_turn_transcription(self, result: dict):
        text = self._final_text.strip()
        language = self._language()
        self._final_text = ""
        self._partial_text = ""
        self._in_turn = False
        if not text:
            return

        # Report usage before the transcription frame so tracing can attach it
        # to the STT span the frame closes.
        await self.emit_stt_usage_metrics()
        await self.push_frame(
            TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), language, result=result, finalized=True
            )
        )
        await self._handle_transcription(text, True, language)

    def _language(self) -> Language | None:
        if self._detected_language:
            return self._detected_language
        language = self._settings.language
        return language if isinstance(language, Language) else None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Record a final transcription for tracing."""
        pass


def _parse_language(code: str | None) -> Language | None:
    if not code:
        return None
    try:
        return Language(code)
    except ValueError:
        return None
