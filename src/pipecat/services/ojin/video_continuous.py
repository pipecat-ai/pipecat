"""Continuous audio streaming variant of OjinVideoService.

This service sends a continuous stream of audio to the inference server.
When TTS audio is available it sends that, otherwise it sends silence.
The server treats all audio the same - no differentiation between speech and silence.
"""

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from loguru import logger
from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    IOjinClient,
    OjinAudioInputMessage,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
)
from ojin.profiling_utils import FPSTracker
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class OjinVideoInitializedFrame(Frame):
    """Frame indicating that the service has been initialized."""

    session_data: Optional[dict] = None


class OjinBotStartedSpeakingFrame(Frame):
    """Frame sent after X seconds of receiving TTS audio."""

    pass


class OjinBotStoppedSpeakingFrame(Frame):
    """Frame sent after Y seconds of running out of TTS audio."""

    pass


OJIN_PERSONA_SAMPLE_RATE = 16000
SPEECH_FILTER_AMOUNT = 1000.0
SPEECH_MOUTH_OPENING_SCALE = 1.0


class OjinVideoContinuousState(Enum):
    """Service states."""

    CONNECTING = "connecting"
    STREAMING = "streaming"


@dataclass
class VideoFrame:
    """Represents a video frame with bundled audio from the server."""

    frame_idx: int
    image_bytes: bytes
    audio_bytes: bytes
    is_final: bool


@dataclass
class OjinVideoContinuousSettings:
    """Settings for Ojin Video Continuous Service."""

    api_key: str = field(default="")
    ws_url: str = field(default="wss://models.ojin.ai/realtime")
    client_connect_max_retries: int = field(default=3)
    client_reconnect_delay: float = field(default=3.0)
    config_id: str = field(default="")
    image_size: Tuple[int, int] = field(default=(1280, 720))
    tts_audio_passthrough: bool = field(default=False)
    # Audio streaming settings
    audio_chunk_duration_ms: int = field(default=500)  # Duration of each audio chunk in ms
    # Speaking notification delays
    started_speaking_delay_s: float = field(default=0.5)  # Delay before sending StartedSpeaking
    stopped_speaking_delay_s: float = field(default=0.5)  # Delay before sending StoppedSpeaking


class OjinVideoContinuousService(FrameProcessor):
    """Continuous audio streaming service.

    Sends a continuous stream of audio to the server:
    - When TTS audio is available, sends TTS audio
    - When no TTS audio, sends silence
    - Server treats all audio the same (no is_speech differentiation)
    """

    def __init__(
        self,
        settings: OjinVideoContinuousSettings,
        client: IOjinClient | None = None,
    ) -> None:
        super().__init__()
        logger.debug(f"OjinVideoContinuousService initialized with settings {settings}")

        self._settings = settings
        if client is None:
            self._client = OjinClient(
                ws_url=settings.ws_url,
                api_key=settings.api_key,
                config_id=settings.config_id,
                mode=os.getenv("OJIN_MODE", ""),
            )
        else:
            self._client = client

        self._state = OjinVideoContinuousState.CONNECTING
        self._session_data: Optional[dict] = None

        # Video frames queue from server
        self._video_frames: deque[VideoFrame] = deque()

        # Speech audio buffer - raw bytes ready to send
        self._speech_buffer: bytearray = bytearray()

        # Playback state
        self.fps = 25
        self._current_frame_idx = -1
        self._played_frame_idx = -1
        self._last_played_image_bytes: Optional[bytes] = None

        # Audio input handling
        self._resampler = create_default_resampler()

        # Tasks
        self._receive_msg_task: Optional[asyncio.Task] = None
        self._audio_input_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None

        # Server tracking
        self._server_fps_tracker = FPSTracker("OjinVideoContinuousService")

        # Audio streaming constants
        self._chunk_duration_s = settings.audio_chunk_duration_ms / 1000.0
        self._chunk_size_bytes = 2 * int(self._chunk_duration_s * OJIN_PERSONA_SAMPLE_RATE)
        self._silence_chunk = b"\x00" * self._chunk_size_bytes

        # Frame timing
        self._frame_duration = 1.0 / self.fps
        self._audio_bytes_per_frame = 2 * int(self._frame_duration * OJIN_PERSONA_SAMPLE_RATE)

        # Speaking state tracking
        self._first_tts_received_at: Optional[float] = None
        self._tts_empty_since: Optional[float] = None
        self._bot_speaking = False

    async def connect_with_retry(self) -> bool:
        """Attempt to connect with configurable retry mechanism."""
        last_error: Optional[Exception] = None
        assert self._client is not None

        for attempt in range(self._settings.client_connect_max_retries):
            try:
                logger.info(
                    f"Connection attempt {attempt + 1}/{self._settings.client_connect_max_retries}"
                )
                await self._client.connect()
                logger.info("Successfully connected!")
                return True

            except ConnectionError as e:
                last_error = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < self._settings.client_connect_max_retries - 1:
                    logger.info(f"Retrying in {self._settings.client_reconnect_delay} seconds...")
                    await asyncio.sleep(self._settings.client_reconnect_delay)

        logger.error(
            f"Failed to connect after {self._settings.client_connect_max_retries} attempts. "
            f"Last error: {last_error}"
        )
        await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            logger.debug("StartFrame received")
            await self.push_frame(frame, direction)
            await self._start()

        elif isinstance(frame, TTSStartedFrame):
            logger.debug("TTSStartedFrame received")
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            await self._buffer_tts_audio(frame)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _buffer_tts_audio(self, frame: TTSAudioRawFrame):
        """Buffer incoming TTS audio for continuous streaming."""
        # Resample audio to target sample rate
        resampled_audio = await self._resampler.resample(
            frame.audio, frame.sample_rate, OJIN_PERSONA_SAMPLE_RATE
        )

        # Add to speech buffer
        self._speech_buffer.extend(resampled_audio)
        logger.debug(f"Buffered TTS audio, buffer size: {len(self._speech_buffer)} bytes")

        # Track first TTS audio for started speaking notification
        if self._first_tts_received_at is None:
            self._first_tts_received_at = time.perf_counter()
            logger.debug("First TTS audio received, starting speaking timer")

        # Reset empty timer since we have audio
        self._tts_empty_since = None

        if self._settings.tts_audio_passthrough:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def _audio_input_loop(self):
        """Continuously send audio to the server at a fixed rate.

        Sends speech audio when available, silence otherwise.
        """
        logger.info(
            f"Starting audio input loop - chunk_size={self._chunk_size_bytes} bytes, "
            f"rate={self._chunk_duration_s * 1000}ms"
        )

        # Wait for session to be ready
        while self._state == OjinVideoContinuousState.CONNECTING:
            await asyncio.sleep(0.01)

        # Start an interaction for continuous streaming
        await self._client.start_interaction()

        next_send_time = time.perf_counter() + self._chunk_duration_s

        while True:
            # Sleep until next send time
            now = time.perf_counter()
            sleep_time = next_send_time - now - 0.002
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Spin lock for precise timing
            while time.perf_counter() < next_send_time:
                pass

            next_send_time += self._chunk_duration_s

            # Check if we should send StartedSpeaking
            await self._check_started_speaking()

            # Check if we should send StoppedSpeaking
            await self._check_stopped_speaking()

            # Get audio chunk to send - dequeue all available speech audio and pad with silence
            if len(self._speech_buffer) >= self._chunk_size_bytes:
                # Send speech audio
                audio_chunk = bytes(self._speech_buffer[: self._chunk_size_bytes])
                del self._speech_buffer[: self._chunk_size_bytes]
                logger.debug(
                    f"Sending speech chunk, remaining buffer: {len(self._speech_buffer)} bytes"
                )

            else:
                # Send silence
                audio_chunk = self._silence_chunk

            # Send audio to server
            await self._client.send_message(
                OjinAudioInputMessage(
                    audio_int16_bytes=audio_chunk,
                    params={
                        "filter_amount": SPEECH_FILTER_AMOUNT,
                        "mouth_opening_scale": SPEECH_MOUTH_OPENING_SCALE,
                    },
                )
            )

    async def _handle_ojin_message(self, message: BaseModel):
        """Process incoming messages from the server."""
        if isinstance(message, OjinSessionReadyMessage):
            if message.parameters is not None:
                self._session_data = message.parameters

            logger.info(f"Received Session Ready: {message}")
            if self._session_data and self._session_data.get("server_id"):
                logger.info(f"Connected to server: {self._session_data.get('server_id')}")

            self._server_fps_tracker.start()

            # Transition to streaming state
            self._state = OjinVideoContinuousState.STREAMING

            # Start the playback loop
            self._playback_task = self.create_task(self._playback_loop())

            # Notify that we're ready
            initialized_frame = OjinVideoInitializedFrame(session_data=self._session_data)
            await self.push_frame(initialized_frame, direction=FrameDirection.DOWNSTREAM)
            await self.push_frame(initialized_frame, direction=FrameDirection.UPSTREAM)

        elif isinstance(message, OjinInteractionResponseMessage):
            frame_idx = message.index

            # Queue video frame with bundled audio
            video_frame = VideoFrame(
                frame_idx=frame_idx,
                image_bytes=message.video_frame_bytes,
                audio_bytes=message.audio_frame_bytes,
                is_final=message.is_final_response,
            )

            self._video_frames.append(video_frame)
            logger.debug(f"Received video frame {frame_idx}, is_final={message.is_final_response}")

        elif isinstance(message, ErrorResponseMessage):
            is_fatal = False
            if message.payload.code == "NO_BACKEND_SERVER_AVAILABLE":
                logger.error("No OJIN servers available. Please try again later.")
                is_fatal = True
            elif message.payload.code == "FRAME_SIZE_TOO_BIG":
                logger.error("Frame Size sent to Ojin server exceeded max limit.")
            elif message.payload.code == "INVALID_INTERACTION_ID":
                logger.error("Invalid interaction ID sent to Ojin server")
            elif message.payload.code == "FAILED_CREATE_MODEL":
                is_fatal = True
                logger.error("Ojin couldn't create a model from supplied persona ID.")
            elif message.payload.code == "INVALID_PERSONA_ID_CONFIGURATION":
                is_fatal = True
                logger.error("Ojin couldn't load the configuration from the supplied persona ID.")

            if is_fatal:
                await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
                await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    async def _receive_ojin_messages(self):
        """Continuously receive and process messages from the server."""
        while True:
            assert self._client is not None
            message = await self._client.receive_message()
            if message is not None:
                await self._handle_ojin_message(message)

    async def _playback_loop(self):
        """Main playback loop - outputs video frames and audio at fixed fps."""
        logger.info("Starting playback loop")

        silence_audio = b"\x00" * self._audio_bytes_per_frame

        start_ts = time.perf_counter()
        next_frame_time = start_ts + self._frame_duration

        while True:
            # Sleep for most of the wait time
            now = time.perf_counter()
            sleep_time = next_frame_time - now - 0.01
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Spin lock for precise timing
            while time.perf_counter() < next_frame_time:
                pass

            next_frame_time += self._frame_duration

            # Determine which frame to play
            image_bytes: Optional[bytes] = None
            audio_bytes = silence_audio

            # Check if we have a ready video frame
            if len(self._video_frames) > 0:
                video_frame = self._video_frames.popleft()
                image_bytes = video_frame.image_bytes
                audio_bytes = video_frame.audio_bytes if video_frame.audio_bytes else silence_audio
                self._last_played_image_bytes = image_bytes
                logger.debug(f"Playing video frame {video_frame.frame_idx}")

            elif self._last_played_image_bytes:
                # Repeat last frame to avoid stutter
                image_bytes = self._last_played_image_bytes

            else:
                # No frame to show yet - just continue
                continue

            if image_bytes:
                image_frame = OutputImageRawFrame(
                    image=image_bytes, size=self._settings.image_size, format="RGB"
                )
                image_frame.pts = next_frame_time
                await self.push_frame(image_frame)

            audio_frame = OutputAudioRawFrame(
                audio=audio_bytes,
                sample_rate=OJIN_PERSONA_SAMPLE_RATE,
                num_channels=1,
            )
            audio_frame.pts = next_frame_time
            await self.push_frame(audio_frame)

    async def _start(self):
        """Initialize the service and start processing."""
        is_connected = await self.connect_with_retry()
        if not is_connected:
            return

        # Start receiving messages
        self._receive_msg_task = self.create_task(self._receive_ojin_messages())

        # Start continuous audio input
        self._audio_input_task = self.create_task(self._audio_input_loop())

    async def _stop(self):
        """Stop the service and clean up resources."""
        if self._audio_input_task:
            await self.cancel_task(self._audio_input_task)
            self._audio_input_task = None

        if self._receive_msg_task:
            await self.cancel_task(self._receive_msg_task)
            self._receive_msg_task = None

        if self._playback_task:
            await self.cancel_task(self._playback_task)
            self._playback_task = None

        if self._client:
            await self._client.close()
            self._client = None

        logger.debug(f"OjinVideoContinuousService {self._settings.config_id} stopped")

    async def _check_started_speaking(self):
        """Check if we should send StartedSpeaking frame."""
        if self._bot_speaking:
            return

        if self._first_tts_received_at is None:
            return

        elapsed = time.perf_counter() - self._first_tts_received_at
        if elapsed >= self._settings.started_speaking_delay_s:
            self._bot_speaking = True
            logger.info(f"Bot started speaking after {elapsed:.3f}s")
            await self.push_frame(OjinBotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(OjinBotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _check_stopped_speaking(self):
        """Check if we should send StoppedSpeaking frame."""
        if not self._bot_speaking:
            return

        # Start tracking empty time if not already
        if self._tts_empty_since is None:
            self._tts_empty_since = time.perf_counter()
            return

        elapsed = time.perf_counter() - self._tts_empty_since
        if elapsed >= self._settings.stopped_speaking_delay_s:
            self._bot_speaking = False
            self._first_tts_received_at = None  # Reset for next speech
            self._tts_empty_since = None
            logger.info(f"Bot stopped speaking after {elapsed:.3f}s of silence")
            await self.push_frame(OjinBotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(OjinBotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
