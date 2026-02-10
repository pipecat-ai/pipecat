"""Continuous audio streaming variant of OjinVideoService.

This service sends TTS audio to the inference server on-demand.
The server maintains a virtual timeline at 25fps and generates silence
when no client audio is available.
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


@dataclass
class OjinLatencyFrame(Frame):
    """Frame indicating that the service has been initialized."""

    latency: float


class OjinBotStartedSpeakingFrame(Frame):
    """Frame sent after X seconds of receiving TTS audio."""

    pass


class OjinBotStoppedSpeakingFrame(Frame):
    """Frame sent after Y seconds of running out of TTS audio."""

    pass


OJIN_PERSONA_SAMPLE_RATE = 16000
SPEECH_FILTER_AMOUNT = 5
SPEECH_MOUTH_OPENING_SCALE = 1.0


@dataclass
class VideoFrame:
    """Represents a video frame with bundled audio from the server."""

    frame_idx: int
    image_bytes: bytes
    audio_bytes: bytes
    is_final: bool


@dataclass
class OjinVideoSettings:
    """Settings for Ojin Video Continuous Service."""

    api_key: str = field(default="")
    ws_url: str = field(default="wss://models.ojin.ai/realtime")
    client_connect_max_retries: int = field(default=3)
    client_reconnect_delay: float = field(default=3.0)
    config_id: str = field(default="")
    image_size: Tuple[int, int] = field(default=(1280, 720))
    tts_audio_passthrough: bool = field(default=False)
    # Speaking notification delays
    started_speaking_delay_s: float = field(default=0.5)  # Delay before sending StartedSpeaking
    stopped_speaking_delay_s: float = field(default=0.5)  # Delay before sending StoppedSpeaking
    frame_debugging_enabled: bool = field(default=False)


class OjinVideoService(FrameProcessor):
    """Continuous audio streaming service.

    Sends TTS audio to the server on-demand:
    - Client sends TTS audio immediately when received
    - Server maintains virtual timeline at 25fps
    - Server generates silence when no client audio available
    """

    def __init__(
        self,
        settings: OjinVideoSettings,
        client: IOjinClient | None = None,
    ) -> None:
        super().__init__()
        logger.debug(f"OjinVideoService initialized with settings {settings}")

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
        self._playback_task: Optional[asyncio.Task] = None
        self._stopped = False

        # Server tracking
        self._server_fps_tracker = FPSTracker("OjinVideoService")

        # Frame timing
        self._frame_duration = 0.04
        self._audio_bytes_per_frame = (
            640  # 2 * int(self._frame_duration * OJIN_PERSONA_SAMPLE_RATE)
        )

        # Speaking state tracking
        self._first_tts_received_at: Optional[float] = None
        self._latency_start_ts: Optional[float] = None
        self._latency: Optional[float] = None
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

        await self.push_error(
            error_msg=f"Failed to connect after {self._settings.client_connect_max_retries} attempts. error: {last_error}",
            fatal=True,
        )
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
            if self._latency_start_ts is None:
                self._latency_start_ts = time.perf_counter()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            await self._send_tts_audio(frame)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _send_tts_audio(self, frame: TTSAudioRawFrame):
        """Send TTS audio to the server immediately.

        The server maintains a virtual timeline and will generate silence
        when no client audio is available.
        """
        # Resample audio to target sample rate
        resampled_audio = await self._resampler.resample(
            frame.audio, frame.sample_rate, OJIN_PERSONA_SAMPLE_RATE
        )

        # Track first TTS audio for started speaking notification
        if self._first_tts_received_at is None:
            self._first_tts_received_at = time.perf_counter()
            # logger.debug("First TTS audio received, starting speaking timer")

        # Reset empty timer since we have audio
        self._tts_empty_since = None

        # Send audio to server immediately
        # logger.debug(f"Sending TTS audio to server, size: {len(resampled_audio)} bytes")
        await self._client.send_message(
            OjinAudioInputMessage(
                audio_int16_bytes=resampled_audio,
                params={
                    "filter_amount": SPEECH_FILTER_AMOUNT,
                    "mouth_opening_scale": SPEECH_MOUTH_OPENING_SCALE,
                },
            )
        )

        if self._settings.tts_audio_passthrough:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def _handle_ojin_message(self, message: BaseModel):
        """Process incoming messages from the server."""
        if isinstance(message, OjinSessionReadyMessage):
            if message.parameters is not None:
                self._session_data = message.parameters

            logger.info(f"Received Session Ready: {message}")
            if self._session_data and self._session_data.get("server_id"):
                logger.info(f"Connected to server: {self._session_data.get('server_id')}")

            self._server_fps_tracker.start()

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
            if self._settings.frame_debugging_enabled:
                logger.debug(
                    f"Received video frame {frame_idx}, is_final={message.is_final_response}"
                )
            if self._latency_start_ts is not None and video_frame.frame_idx == 1:
                self._latency = time.perf_counter() - self._latency_start_ts
                self._latency_start_ts = None
                await self.push_frame(
                    OjinLatencyFrame(latency=self._latency), direction=FrameDirection.DOWNSTREAM
                )
                self._latency = None

        elif isinstance(message, ErrorResponseMessage):
            await self.push_error(
                error_msg=f"Ojin server error: {message.payload.code}", fatal=True
            )

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

        start_ts = time.perf_counter()
        next_frame_time = start_ts + self._frame_duration
        initial_buffer_filled = False
        is_silence = False
        frame_count = 0
        # Determine which frame to play
        image_bytes: Optional[bytes] = None
        audio_bytes: Optional[bytes] = None
        skip_count = 0
        MAX_FRAMES_BUFFER = 10

        while not self._stopped:
            # Check speaking state notifications
            await self._check_started_speaking()
            await self._check_stopped_speaking()

            # Sleep for most of the wait time
            now = time.perf_counter()
            sleep_time = next_frame_time - now - 0.01
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Spin lock for precise timing
            while time.perf_counter() < next_frame_time:
                pass

            next_frame_time += self._frame_duration

            if not initial_buffer_filled:
                if len(self._video_frames) >= MAX_FRAMES_BUFFER:
                    initial_buffer_filled = True
                else:
                    continue

            # Check if we have a ready video frame
            if len(self._video_frames) > 0:
                video_frame = self._video_frames.popleft()
                frame_count += 1
                image_bytes = video_frame.image_bytes
                audio_bytes = video_frame.audio_bytes
                self._last_played_image_bytes = image_bytes
                is_silence = video_frame.frame_idx == 0
                if self._settings.frame_debugging_enabled:
                    if is_silence:
                        logger.debug(
                            f"[SILENCE] Playing frame {frame_count}, buffer left: {len(self._video_frames)}"
                        )
                    else:
                        logger.debug(
                            f"[SPEECH] Playing frame {frame_count}, buffer left: {len(self._video_frames)}"
                        )
                all_silence = len(self._video_frames) > MAX_FRAMES_BUFFER and all(
                    f.frame_idx == 0 for f in self._video_frames
                )
                if all_silence:
                    skip_count += 1
                    if skip_count % 2 == 0:
                        self._video_frames.popleft()
                        logger.debug(
                            f"Skipping silence frame: {len(self._video_frames)}, target: {MAX_FRAMES_BUFFER}"
                        )

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

            elif self._last_played_image_bytes:
                # Repeat last frame to avoid stutter
                logger.debug(f"frame miss, repeating frame {frame_count}")
                continue
            else:
                # No frame to show yet - just continue
                continue

    async def _start(self):
        """Initialize the service and start processing."""
        is_connected = await self.connect_with_retry()
        if not is_connected:
            return

        # Start receiving messages
        self._receive_msg_task = self.create_task(self._receive_ojin_messages())

        # Start an interaction for continuous streaming
        await self._client.start_interaction()

    async def _stop(self):
        self._stopped = True
        if self._client:
            await self._client.close()
            self._client = None

        """Stop the service and clean up resources."""
        if self._receive_msg_task:
            await self.cancel_task(self._receive_msg_task)
            self._receive_msg_task = None

        if self._playback_task:
            await self.cancel_task(self._playback_task)
            self._playback_task = None

        logger.debug(f"OjinVideoService {self._settings.config_id} stopped")

    async def _check_started_speaking(self):
        """Check if we should send StartedSpeaking frame."""
        if self._bot_speaking:
            return

        if self._first_tts_received_at is None:
            return

        elapsed = time.perf_counter() - self._first_tts_received_at
        if elapsed >= self._settings.started_speaking_delay_s:
            self._bot_speaking = True
            # logger.info(f"Bot started speaking after {elapsed:.3f}s")
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
            # logger.info(f"Bot stopped speaking after {elapsed:.3f}s of silence")
            await self.push_frame(OjinBotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(OjinBotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
