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
from typing import Optional, Tuple, Type

import cv2
import numpy as np
from loguru import logger
from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_client import OjinClient
from ojin.ojin_client_messages import (
    IOjinClient,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
)
from ojin.profiling_utils import FPSTracker
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler, is_silence
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
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
BYTES_PER_FRAME = int(OJIN_PERSONA_SAMPLE_RATE / 25 * 2)
MAX_FRAMES_BUFFER = 7
MIN_FRAMES_BUFFER = 2


@dataclass
class VideoFrame:
    """Represents a video frame with bundled audio from the server."""

    frame_idx: int
    image_bytes: bytes
    audio_bytes: bytes
    is_final: bool
    volume: int
    is_first_speech_frame: bool = False

    def is_silence(self) -> bool:
        return self.frame_idx == 0


class InterruptStrategy(Enum):
    """Interruption strategy for the playback loop.

    SMOOTH: keep playing all buffered frames (video + audio).
    INSTANT_CUT: clear buffer, discard speech frames until first silence.
    SMOOTH_VIDEO_HARD_AUDIO: keep video playing, stop audio immediately.
    """

    SMOOTH = "smooth"
    INSTANT_CUT = "instant_cut"
    SMOOTH_VIDEO_HARD_AUDIO = "smooth_video_hard_audio"


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
    start_frame_cls: Type[Frame] = field(default=StartFrame)
    interrupt_strategy: InterruptStrategy = field(default=InterruptStrategy.INSTANT_CUT)


OJIN_VIDEO_SERVICE_VERSION = 20


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
        super().__init__(name="ojin")
        logger.debug(
            f"OjinVideoService initialized with settings {settings} version: {OJIN_VIDEO_SERVICE_VERSION}"
        )

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

        # Speech audio buffer - raw bytes ready to send (resampled to OJIN_PERSONA_SAMPLE_RATE)
        self._speech_buffer: bytearray = bytearray()

        # Playback state
        self.fps = 25
        self.fps_tracker = FPSTracker("Ojin")

        self._initialized = False
        self.last_frame_time = 0
        self._is_playing_speech_audio = False
        self._waiting_for_first_tts = False
        self._current_frame_idx = -1
        self._played_frame_idx = -1
        self._first_silence_frame: VideoFrame = None
        self._last_played_image_bytes: Optional[bytes] = None

        # Audio input handling
        self._resampler = create_default_resampler()

        # Tasks
        self._receive_msg_task: Optional[asyncio.Task] = None
        self._video_playback_task: Optional[asyncio.Task] = None

        # Frame timing
        self._frame_duration = 0.04

        # Turn-boundary detection flags (set on receive, consumed on playback)
        self._turn = 0
        self._last_frame_idx = -1
        self._pending_speech_start = False

        # Interruption state
        self._interrupting = False
        self._discard_speech_until_silence = False
        self._pending_latency_report: Optional[tuple[float, float]] = None

    def can_generate_metrics(self) -> bool:
        """Enable TTFB metrics reporting via the standard pipecat metrics system."""
        return True

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
        await self._stop()
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, self._settings.start_frame_cls):
            logger.debug(f"{self._settings.start_frame_cls.__name__} received")
            await self.push_frame(frame, direction)
            await self._start()

        elif isinstance(frame, TTSStartedFrame):
            logger.warning("TTSStartedFrame received")
            self._interrupting = False
            self._waiting_for_first_tts = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            if self._interrupting:
                logger.debug("Interrupting, dropping TTS audio")
                return
            await self._send_tts_audio(frame)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            if not self._interrupting and self._client is not None:
                logger.debug("Start interrupting")
                self._interrupting = True
                self._waiting_for_first_tts = False
                strategy = self._settings.interrupt_strategy

                if strategy == InterruptStrategy.SMOOTH:
                    # Keep playing all buffered frames naturally
                    pass
                elif strategy == InterruptStrategy.INSTANT_CUT:
                    # Clear all buffers, discard speech until silence
                    self._video_frames.clear()
                    self._speech_buffer.clear()
                    self._discard_speech_until_silence = True
                    if self._first_silence_frame is not None:
                        self._video_frames.append(self._first_silence_frame)
                    self._turn += 1
                    self._last_frame_idx = 0
                    await self._stop_audio_playback()
                elif strategy == InterruptStrategy.SMOOTH_VIDEO_HARD_AUDIO:
                    # Stop audio immediately, keep video for smooth transition
                    self._speech_buffer.clear()
                    await self._stop_audio_playback()

                await self._client.send_message(OjinCancelInteractionMessage())
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _send_tts_audio(self, frame: TTSAudioRawFrame):
        """Send TTS audio to the server immediately.

        The server maintains a virtual timeline and will generate silence
        when no client audio is available.
        """
        if self._client is None or not self._initialized:
            logger.warning(f"Discarded TTSAudioRawFrame because client is not ready")
            return

        # Resample audio to target sample rate
        resampled_audio = await self._resampler.resample(
            frame.audio, frame.sample_rate, OJIN_PERSONA_SAMPLE_RATE
        )
        self._speech_buffer.extend(resampled_audio)

        if self._waiting_for_first_tts:
            self._waiting_for_first_tts = False
            await self.start_ttfb_metrics()
        # Send audio to server immediately
        logger.debug(f"Sending TTS audio to server, size: {len(resampled_audio)} bytes")
        await self._client.send_message(
            OjinAudioInputMessage(
                audio_int16_bytes=resampled_audio,
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

            # Start the playback loop
            if self._video_playback_task is None:
                self._video_playback_task = self.create_task(self._video_playback_loop())

            # Notify that we're ready
            self._initialized = True
            initialized_frame = OjinVideoInitializedFrame(session_data=self._session_data)
            await self.push_frame(initialized_frame, direction=FrameDirection.DOWNSTREAM)
            await self.push_frame(initialized_frame, direction=FrameDirection.UPSTREAM)

            await self._client.send_message(
                OjinAudioInputMessage(
                    audio_int16_bytes=b"\x00" * BYTES_PER_FRAME,
                )
            )

        elif isinstance(message, OjinInteractionResponseMessage):
            if not self.fps_tracker.is_running:
                self.fps_tracker.start()
            self.fps_tracker.update(1)

            self.last_frame_time = time.monotonic()
            frame_idx = message.index
            samples = [
                int.from_bytes(message.audio_frame_bytes[i : i + 2], "little", signed=True)
                for i in range(0, len(message.audio_frame_bytes) - 1, 2)
            ]
            volume = 0 if len(samples) == 0 else (sum(s * s for s in samples) / len(samples)) ** 0.5

            if self._settings.frame_debugging_enabled:
                logger.debug(
                    f"Received video frame {message.index} [{volume}], delta: {time.monotonic() - self.last_frame_time} buffer: {len(self._video_frames)}"
                )
                if self.fps_tracker.total_frames % 25 == 0:
                    self.fps_tracker.log()

            # Queue video frame with bundled audio
            video_frame = VideoFrame(
                frame_idx=frame_idx,
                image_bytes=message.video_frame_bytes,
                audio_bytes=message.audio_frame_bytes,
                is_final=message.is_final_response,
                volume=volume,
            )

            if self._interrupting and not video_frame.is_silence():
                logger.debug("Interrupting, dropping non-silence frame")
                return

            if self._interrupting and video_frame.is_silence():
                logger.debug("First silence received after end of interruption")
                self._interrupting = False

            if self._first_silence_frame is None and video_frame.is_silence():
                self._first_silence_frame = video_frame

            if self._last_frame_idx == 0 and not video_frame.is_silence():
                excess = max(0, len(self._video_frames) - MIN_FRAMES_BUFFER)
                if excess > 0:
                    for _ in range(excess):
                        self._video_frames.popleft()
                    logger.warning(
                        f"First speech frame: dropped {excess} frames, buffer now {len(self._video_frames)}"
                    )
                video_frame.is_first_speech_frame = True

            if frame_idx != self._last_frame_idx:
                self._turn += 1
                self._last_frame_idx = frame_idx

            self._video_frames.append(video_frame)

        elif isinstance(message, ErrorResponseMessage):
            await self.push_error(
                error_msg=f"Ojin server error: {message.payload.code}", fatal=True
            )
            await self._stop()

    async def _receive_ojin_messages(self):
        """Continuously receive and process messages from the server."""
        while True:
            client = self._client
            if client is None:
                break
            try:
                message = await client.receive_message()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Ojin receive loop exiting: {e}")
                break
            if self._client is None or message is None:
                break
            await self._handle_ojin_message(message)

    async def _video_playback_loop(self):
        """Main playback loop — state machine with audio-as-clock sync.

        State machine:
          IDLE ──(first speech frame + audio ready)──> SPEAKING
          SPEAKING ──(silence frame played)──────────> IDLE
          SPEAKING ──(audio exhausted + silence pending)> IDLE
          SPEAKING ──(interrupt)─────────────────────> depends on strategy

        Each tick (40ms):
          1. Detect silence→speech pre-transition
          2. Release one audio chunk (if speaking and audio available)
          3. Audio-exhaustion guard (deadlock prevention)
          4. Consume / sync one video frame
        """
        logger.info("Starting playback loop")

        start_ts = time.perf_counter()
        next_frame_time = start_ts + self._frame_duration
        frame_count = 0
        initial_buffer = 6

        # Audio-video sync counters (reset each speech turn)
        audio_frames_released = 0
        video_frames_sent = 0

        # Audio playback constants
        sample_rate = OJIN_PERSONA_SAMPLE_RATE
        num_channels = 1
        chunk_size = int(sample_rate * self._frame_duration) * num_channels * 2
        is_first_audio_frame = True
        audio_frame = None
        silence_frame = TTSAudioRawFrame(
            audio=b"\x00" * chunk_size,
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        num_silence_frames_played = 0
        output_vide_frame = None
        while self._initialized:
            # Sleep for most of the wait time
            now = time.perf_counter()
            sleep_time = next_frame_time - now - 0.003
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # Spin lock for precise timing
            while time.perf_counter() < next_frame_time:
                pass

            next_frame_time += self._frame_duration

            # Wait for initial buffer to fill
            if len(self._video_frames) > 0 and initial_buffer > 0:
                initial_buffer -= 1
                continue

            num_next_silence_frames = self.get_num_next_silence_frames()
            should_play_speech_video_frame = (
                num_next_silence_frames == 0 and len(self._video_frames) > 0
            )
            should_start_playing_audio = (
                should_play_speech_video_frame
                and not self._interrupting
                and not self._is_playing_speech_audio
                and self._speech_buffer
            )
            should_stop_playing_audio = self._is_playing_speech_audio and (
                num_silence_frames_played > 5 or not self._speech_buffer
            )
            # ── Step 1: Silence→speech pre-check ──
            # Requires BOTH first_speech_frame AND audio available.
            if should_start_playing_audio:
                logger.debug("Started playing audio")
                self._is_playing_speech_audio = True
                audio_frames_released = 0
                video_frames_sent = 0
                is_first_audio_frame = True
                self._discard_speech_until_silence = False
                await self._start_audio_playback()
            elif should_stop_playing_audio:
                logger.debug(
                    f"Stopped playing audio num_next_silence_frames: {num_next_silence_frames} speech_buffer_empty: {not self._speech_buffer}"
                )
                self._is_playing_speech_audio = False
                await self._stop_audio_playback()

            # ── Step 2: Prepare audio ──
            if self._is_playing_speech_audio and self._speech_buffer:
                if len(self._speech_buffer) < chunk_size:
                    audio = bytes(self._speech_buffer)
                    self._speech_buffer.clear()
                else:
                    audio = bytes(self._speech_buffer[:chunk_size])
                    del self._speech_buffer[:chunk_size]

                audio_frame = OutputAudioRawFrame(
                    audio=audio,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                )
                audio_frame.pts = int(time.monotonic() * 1_000_000_000)
                if is_first_audio_frame:
                    logger.warning(f"First audio frame played! size: {len(audio_frame.audio)}")
                    is_first_audio_frame = False
                audio_frames_released += 1
            else:
                audio_frame = None

            # ── Step 4: Consume video frame ──
            video_frame: Optional[VideoFrame] = None

            if should_play_speech_video_frame:
                video_frame, skipped_speech = await self._consume_speech_frame(
                    audio_frames_released, video_frames_sent
                )
                video_frames_sent += skipped_speech
                if video_frame is not None:
                    video_frames_sent += 1
                else:
                    logger.debug(f"frame miss, repeating frame {frame_count}")
                    video_frame = VideoFrame(
                        frame_idx=frame_count,
                        image_bytes=self._last_played_image_bytes,
                        audio_bytes=b"",
                        is_final=False,
                        volume=1,
                        is_first_speech_frame=False,
                    )
            else:
                video_frame = await self._consume_idle_frame(num_next_silence_frames)

            if video_frame is not None:
                if video_frame.is_silence():
                    num_silence_frames_played += 1
                else:
                    num_silence_frames_played = 0

                frame_count += 1
                self._last_played_image_bytes = video_frame.image_bytes
                if self._settings.frame_debugging_enabled:
                    mode = "SPEECH" if self._is_playing_speech_audio else "SILENCE"
                    logger.debug(
                        f"[{mode}] Playing frame {frame_count}, "
                        f"audio_released: {audio_frames_released}, "
                        f"video_sent: {video_frames_sent}, "
                        f"buffer: {len(self._video_frames)}"
                    )
                output_vide_frame = await self.prepare_video_frame(
                    video_frame.image_bytes, video_frame.is_first_speech_frame
                )
                if output_vide_frame is None:
                    logger.warning(
                        f"Dropping undecodable video frame "
                        f"(frame_idx={video_frame.frame_idx}, "
                        f"is_silence={video_frame.is_silence()}, "
                        f"image_bytes_len={len(video_frame.image_bytes)})"
                    )
                else:
                    await self.push_frame(output_vide_frame)

            if audio_frame is not None:
                await self.push_frame(audio_frame)
            else:
                await self.push_frame(silence_frame)

    async def _consume_speech_frame(
        self, audio_frames_released: int, video_frames_sent: int
    ) -> tuple[Optional[VideoFrame], int]:
        frame = self._video_frames.popleft()
        return frame, 0

    async def _consume_idle_frame(self, num_next_silence_frames: int) -> Optional[VideoFrame]:
        """Consume one video frame in idle (silence) mode.

        Plays frames at normal 25fps rate. Handles instant-cut discard logic.
        Does not consume first_speech_frame — those wait for audio.
        """
        if not self._video_frames:
            return None

        frame = self._video_frames.popleft()
        num_next_silence_frames -= 1
        # Buffer management for silence frames
        if num_next_silence_frames > MAX_FRAMES_BUFFER:
            self._video_frames.popleft()
            num_next_silence_frames -= 1
            if self._settings.frame_debugging_enabled:
                logger.debug(
                    f"Skipping silence frame num_next_silence_frames: {num_next_silence_frames}, "
                    f"target: {MAX_FRAMES_BUFFER}"
                )

        if self.is_pending_speech_waiting_in_buffer() and num_next_silence_frames:
            self._video_frames.popleft()
            num_next_silence_frames -= 1
            if self._settings.frame_debugging_enabled:
                logger.debug(
                    f"Skipping additional silence frame num_next_silence_frames: {num_next_silence_frames}"
                )
        return frame

    def is_pending_speech_waiting_in_buffer(self) -> bool:
        """Check if there are any pending speech frames in the buffer."""
        return any(frame.is_first_speech_frame for frame in self._video_frames)

    def get_num_next_silence_frames(self) -> int:
        """Count consecutive silence frames from the front of the buffer."""
        count = 0
        for frame in self._video_frames:
            if frame.is_silence() or (frame.volume == 0 and not self._is_playing_speech_audio):
                count += 1
            else:
                break
        return count

    async def _start_audio_playback(self):
        logger.warning("Starting audio playback")
        self._is_playing_speech_audio = True
        await self.push_frame(OjinBotStartedSpeakingFrame(), direction=FrameDirection.DOWNSTREAM)
        await self.stop_ttfb_metrics()

    async def _stop_audio_playback(self):
        if not self._is_playing_speech_audio:
            return

        self._is_playing_speech_audio = False
        await self.push_frame(OjinBotStoppedSpeakingFrame(), direction=FrameDirection.DOWNSTREAM)
        self._speech_buffer.clear()

    async def _start(self):
        """Initialize the service and start processing."""
        is_connected = await self.connect_with_retry()
        if not is_connected:
            return

        # Start receiving messages
        self._receive_msg_task = self.create_task(self._receive_ojin_messages())

        # Start an interaction for continuous streaming
        await self._client.start_interaction()

    async def prepare_video_frame(
        self, video: bytes, is_first: bool = False
    ) -> OutputImageRawFrame:
        image_array = np.frombuffer(video, dtype=np.uint8)
        image_size = self._settings.image_size
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if decoded_image is not None:
            h, w = decoded_image.shape[:2]
            target_w, target_h = image_size

            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            scaled_image = cv2.resize(decoded_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
            rgb_bytes = rgb_image.tobytes()

            rgb_frame = OutputImageRawFrame(image=rgb_bytes, size=(new_w, new_h), format="RGB")
            rgb_frame.pts = int(time.monotonic() * 1_000_000_000)
            if is_first:
                logger.warning(f"First image frame played!")
            return rgb_frame
        return None

    async def _stop(self):
        self._initialized = False
        self._is_playing_speech_audio = False
        if self._client:
            await self._client.close()
            self._client = None

        """Stop the service and clean up resources."""
        if self._receive_msg_task:
            await self.cancel_task(self._receive_msg_task)
            self._receive_msg_task = None

        if self._video_playback_task:
            await self.cancel_task(self._video_playback_task)
            self._video_playback_task = None

        logger.debug(f"OjinVideoService {self._settings.config_id} stopped")
