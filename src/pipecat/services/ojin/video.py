import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from loguru import logger
from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_stv_client import OjinSTVClient
from ojin.ojin_stv_messages import (
    IOjinSTVClient,
    OjinSTVCancelInteractionMessage,
    OjinSTVInteractionInputMessage,
    OjinSTVInteractionResponseMessage,
    OjinSTVSessionReadyMessage,
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
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class OjinVideoServiceInitializedFrame(Frame):
    """Frame indicating that the persona has been initialized and can now output frames."""

    session_data: Optional[dict] = None


class OjinFirstFramePlayedFrame(Frame):
    """Frame indicating that the first frame has been played."""

    pass


class OjinLastFramePlayedFrame(Frame):
    """Frame indicating that the last frame has been played."""

    pass


OJIN_PERSONA_SAMPLE_RATE = 16000
SPEECH_FILTER_AMOUNT = 1000.0
IDLE_FILTER_AMOUNT = 1000.0
IDLE_MOUTH_OPENING_SCALE = 0.0
SPEECH_MOUTH_OPENING_SCALE = 1.0


class OjinVideoServiceState(Enum):
    """Simplified persona states."""

    INITIALIZING = "initializing"  # Caching idle frames
    IDLE = "idle"  # Playing idle animation
    SPEAKING = "speaking"  # Playing speech frames
    INTERRUPTING = "interrupting"  # Interrupting speech


@dataclass
class VideoFrame:
    """Represents a video frame with bundled audio from the server.

    The server sends video frames with their corresponding audio already matched,
    simplifying client-side synchronization.
    """

    frame_idx: int
    image_bytes: bytes
    audio_bytes: bytes
    is_final: bool


@dataclass
class OjinVideoServiceSettings:
    """Settings for Ojin Video Service service."""

    api_key: str = field(default="")
    ws_url: str = field(default="wss://models.ojin.ai/realtime")
    client_connect_max_retries: int = field(default=3)
    client_reconnect_delay: float = field(default=3.0)
    config_id: str = field(default="")
    image_size: Tuple[int, int] = field(default=(1280, 720))
    tts_audio_passthrough: bool = field(default=False)
    extra_frames_lat: int = field(default=15)


@dataclass
class IdleFrame:
    """Represents a single idle animation frame."""

    frame_idx: int
    image_bytes: bytes


class OjinVideoService(FrameProcessor):
    """Simplified Ojin Video Service integration for Pipecat.

    This service abstracts interaction management - the server handles all complexity
    of matching audio to video frames. The client simply:
    1. Sends audio input immediately when received
    2. Receives video frames with bundled audio
    3. Plays frames at the correct fps
    4. Handles interruptions by clearing buffers
    """

    def __init__(
        self,
        settings: OjinVideoServiceSettings,
        client: IOjinSTVClient | None = None,
    ) -> None:
        super().__init__()
        logger.debug(f"OjinVideoService initialized with settings {settings}")

        self._settings = settings
        if client is None:
            self._client = OjinSTVClient(
                ws_url=settings.ws_url,
                api_key=settings.api_key,
                config_id=settings.config_id,
                mode=os.getenv("OJIN_MODE", ""),
            )
        else:
            self._client = client

        self._state = OjinVideoServiceState.INITIALIZING

        # Idle animation frames (cached during initialization)
        self._idle_frames: list[IdleFrame] = []
        self._is_mirrored_loop: bool = True

        # Speech frames queue (video + audio bundled)
        self._speech_frames: deque[VideoFrame] = deque()

        # Playback state
        self.fps = 25
        self._num_speech_frames_played = 0
        self._current_frame_idx = -1
        self._played_frame_idx = -1
        self._last_queued_frame_idx = -1

        # Audio input handling
        self._resampler = create_default_resampler()

        # Tasks
        self._receive_msg_task: Optional[asyncio.Task] = None
        self._send_audio_task: Optional[asyncio.Task] = None
        self._run_loop_task: Optional[asyncio.Task] = None

        # Server tracking
        self._server_fps_tracker = FPSTracker("OjinVideoService")

        # Debugging: Track time from TTSStartedFrame to first video frame
        self._tts_started_timestamp: Optional[float] = None
        self._first_frame_received_timestamp: Optional[float] = None
        self._time_to_first_frame_measurements: list[float] = []

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
            f"Failed to connect after {self._settings.client_connect_max_retries} attempts. Last error: {last_error}"
        )
        await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            logger.debug("StartFrame")
            await self.push_frame(frame, direction)
            await self._start()
        elif isinstance(frame, TTSStartedFrame):
            logger.debug("StartInterruptionFrame")
            # Clear speech frames buffer
            self._speech_frames.clear()
            self._first_frame_received_timestamp = None
            self._first_frame_played_timestamp = None

            # Track timestamp for debugging time to first video frame
            self._tts_started_timestamp = time.perf_counter()

            await self.set_state(OjinVideoServiceState.IDLE)
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            if self._state == OjinVideoServiceState.INTERRUPTING:
                logger.debug("TTSAudioRawFrame while interrupting speech - discarded")
                return

            logger.debug("TTSAudioRawFrame - sending audio immediately")
            # Resample and buffer audio - will be sent immediately by send_audio_task
            resampled_audio = await self._resampler.resample(
                frame.audio, frame.sample_rate, OJIN_PERSONA_SAMPLE_RATE
            )
            await self._client.send_message(
                OjinSTVInteractionInputMessage(
                    audio_int16_bytes=resampled_audio,
                    params={
                        "client_frame_index": self._compute_frame_index_for_server(),
                        "filter_amount": SPEECH_FILTER_AMOUNT,
                        "mouth_opening_scale": SPEECH_MOUTH_OPENING_SCALE,
                    },
                )
            )
            await self.set_state(OjinVideoServiceState.SPEAKING)
            if self._settings.tts_audio_passthrough:
                await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)

        elif isinstance(frame, StartInterruptionFrame):
            logger.debug("StartInterruptionFrame")
            if self._state != OjinVideoServiceState.INITIALIZING:
                await self._interrupt()

            await self.push_frame(frame, direction)

        else:
            # Pass through any other frames
            await self.push_frame(frame, direction)

    async def _handle_ojin_message(self, message: BaseModel):
        """Process incoming messages from the server."""
        if isinstance(message, OjinSTVSessionReadyMessage):
            if message.parameters is not None:
                self._is_mirrored_loop = message.parameters.get("is_mirrored_loop", True)
                self._session_data = message.parameters

            logger.info(f"Received Session Ready session data: {message}")
            if self._session_data and self._session_data.get("server_id"):
                logger.info(f"Connected to server: {self._session_data.get('server_id')}")

            self._server_fps_tracker.start()

            logger.info("Requesting idle frames")
            # Request idle frames from server
            await self._client.start_interaction()
            await self._client.send_message(
                OjinSTVInteractionInputMessage(
                    audio_int16_bytes=bytes(),
                    params={
                        "filter_amount": IDLE_FILTER_AMOUNT,
                        "mouth_opening_scale": IDLE_MOUTH_OPENING_SCALE,
                        "generate_idle_frames": True,
                    },
                )
            )

        elif isinstance(message, OjinSTVInteractionResponseMessage):
            frame_idx = message.index

            if self._state == OjinVideoServiceState.INITIALIZING:
                # Caching idle frames
                idle_frame = IdleFrame(
                    frame_idx=frame_idx,
                    image_bytes=message.video_frame_bytes,
                )
                self._idle_frames.append(idle_frame)

                if message.is_final_response:
                    logger.info(f"Cached {len(self._idle_frames)} idle frames")
                    await self.set_state(OjinVideoServiceState.IDLE)
                    self._run_loop_task = self.create_task(self._run_loop())

                    # Notify that we're ready
                    initialized_frame = OjinVideoServiceInitializedFrame(session_data=self._session_data)
                    await self.push_frame(
                        initialized_frame,
                        direction=FrameDirection.DOWNSTREAM,
                    )
                    await self.push_frame(
                        initialized_frame, direction=FrameDirection.UPSTREAM
                    )
            else:
                # Avoid getting frames that are not suposed to be part of the speak (remainings of old speech)
                if self._state == OjinVideoServiceState.IDLE:
                    return

                if self._first_frame_received_timestamp is None:
                    self._first_frame_received_timestamp = time.perf_counter()
                    self._time_to_first_frame_measurements.append(
                        self._first_frame_received_timestamp - self._tts_started_timestamp
                    )
                    logger.info(
                        f"Time to first video frame received for interaction id: {message.interaction_id}: {self._time_to_first_frame_measurements[-1] * 1000:.2f}ms"
                    )

                # Speech frame with bundled audio
                # NOTE: Server sends audio_bytes along with video frame
                video_frame = VideoFrame(
                    frame_idx=frame_idx,
                    image_bytes=message.video_frame_bytes,
                    audio_bytes=message.audio_frame_bytes,
                    is_final=message.is_final_response,
                )

                self._speech_frames.append(video_frame)
                self._last_queued_frame_idx = frame_idx
                logger.debug(
                    f"Received video frame {frame_idx}, is_final={message.is_final_response}"
                )

                # Transition to IDLE when interrupting and final response is received
                # if self._state == OjinVideoServiceState.INTERRUPTING and message.is_final_response:
                #     # Clear speech frames buffer
                #     self._speech_frames.clear()

                #     self.set_state(OjinVideoServiceState.IDLE)

        elif isinstance(message, ErrorResponseMessage):
            is_fatal = False
            if message.payload.code == "NO_BACKEND_SERVER_AVAILABLE":
                logger.error("No OJIN servers available. Please try again later.")
                is_fatal = True
            elif message.payload.code == "FRAME_SIZE_TOO_BIG":
                logger.error(
                    "Frame Size sent to Ojin server was higher than the allowed max limit."
                )
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

    async def set_state(self, state: OjinVideoServiceState):
        """Update the persona state."""
        if self._state == state:
            return

        old_state = self._state
        logger.debug(f"OjinVideoServiceState changed from {old_state} to {state}")
        match state:
            case OjinVideoServiceState.IDLE:
                # Push last frame played if we're coming from SPEAKING state
                if old_state == OjinVideoServiceState.SPEAKING:
                    await self.push_frame(OjinLastFramePlayedFrame(), FrameDirection.UPSTREAM)
                    await self.push_frame(OjinLastFramePlayedFrame(), FrameDirection.DOWNSTREAM)
                self._num_speech_frames_played = 0
        self._state = state

    async def _receive_ojin_messages(self):
        """Continuously receive and process messages from the server."""
        while True:
            assert self._client is not None
            message = await self._client.receive_message()
            if message is not None:
                await self._handle_ojin_message(message)
            await asyncio.sleep(0.001)

    def _compute_frame_index_for_server(self) -> int:
        """Compute the frame index to send to server for proper synchronization."""
        return self._played_frame_idx + self._settings.extra_frames_lat

    def _get_idle_frame_for_index(self, index: int) -> IdleFrame:
        """Get idle frame for the given index with mirroring support."""
        mirror_idx = self._mirror_index(
            index, len(self._idle_frames), 2 if self._is_mirrored_loop else 1
        )
        return self._idle_frames[mirror_idx]

    def _mirror_index(self, index: int, size: int, period: int = 2) -> int:
        """Calculate a mirrored index for ping-pong animation effect."""
        turn = index // size
        res = index % size
        if turn % period == 0:
            return res
        else:
            return size - res - 1

    async def _run_loop(self):
        """Main playback loop - runs continuously after initialization.

        Handles both idle and speech playback seamlessly.
        """
        logger.info("Starting playback loop")

        # Wait for initialization to complete
        while self._state == OjinVideoServiceState.INITIALIZING:
            await asyncio.sleep(0.1)

        silence_duration = 1 / self.fps
        audio_bytes_length_for_one_frame = 2 * int(silence_duration * OJIN_PERSONA_SAMPLE_RATE)
        silence_audio_for_one_frame = b"\x00" * audio_bytes_length_for_one_frame

        start_ts = time.perf_counter()
        self._played_frame_idx = -1

        while True:
            elapsed_time = time.perf_counter() - start_ts
            next_frame_idx = int(elapsed_time * self.fps)

            # Wait for next frame time
            if next_frame_idx <= self._current_frame_idx:
                next_frame_time = (self._current_frame_idx + 1) / self.fps
                waiting_time = next_frame_time - elapsed_time - 0.005
                await asyncio.sleep(max(0, waiting_time))

                # Spin lock for precise timing
                elapsed_time = time.perf_counter() - start_ts
                next_frame_idx = self._current_frame_idx + 1
                calculated_frame_idx = int(elapsed_time * self.fps)
                while calculated_frame_idx < next_frame_idx:
                    elapsed_time = time.perf_counter() - start_ts
                    calculated_frame_idx = int(elapsed_time * self.fps)

            self._current_frame_idx = next_frame_idx

            # Determine which frame to play
            image_bytes = None
            audio_bytes = silence_audio_for_one_frame

            # Check if we have a ready speech frame
            if (
                len(self._speech_frames) > 0
                and self._speech_frames[0].frame_idx <= self._current_frame_idx
            ):
                # Play speech frame
                video_frame = self._speech_frames.popleft()
                image_bytes = video_frame.image_bytes
                audio_bytes = (
                    video_frame.audio_bytes
                    if video_frame.audio_bytes
                    else silence_audio_for_one_frame
                )
                self._played_frame_idx = video_frame.frame_idx

                logger.debug(f"Playing speech frame {video_frame.frame_idx}")
                self._num_speech_frames_played += 1

                if self._first_frame_played_timestamp is None:
                    self._first_frame_played_timestamp = time.perf_counter()
                    logger.info(
                        f"Time to first video frame played: {self._first_frame_played_timestamp - self._tts_started_timestamp}s"
                    )
                    await self.push_frame(OjinFirstFramePlayedFrame(), FrameDirection.UPSTREAM)
                    await self.push_frame(OjinFirstFramePlayedFrame(), FrameDirection.DOWNSTREAM)

                # Check if this was the last speech frame
                if video_frame.is_final and len(self._speech_frames) == 0:
                    logger.info("Last speech frame played, transitioning to IDLE")
                    await self.set_state(OjinVideoServiceState.IDLE)

            else:
                if self._num_speech_frames_played > 0:
                    logger.debug(f"frame missed: {self._current_frame_idx}")
                    self._current_frame_idx -= 1
                    await asyncio.sleep(0.005)
                    continue

                # Play idle frame
                self._played_frame_idx += 1
                idle_frame = self._get_idle_frame_for_index(self._played_frame_idx)
                image_bytes = idle_frame.image_bytes
                # audio_bytes is already set to silence

                if self._played_frame_idx % 150 == 0:
                    logger.debug(f"Playing idle frame (%150) {self._played_frame_idx}")
                # if self._state == OjinVideoServiceState.IDLE:
                #     if self._played_frame_idx % 25 == 0:
                #         logger.debug(f"Playing idle frame (%25) {self._played_frame_idx}")
                # else:
                #     logger.debug(f"Playing idle frame {self._played_frame_idx}")

            # Output frame and audio
            image_frame = OutputImageRawFrame(
                image=image_bytes, size=self._settings.image_size, format="RGB"
            )
            audio_frame = OutputAudioRawFrame(
                audio=audio_bytes,
                sample_rate=OJIN_PERSONA_SAMPLE_RATE,
                num_channels=1,
            )
            await self.push_frame(image_frame)
            await self.push_frame(audio_frame)

    async def _start(self):
        """Initialize the persona service and start processing."""
        is_connected = await self.connect_with_retry()
        if not is_connected:
            return

        self._receive_msg_task = self.create_task(self._receive_ojin_messages())

    async def _stop(self):
        """Stop the persona service and clean up resources."""
        if self._send_audio_task:
            await self.cancel_task(self._send_audio_task)
            self._send_audio_task = None

        if self._receive_msg_task:
            await self.cancel_task(self._receive_msg_task)
            self._receive_msg_task = None

        if self._run_loop_task:
            await self.cancel_task(self._run_loop_task)
            self._run_loop_task = None

        if self._client:
            await self._client.close()
            self._client = None

        logger.debug(f"OjinVideoService {self._settings.config_id} stopped")

    async def _interrupt(self):
        """Interrupt current speech.

        Server will flag next frame as final and drop remaining frames.
        Client just needs to clear buffers.
        """
        logger.info("Interrupting speech")

        # Send cancel to server
        if self._client is not None:
            await self._client.send_message(OjinSTVCancelInteractionMessage())

        await self.set_state(OjinVideoServiceState.INTERRUPTING)
