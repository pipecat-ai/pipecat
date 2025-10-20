import asyncio
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

# Will use numpy when implementing persona-specific processing
from loguru import logger
from ojin.entities.interaction_messages import ErrorResponseMessage
from ojin.ojin_persona_client import OjinPersonaClient
from ojin.ojin_persona_messages import (
    IOjinPersonaClient,
    OjinPersonaCancelInteractionMessage,
    OjinPersonaEndInteractionMessage,
    OjinPersonaInteractionInputMessage,
    OjinPersonaInteractionResponseMessage,
    OjinPersonaSessionReadyMessage,
)
from ojin.profiling_utils import FPSTracker
from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class OjinPersonaInitializedFrame(Frame):
    """Frame indicating that the persona has been initialized and can now output frames."""

    pass


OJIN_PERSONA_SAMPLE_RATE = 16000
SPEECH_FILTER_AMOUNT = 1000.0
IDLE_FILTER_AMOUNT = 1000.0
IDLE_MOUTH_OPENING_SCALE = 0.0
SPEECH_MOUTH_OPENING_SCALE = 1.0

IDLE_ANIMATION_KEYFRAMES_SLOT = 0


async def async_accurate_sleep(duration_s: float):
    if duration_s < 0:
        return
    millisecond = 0.001
    epsilon = 0.5 * millisecond
    to_sleep = max(0, duration_s - epsilon)
    start = time.perf_counter()
    await asyncio.sleep(to_sleep)
    observed = time.perf_counter() - start
    to_sleep -= observed

    start = time.perf_counter()
    while to_sleep > time.perf_counter() - start:
        continue


# NOTE(mouad): source: https://blat-blatnik.github.io/computerBear/making-accurate-sleep-function/
def accurate_sleep(duration_s: float):
    millisecond = 0.001
    to_sleep = duration_s
    estimate = 5 * millisecond
    mean = 5 * millisecond
    count = 1
    m2 = 0.0

    while to_sleep > estimate:
        start = time.perf_counter()
        time.sleep(1 * millisecond)
        observed = time.perf_counter() - start
        to_sleep -= observed

        count += 1

        delta = observed - mean
        mean += delta / count
        m2 += delta * (observed - mean)
        stddev = math.sqrt(m2 / (count - 1))
        estimate = mean + stddev

    # NOTE(mouad): spin-lock
    start = time.perf_counter()
    while to_sleep > time.perf_counter() - start:
        continue


class PersonaState(Enum):
    """Enum representing the possible states of the persona conversation FSM."""

    INVALID = "invalid"
    INITIALIZING = "initializing"
    ACTIVE = "active"


class InteractionState(Enum):
    """Enum representing the possible states of an persona interaction.

    These states track the lifecycle of an interaction from creation to completion.
    """

    INACTIVE = "inactive"
    ACTIVE = "active"
    ALL_AUDIO_PROCESSED = "all_audio_processed"


@dataclass
class OjinPersonaInteraction:
    """Represents an interaction session between a user and the Ojin persona.

    This class maintains the state of an ongoing interaction, including audio queues,
    frame tracking for animations, and interaction lifecycle state. It handles the
    buffering of audio inputs and manages the interaction's state transitions.
    """

    interaction_id: str = ""

    state: InteractionState = InteractionState.INACTIVE
    received_frames: int = 0
    expected_frames: float = 0
    has_received_stop_frame: bool = False

    def __post_init__(self):
        self.pending_audio = bytearray()

    def next_frame(self):
        """Advance to the next frame in the animation sequence.

        Updates the frame index and applies mirroring for smooth looping animations.
        """
        self.frame_idx += 1

    def close(self):
        """Close the interaction."""
        self.pending_audio.clear()
        self.state = InteractionState.INACTIVE

    def set_state(self, new_state: InteractionState):
        """Update the interaction state.

        Changes the interaction's state and logs the transition if the state
        actually changes.

        Args:
            new_state: The new state to transition to

        """
        if self.state == new_state:
            return

        logger.debug(f"Old Interaction state: {self.state}, New Interaction state: {new_state}")
        self.state = new_state

    def update_expected_frames(self, audio_buffer_len: int, sample_rate: int, fps: int):
        bytes_per_sample = 2
        samples_count = audio_buffer_len // bytes_per_sample
        self.expected_frames += (samples_count / sample_rate) * fps
        logger.debug(
            f"pushing audio samples count: {samples_count}, next expected frames: {self.expected_frames}"
        )


@dataclass
class OjinPersonaSettings:
    """Settings for Ojin Persona service.

    This class encapsulates all configuration parameters for the OjinPersonaService.
    """

    api_key: str = field(default="")  # api key for Ojin platform
    ws_url: str = field(default="wss://models.ojin.ai/realtime")  # websocket url for Ojin platform
    client_connect_max_retries: int = field(
        default=3
    )  # amount of times it will try to reconnect to the server
    client_reconnect_delay: float = field(default=3.0)  # time between connecting retries
    persona_config_id: str = field(default="")  # config id of the persona to use from Ojin platform
    image_size: Tuple[int, int] = field(default=(1920, 1080))
    cache_idle_sequence: bool = field(
        default=False
    )  # whether to cache the idle sequence loop to avoid doing inference while persona is not speaking
    idle_sequence_duration: int = field(default=30)  # length of the idle sequence loop in seconds.
    idle_to_speech_seconds: float = field(
        default=0.75
    )  # seconds to wait before starting speech, recommended not less than 0.75 to avoid missing frames. This ensures smooth transition between idle frames and speech frames
    tts_audio_passthrough: bool = field(
        default=False
    )  # whether to pass through TTS audio to the output
    push_bot_stopped_speaking_frames: bool = field(
        default=True
    )  # whether to push bot stopped speaking frames to the output
    frame_count_threshold_for_end_interaction: int = field(
        default=-1
    )  # If -1 then it will not end the interaction based on frame count only when receiving TTSStoppedFrame. If the number of frames in the loopback is less than or equal to this value then end the interaction to avoid frame misses.

    extra_frames_lat: int = field(
        default=15,
    )  # round trip latency between server and client, make sure to specify extra room for error


@dataclass
class AnimationKeyframe:
    """Represents a single frame in an animation sequence.

    This class stores information about a specific keyframe in an animation,
    including its position in the sequence and the image data.

    Attributes:
        mirror_frame_idx (int): Index used for mirrored animation playback
        frame_idx (int): The sequential index of this frame in the animation
        image (bytes): The binary image data for this keyframe

    """

    frame_idx: int
    image: bytes
    is_final_frame: bool


@dataclass
class Audio_Input:
    frame: OutputAudioRawFrame
    duration: float


class OjinPersonaService(FrameProcessor):
    """Ojin Persona integration for Pipecat.

    This class provides integration between Ojin personas and the Pipecat framework.
    """

    def __init__(
        self,
        settings: OjinPersonaSettings,
        client: IOjinPersonaClient | None = None,
    ) -> None:
        super().__init__()
        logger.debug(f"OjinPersonaService initialized with settings {settings}")
        # Use provided settings or create default settings
        self._settings = settings
        if client is None:
            self._client = OjinPersonaClient(
                ws_url=settings.ws_url,
                api_key=settings.api_key,
                config_id=settings.persona_config_id,
                mode=os.getenv("OJIN_MODE", ""),
            )
        else:
            self._client = client

        assert self._settings.persona_config_id is not None
        self.num_speech_frames_played: int = 0
        self._processed_queued_audio_task: Optional[asyncio.Task] = None
        self._run_loop_task: Optional[asyncio.Task] = None

        self._resampler = create_default_resampler()
        self._server_fps_tracker = FPSTracker("OjinPersonaService")
        self._fsm_fps_tracker = FPSTracker("OjinPersonaService")

        self._receive_msg_task: Optional[asyncio.Task] = None

        self._interaction: Optional[OjinPersonaInteraction] = None
        self._pending_interaction: Optional[OjinPersonaInteraction] = None

        self.idle_frames: list[AnimationKeyframe] = []  # Keyframes of the idle animation
        self.pending_speech_frames: deque[AnimationKeyframe] = deque()

        self.is_mirrored_loop: bool = True
        self.fps = 25
        self.current_frame_index = -1
        self.last_queued_frame_index = -1
        self.last_played_frame_index = -1

        self.pedning_audio_mutex: threading.Lock = threading.Lock()
        self.pending_audio_to_play = bytearray()
        self.extra_frames_lat = settings.extra_frames_lat

        self.audio_queue: queue.Queue[Audio_Input] = queue.Queue()
        self.event_loop = asyncio.get_event_loop()
        self.should_stop_updated_loop = threading.Event()
        self.persona_state = PersonaState.INITIALIZING

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

                # If this was the last attempt, don't wait
                if attempt < self._settings.client_connect_max_retries - 1:
                    logger.info(f"Retrying in {self._settings.client_reconnect_delay} seconds...")
                    await asyncio.sleep(self._settings.client_reconnect_delay)

        # All retry attempts failed
        logger.error(
            f"Failed to connect after {self._settings.client_connect_max_retries} attempts. Last error: {last_error}"
        )
        await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        return False

    async def _handle_ojin_message(self, message: BaseModel):
        """Process incoming messages from the proxy.

        Handles different message types including authentication responses,
        interaction ready notifications, and interaction responses containing
        video frames.

        Args:
            message: The message received from the proxy

        """
        if isinstance(message, OjinPersonaSessionReadyMessage):
            logger.info("Received Session Ready")

            self.persona_state = PersonaState.INITIALIZING

            assert self._client is not None
            # TODO(mouad): persona fps tracker
            self._server_fps_tracker.start()
            if message.parameters is not None:
                self.is_mirrored_loop = message.parameters.get("is_mirrored_loop", True)

            logger.info("Sending silence to init idle frames")
            # Starting an interaction with "generate_idle_frames" will generate all available frames (depending on video source lengh), we don't need to send audio or end the interaction manually
            # This is equivalent to sending silence for the exact amount of duration of the source video
            interaction_id = await self._client.start_interaction(
                params={
                    "filter_amount": IDLE_FILTER_AMOUNT,
                    "mouth_opening_scale": IDLE_MOUTH_OPENING_SCALE,
                    "source_keyframes_index": IDLE_ANIMATION_KEYFRAMES_SLOT,
                    "destination_keyframes_index": IDLE_ANIMATION_KEYFRAMES_SLOT,
                    "generate_idle_frames": True,
                }
            )
            await self._client.send_message(
                OjinPersonaEndInteractionMessage(
                    interaction_id=interaction_id,
                )
            )

        elif isinstance(message, OjinPersonaInteractionResponseMessage):
            frame_idx = message.index
            animation_frame = AnimationKeyframe(
                frame_idx=frame_idx,
                image=message.video_frame_bytes,
                is_final_frame=message.is_final_response,
            )
            if self.persona_state == PersonaState.INITIALIZING:
                # IDLE frame
                self.idle_frames.append(animation_frame)
            else:
                if not self.can_receive_video_frames():
                    logger.warning("Received video frame with no active speech interaction")
                    return
                assert self._interaction is not None

                logger.debug(f"Received video frame {frame_idx}")
                self._interaction.received_frames += 1
                self.pending_speech_frames.append(animation_frame)
                self.last_queued_frame_index = animation_frame.frame_idx

            if animation_frame.is_final_frame:
                self._interaction = None

                if self.persona_state == PersonaState.INITIALIZING:
                    self._run_loop_task = self.create_task(self._run_loop())
                    # threading.Thread(target=self.update_loop_worker).start()
                    self.persona_state = PersonaState.ACTIVE
                    await self.push_frame(
                        OjinPersonaInitializedFrame(), direction=FrameDirection.DOWNSTREAM
                    )
                    await self.push_frame(
                        OjinPersonaInitializedFrame(), direction=FrameDirection.UPSTREAM
                    )

                if self._pending_interaction is not None:
                    await self._start_speech_interaction(self._pending_interaction)
                    self._pending_interaction = None

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

    def can_receive_video_frames(self):
        return (
            self._interaction is not None and self._interaction.state != InteractionState.INACTIVE
        )

    async def _receive_ojin_messages(self):
        """Continuously receive and process messages from the proxy.

        This method runs as a background task and handles all incoming messages
        from the proxy.
        """
        while True:
            assert self._client is not None
            message = await self._client.receive_message()
            if message is not None:
                await self._handle_ojin_message(message)
            await asyncio.sleep(0.001)

    async def _start(self):
        """Initialize the persona service and start processing.

        Authenticates with the proxy and creates tasks for processing
        audio and receiving messages.
        """
        is_connected = await self.connect_with_retry()

        if not is_connected:
            return

        self._receive_msg_task = self.create_task(self._receive_ojin_messages())
        self._processed_queued_audio_task = self.create_task(self._process_queued_audio())

    async def _stop(self):
        """Stop the persona service and clean up resources.

        Cancels all running tasks, closes connections, and resets the state.
        """
        # Cancel queued audio processing task
        if self._processed_queued_audio_task:
            await self.cancel_task(self._processed_queued_audio_task)
            self._processed_queued_audio_task = None

        if self._receive_msg_task:
            await self.cancel_task(self._receive_msg_task)
            self._receive_msg_task = None

        if self._client:
            await self._client.close()
            self._client = None

        # Clear all buffers
        await self._interrupt()

        if self._run_loop_task:
            await self.cancel_task(self._run_loop_task)

        self.should_stop_updated_loop.set()
        logger.debug(f"OjinPersonaService {self._settings.persona_config_id} stopped")

    async def _interrupt(self):
        """Interrupt the current interaction.

        Sends a cancel message to the backend, updates the FSM state, and
        cleans up the current interaction.
        """
        if self._pending_interaction is not None:
            self._pending_interaction.close()
            self._pending_interaction = None

        if self._interaction is None or self._interaction.interaction_id is None:
            logger.debug("Trying to interrupt an interaction but none is active")
            return

        logger.debug(f"Try interrupt interaction in state {self._interaction.state}")
        if self._interaction.state != InteractionState.INACTIVE:
            logger.warning("Sending CancelInteractionMessage")
            assert self._client is not None
            await self._client.send_message(
                OjinPersonaCancelInteractionMessage(
                    interaction_id=self._interaction.interaction_id,
                )
            )
            # TODO(mouad): interpolate towards silence instead of hard stop?
            with self.pedning_audio_mutex:
                self.pending_audio_to_play.clear()

            self.pending_speech_frames.clear()
            self.num_speech_frames_played = 0
            self.last_queued_frame_index = self.current_frame_index

            self._interaction.close()
            self._interaction = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline.

        This method handles different frame types and routes them to the appropriate
        handler methods. It manages service lifecycle events (Start/End/Cancel),
        audio processing, and interaction state transitions.

        Args:
            frame: The frame to process
            direction: The direction of the frame (input or output)

        """
        await super().process_frame(frame, direction)

        # logger.debug(f"Processing frame: {type(frame)}")
        if isinstance(frame, StartFrame):
            logger.debug("StartFrame")
            await self.push_frame(frame, direction)
            await self._start()

        elif isinstance(frame, TTSStartedFrame):
            logger.debug("TTSStartedFrame")
            await self._start_speech_interaction()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSStoppedFrame):
            logger.debug("TTSStoppedFrame")
            await self._end_interaction()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            logger.debug("TTSAudioRawFrame")
            # TODO(@JM): Check if speech interaction is already possible
            await self._handle_input_audio(frame)
            if self._settings.tts_audio_passthrough:
                await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)

        elif isinstance(frame, StartInterruptionFrame):
            logger.debug("StartInterruptionFrame")
            # only interrupt if we are allowed to send TTS input
            if not self.is_pending_initialization():
                await self._interrupt()

            await self.push_frame(frame, direction)

        else:
            # Pass through any other frames
            await self.push_frame(frame, direction)

    async def _handle_input_audio(self, frame: TTSAudioRawFrame):
        """Process incoming audio frames from the TTS service.

        Handles audio frames based on the current persona state. If the persona is not
        ready to receive input, the audio is queued for later processing. Otherwise,
        it starts or continues an active interaction.

        Resamples the audio to the target sample rate and either sends it directly
        to the backend if an interaction is running, or queues it for later processing.

        Args:
            frame: The audio frame to process

        """
        resampled_audio = await self._resampler.resample(
            frame.audio, frame.sample_rate, OJIN_PERSONA_SAMPLE_RATE
        )

        interaction_to_use: OjinPersonaInteraction = (
            self._pending_interaction
            if self._pending_interaction is not None
            else self._interaction
        )

        if interaction_to_use is None:
            logger.error("Trying to process audio input when no interaction is available")
            return

        interaction_to_use.pending_audio.extend(resampled_audio)

        interaction_to_use.update_expected_frames(
            len(resampled_audio), OJIN_PERSONA_SAMPLE_RATE, self.fps
        )

    def is_pending_initialization(self) -> bool:
        """Check if the persona is ready to receive TTS input.

        Returns:
            True if the persona is in a state that can accept TTS input, False otherwise

        """
        return self.persona_state in [
            PersonaState.INITIALIZING,
            PersonaState.INVALID,
        ]

    async def _start_speech_interaction(
        self,
        new_interaction: Optional[OjinPersonaInteraction] = None,
        active_keyframes_slot: int = IDLE_ANIMATION_KEYFRAMES_SLOT,
        keyframe_slot_to_update: int = -1,
    ):
        """Start a new interaction with the persona.

        Creates a new interaction or uses the provided one, initializes it with
        the appropriate state and parameters, and sends a start message to the backend.

        Args:
            new_interaction: Optional existing interaction to use instead of creating a new one
            is_speech: Whether this interaction is speech-based (True) or idle (False)

        """
        if self.is_pending_initialization() or (
            self._interaction is not None
            and self._interaction.state == InteractionState.ALL_AUDIO_PROCESSED
        ):
            self._pending_interaction = OjinPersonaInteraction()
            self._pending_interaction.set_state(InteractionState.INACTIVE)
        elif self._interaction is None or self._interaction.state == InteractionState.INACTIVE:
            self._interaction = new_interaction or OjinPersonaInteraction()
            self._interaction.set_state(InteractionState.ACTIVE)
            assert self._client is not None
            start_generation_frame_index = (
                self.last_queued_frame_index
                if self.last_queued_frame_index > self.last_played_frame_index
                else self.current_frame_index
            )
            start_generation_frame_index += self.extra_frames_lat
            logger.info(f"Starting interaction at frame index: {start_generation_frame_index}")
            interaction_id = await self._client.start_interaction(
                params={
                    "client_frame_index": start_generation_frame_index,
                    "filter_amount": SPEECH_FILTER_AMOUNT,
                    "mouth_opening_scale": SPEECH_MOUTH_OPENING_SCALE,
                    "source_keyframes_index": active_keyframes_slot,
                    "destination_keyframes_index": keyframe_slot_to_update,
                }
            )
            logger.debug(f"Started interaction with id: {interaction_id}")
            self._interaction.interaction_id = interaction_id
        else:
            logger.exception(
                f"Error trying to start interaction in interaction_state: {self._interaction.state} and persona_state: {self.persona_state}"
            )

    async def _end_interaction(self):
        """End the current interaction.

        Sets received_all_interaction_inputs flag to True, which will trigger cleanup
        once all queued audio has been processed.
        """
        logger.debug("Ending interaction")
        if self._client is None:
            return

        # Cover case where speech audio arrives before initialization. We mark pending interaction as received stop frame to later end it when it becomes the active interaction.
        if (
            self.persona_state == PersonaState.INITIALIZING
            and self._pending_interaction is not None
        ):
            self._pending_interaction.has_received_stop_frame = True
            return

        if self._interaction is None:
            logger.error("_end_interaction but no interaction is set")
            return

        self._interaction.set_state(InteractionState.ALL_AUDIO_PROCESSED)
        await self._client.send_message(
            OjinPersonaEndInteractionMessage(
                interaction_id=self._interaction.interaction_id,
            )
        )

    async def _process_queued_audio(self):
        while True:
            if (
                self._client is None
                or self._interaction is None
                or self._interaction.state == InteractionState.ALL_AUDIO_PROCESSED
            ):
                await asyncio.sleep(0.001)
                continue

            # Mechanism to end interaction if we have audio starvation, considering how many video frames are left to play
            if self._settings.frame_count_threshold_for_end_interaction != -1:
                raise Exception(
                    "frame_count_threshold_for_end_interaction is not supported for now since it won't work with small audio sizes"
                )
                # if (
                #     len(self._interaction.pending_audio) == 0
                #     and self._interaction.received_frames
                #     >= self._settings.frame_count_threshold_for_end_interaction
                #     and (
                #         self.num_speech_frames_played
                #         + self._settings.frame_count_threshold_for_end_interaction
                #         > self._interaction.expected_frames
                #     )
                # ):
                #     logger.debug(
                #         f"Ending interaction because loop doesn't have enough frames queued: expected: {self._interaction.expected_frames}, played: {self.num_speech_frames_played}"
                #     )
                #     await self._end_interaction()

            # while there is more audio coming we wait for it if we don't have any to process atm
            if len(self._interaction.pending_audio) == 0:
                await asyncio.sleep(0.005)
                continue

            audio_bytes = self._interaction.pending_audio.copy()
            self._interaction.pending_audio.clear()
            logger.info(f"sending {len(audio_bytes)} audio bytes to server")
            await self._client.send_message(
                OjinPersonaInteractionInputMessage(
                    audio_int16_bytes=audio_bytes, interaction_id=self._interaction.interaction_id
                )
            )
            with self.pedning_audio_mutex:
                self.pending_audio_to_play.extend(audio_bytes)
            if self._interaction.has_received_stop_frame:
                await self._end_interaction()

    def _get_idle_frame_for_index(self, index: int) -> AnimationKeyframe:
        mirror_frame_idx = mirror_index(
            index, len(self.idle_frames), 2 if self.is_mirrored_loop else 1
        )
        return self.idle_frames[mirror_frame_idx]

    def get_next_pending_frame_and_audio(self) -> tuple[AnimationKeyframe, bytes]:
        frame_duration = 1 / self.fps
        audio_bytes_length_for_one_frame = 2 * int(frame_duration * OJIN_PERSONA_SAMPLE_RATE)
        frame = self.pending_speech_frames.popleft()
        with self.pedning_audio_mutex:
            audio = self.pending_audio_to_play[:audio_bytes_length_for_one_frame]
            self.pending_audio_to_play = self.pending_audio_to_play[
                audio_bytes_length_for_one_frame:
            ]
        return frame, audio

    async def _run_loop(self):
        while self.persona_state == PersonaState.INITIALIZING:
            await asyncio.sleep(0.1)

        silence_duration = 1 / self.fps
        audio_bytes_length_for_one_frame = 2 * int(silence_duration * OJIN_PERSONA_SAMPLE_RATE)
        silence_audio_for_one_frame = b"\x00" * audio_bytes_length_for_one_frame

        start_ts = time.perf_counter()
        self.played_frame_idx = -1
        while True:
            elapsed_time = time.perf_counter() - start_ts
            next_frame_idx = int(elapsed_time * self.fps)
            if next_frame_idx <= self.current_frame_index:
                next_frame_time = (self.current_frame_index + 1) * 0.04
                waiting_time = next_frame_time - elapsed_time - 0.005
                await asyncio.sleep(max(0, waiting_time))

                # spin lock
                elapsed_time = time.perf_counter() - start_ts
                next_frame_idx = self.current_frame_index + 1
                calculated_frame_idx = int(elapsed_time * self.fps)
                while calculated_frame_idx < next_frame_idx:
                    elapsed_time = time.perf_counter() - start_ts
                    calculated_frame_idx = int(elapsed_time * self.fps)

            audio_to_play = silence_audio_for_one_frame
            self.current_frame_index = next_frame_idx
            if (
                len(self.pending_speech_frames) != 0
                and self.pending_speech_frames[0].frame_idx <= self.current_frame_index
            ):
                animation_frame, audio_to_play = self.get_next_pending_frame_and_audio()
                logger.debug(
                    f"played frame {animation_frame.frame_idx} ==? {self.current_frame_index}"
                )
                self.num_speech_frames_played += 1
                if animation_frame.is_final_frame:
                    # Restart number of frames for next interaction (which might already be generating frames)
                    self.num_speech_frames_played = 0

                self.played_frame_idx = animation_frame.frame_idx
            else:
                if self._interaction is not None and self.num_speech_frames_played > 0:
                    logger.debug(f"frame missed: {self.current_frame_index}")
                    self.current_frame_index -= 1

                    silence_5ms = b"\x00\x00" * int(0.005 * OJIN_PERSONA_SAMPLE_RATE)
                    audio_frame = OutputAudioRawFrame(
                        audio=silence_5ms,
                        sample_rate=OJIN_PERSONA_SAMPLE_RATE,
                        num_channels=1,
                    )
                    await self.push_frame(audio_frame)
                    await asyncio.sleep(0.005)
                    continue

                self.played_frame_idx += 1
                self.num_speech_frames_played = 0
                animation_frame = self._get_idle_frame_for_index(self.played_frame_idx)
                logger.debug(f"played idle frame: {self.played_frame_idx}")

            image_frame = OutputImageRawFrame(
                image=animation_frame.image, size=self._settings.image_size, format="RGB"
            )
            audio_frame = OutputAudioRawFrame(
                audio=audio_to_play,
                sample_rate=OJIN_PERSONA_SAMPLE_RATE,
                num_channels=1,
            )
            await self.push_frame(image_frame)
            await self.push_frame(audio_frame)


def mirror_index(index: int, size: int, period: int = 2):
    """Calculate a mirrored index for creating a ping-pong animation effect.

        This method maps a continuously increasing index to a back-and-forth pattern
        within the given size, creating a ping-pong effect for smooth looping animations.

    Args:
            index (int): The original frame index
            size (int): The number of available frames
            period (int): Period of the mirrored indices

    Returns:
            int: The mirrored index that creates the ping-pong effect

    #
    """
    turn = index // size
    res = index % size
    if turn % period == 0:
        return res
    else:
        return size - res - 1
