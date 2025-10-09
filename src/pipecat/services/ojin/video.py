import asyncio
from collections import deque
import math
import time
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
    STARTING = "starting"
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
    expected_frames: float = 0
    
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

        logger.debug(
            f"Old Interaction state: {self.state}, New Interaction state: {new_state}"
        )
        self.state = new_state

    def update_expected_frames(self, audio_buffer_len: int, sample_rate: int, fps: int):
        bytes_per_sample = 2
        samples_count = audio_buffer_len // bytes_per_sample
        self.expected_frames += (samples_count / sample_rate) * fps
        logger.debug(
            f"pushing audio samples count: {samples_count}, next expected frames: {self.expected_frames}"
        )
        logger.debug(
            f"pushing audio samples count: {samples_count}, next expected frames: {self.expected_frames}"
        )


@dataclass
class OjinPersonaSettings:
    """Settings for Ojin Persona service.

    This class encapsulates all configuration parameters for the OjinPersonaService.
    """

    api_key: str = field(default="")  # api key for Ojin platform
    ws_url: str = field(
        default="wss://models.ojin.ai/realtime"
    )  # websocket url for Ojin platform
    client_connect_max_retries: int = field(
        default=3
    )  # amount of times it will try to reconnect to the server
    client_reconnect_delay: float = field(
        default=3.0
    )  # time between connecting retries
    persona_config_id: str = field(
        default=""
    )  # config id of the persona to use from Ojin platform
    image_size: Tuple[int, int] = field(default=(1920, 1080))
    cache_idle_sequence: bool = field(
        default=False
    )  # whether to cache the idle sequence loop to avoid doing inference while persona is not speaking
    idle_sequence_duration: int = field(
        default=30
    )  # length of the idle sequence loop in seconds.
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
        default=35
    )  # If the number of frames in the loopback is less than or equal to this value then end the interaction to avoid frame misses.


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
            )
        else:
            self._client = client

        assert self._settings.persona_config_id is not None

        self._processed_queued_audio_task: Optional[asyncio.Task] = None
        self._run_loop_task: Optional[asyncio.Task] = None

        self._resampler = create_default_resampler()
        self._server_fps_tracker = FPSTracker("OjinPersonaService")
        self._fsm_fps_tracker = FPSTracker("OjinPersonaService")

        self._receive_msg_task: Optional[asyncio.Task] = None

        self._interaction: Optional[OjinPersonaInteraction] = None
        self._pending_interaction: Optional[OjinPersonaInteraction] = None
        
        self.persona_state = PersonaState.INVALID
        self.idle_frames: list[AnimationKeyframe] = []  # Keyframes of the idle animation
        self.pending_speech_frames : deque[AnimationKeyframe] = deque()

        self.num_speech_frames_played: int = 0
        self.is_mirrored_loop: bool = True
        self.fps = 25
        self.current_frame_index = 0

        self.pending_audio_to_play = bytearray()

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
        raise Exception("Failed to connect to Ojin")

    # Disabled for now since it was causing issues with the server not processing all audio
    async def _incomming_frame_task(self):
        while True:
            if self._fsm is not None:
                time_since_transition = (
                    time.perf_counter() - self._fsm._transition_timestamp
                )

                if (
                    self._fsm._state == PersonaState.SPEECH
                    and self._fsm._waiting_for_image_frames
                    and self._last_frame_timestamp is not None
                ):
                    last_received_frame_time = (
                        time.perf_counter() - self._last_frame_timestamp
                    )

                    if last_received_frame_time > 1.5:
                        logger.info("Ending interaction")
                        # We send the Cancel interaction message because we don't send the "last_audio" flag
                        # to the server, therefore the server won't be able to send the last frame and reset the model
                        # Cancelation message resets the model instead.
                        await self.push_ojin_message(
                            OjinPersonaCancelInteractionMessage(
                                interaction_id=self._interaction.interaction_id,
                            )
                        )
                        await self._fsm.on_conversation_signal(
                            ConversationSignal.NO_MORE_IMAGE_FRAMES_EXPECTED
                        )
                        self._last_frame_timestamp = None
                elif (
                    self._fsm._state == PersonaState.SPEECH
                    and self._fsm._waiting_for_image_frames
                    and time_since_transition > 2.5
                ):
                    logger.warning(
                        "No Frames received from the server, stopping interaction by timeout"
                    )
                    # We send the cancel Interaction message to reset the state even when we didn't receive any frame
                    await self.push_ojin_message(
                        OjinPersonaCancelInteractionMessage(
                            interaction_id=self._interaction.interaction_id,
                        )
                    )
                    await self._fsm.on_conversation_signal(
                        ConversationSignal.NO_MORE_IMAGE_FRAMES_EXPECTED
                    )

            await asyncio.sleep(0.01)

    async def _stop(self):
        """Stop the persona service and clean up resources.

        Cancels all running tasks, closes connections, and resets the state.
        """
        # Cancel queued audio processing task
        if self._audio_input_task:
            await self.cancel_task(self._audio_input_task)
            self._audio_input_task = None

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._client:
            await self._client.close()
            self._client = None

        # Clear all buffers
        await self._interrupt()

        if self._fsm:
            await self._fsm.close()
            self._fsm = None

        logger.debug(f"OjinPersonaService {self._settings.persona_config_id} stopped")

    @property
    def fsm_fps_tracker(self):
        return self._fsm.fps_tracker

    @property
    def server_fps_tracker(self):
        return self._server_fps_tracker

    def _start_pushing_audio_output(self):
        logger.warning("Start pushing audio output")
        self._audio_output_task = self.create_task(self._push_audio_output())

    async def _stop_pushing_audio_output(self):
        logger.warning("Stop pushing audio output")
        if self._audio_output_task:
            await self.cancel_task(self._audio_output_task)
            self._audio_output_task = None

    async def _push_audio_output(self):
        """Continuously push audio output to the proxy."""
        while True:
            assert (
                self._interaction is not None
                and self._interaction.audio_output_queue is not None
            )
            try:
                audio_frame = await self._interaction.audio_output_queue.get()
                await self.push_frame(audio_frame, direction=FrameDirection.DOWNSTREAM)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.05)

    async def _receive_messages(self):
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

    async def push_ojin_message(self, message: BaseModel):
        """Send a message to the proxy.

        Args:
            message: The message to send to the proxy

        """
        assert self._client is not None
        if hasattr(message, "interaction_id"):
            logger.info(
                f"Sending message type {type(message)} with interaction {message.interaction_id}"
            )

        await self._client.send_message(message)

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
                self.is_mirrored_loop = message.parameters.get(
                    "is_mirrored_loop", True
                )

            logger.info("Sending silence to init idle frames")
            # NOTE(mouad): IDLE frames generation
            interaction_id = await self._client.start_interaction(
                params={
                    "filter_amount": IDLE_FILTER_AMOUNT,
                    "mouth_opening_scale": IDLE_MOUTH_OPENING_SCALE,
                    "source_keyframes_index": IDLE_ANIMATION_KEYFRAMES_SLOT,
                    "destination_keyframes_index": IDLE_ANIMATION_KEYFRAMES_SLOT,
                }
            )
            silence_duration = self._settings.idle_sequence_duration
            silence_audio = b"\x00\x00" * int(silence_duration * OJIN_PERSONA_SAMPLE_RATE)
            self.num_speech_frames_played = 0
            await self._client.send_message(
                OjinPersonaInteractionInputMessage(
                    audio_int16_bytes=silence_audio,
                    interaction_id=interaction_id,
                )
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
            )
            if self.persona_state == PersonaState.INITIALIZING:
                # IDLE frame
                self.idle_frames.append(
                    animation_frame
                )
            else:
                self.pending_speech_frames.append(
                    animation_frame
                )
            
            if message.is_final_response:
                self._interaction = None
                if self._pending_interaction is not None:
                    await self._start_speech_interaction(
                        self._pending_interaction
                    )
                    self._pending_interaction = None
                
                if self.persona_state == PersonaState.INITIALIZING:
                    self._run_loop_task = self.create_task(self._run_loop())
                    self.persona_state = PersonaState.ACTIVE
                    await self.push_frame(
                        OjinPersonaInitializedFrame(), direction=FrameDirection.DOWNSTREAM
                    )
                    await self.push_frame(
                        OjinPersonaInitializedFrame(), direction=FrameDirection.UPSTREAM
                    )

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

        self._receive_msg_task            = self.create_task(self._receive_ojin_messages())
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

        logger.debug(f"OjinPersonaService {self._settings.persona_config_id} stopped")

    
    async def _interrupt(self):
        """Interrupt the current interaction.

        Sends a cancel message to the backend, updates the FSM state, and
        cleans up the current interaction.
        """
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
            self.pending_audio_to_play.clear()
            self.pending_speech_frames.clear()

            self._interaction.set_state(InteractionState.INACTIVE)

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

        elif isinstance(frame, TTSStoppedFrame):
            logger.debug("TTSStoppedFrame")
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

    async def _interrupt(self):
        """Interrupt the current interaction.

        Sends a cancel message to the backend, updates the FSM state, and
        cleans up the current interaction.
        """
        if self._interaction is None or self._interaction.interaction_id is None:
            logger.debug("Trying to interrupt an interaction but none is active")
            return

        logger.debug(f"Try interrupt interaction in state {self._interaction.state}")
        if self._interaction.state != InteractionState.INACTIVE:
            logger.warning("Sending CancelInteractionMessage")
            await self.push_ojin_message(
                OjinPersonaCancelInteractionMessage(
                    interaction_id=self._interaction.interaction_id,
                )
            )
            if self._fsm is not None:
                await self._fsm.on_conversation_signal(
                    ConversationSignal.USER_INTERRUPTED_AI
                )

            self._close_interaction()

    async def _start_interaction(
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
        self._interaction = new_interaction or OjinPersonaInteraction(
            persona_id=self._settings.persona_config_id,
        )
        self._interaction.num_loop_frames = (
            self._settings.idle_sequence_duration * 25
        )  # 25 fps
        self._interaction.set_state(InteractionState.STARTING)

        assert self._client is not None
        extra_frames_lat = 10 # TODO(mouad): compute the appropriate index
        logger.info(f"Starting interaction at frame index: {self.current_frame_index + extra_frames_lat}")
        interaction_id = await self._client.start_interaction(
            params={
                "client_frame_index": self.current_frame_index + extra_frames_lat, 
                "filter_amount": SPEECH_FILTER_AMOUNT,
                "mouth_opening_scale": SPEECH_MOUTH_OPENING_SCALE,
                "source_keyframes_index": active_keyframes_slot,
                "destination_keyframes_index": keyframe_slot_to_update,
            }
        )
        logger.debug(f"Started interaction with id: {interaction_id}")
        self._interaction.interaction_id = interaction_id

    async def _end_interaction(self):
        """End the current interaction.

        Sets received_all_interaction_inputs flag to True, which will trigger cleanup
        once all queued audio has been processed.
        """
        # TODO Handle possible race conditions i.e. when _interaction.state == STARTING
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
            
            if len(self._interaction.pending_audio) == 0 and (
                self.num_speech_frames_played
                + self._settings.frame_count_threshold_for_end_interaction
                > self._interaction.expected_frames
            ):
                logger.debug(
                    f"Ending interaction because loop doesn't have enough frames queued: expected: {self._interaction.expected_frames}, played: {self.num_speech_frames_played}"
                )
                await self._end_interaction()

            # while there is more audio coming we wait for it if we don't have any to process atm
            if len(self._interaction.pending_audio) == 0:
                await asyncio.sleep(0.005)
                continue

            # Get audio from the queue
            should_finish_task = False
            try:
                message: OjinPersonaInteractionInputMessage = (
                    self._interaction.audio_input_queue.get_nowait()
                )
                message.interaction_id = self._interaction.interaction_id
                should_finish_task = True
            except asyncio.QueueEmpty:
                should_finish_task = False
                assert False, "the queue should never be empty here"

            message.params = {
                "start_frame_idx": self._interaction.start_frame_idx,
                "filter_amount": self._interaction.filter_amount,
                "mouth_opening_scale": self._interaction.mouth_opening_scale,
                "source_keyframes_index": self._interaction.source_keyframes_index,
                "destination_keyframes_index": self._interaction.destination_keyframes_index,
            }
            logger.debug(f"Sending audio int16: {len(message.audio_int16_bytes)}")

            await self.push_ojin_message(message)
            await self.enqueue_audio_output(message.audio_int16_bytes)

            if should_finish_task:
                self._interaction.audio_input_queue.task_done()

    async def enqueue_audio_output(self, audio: bytes):
        """Enqueue audio data to be sent as output frames.

        This method creates an OutputAudioRawFrame from the raw audio bytes
        and adds it to the current interaction's audio output queue.

        Args:
            audio: Raw audio bytes to be sent as output

        """
        assert self._interaction and self._interaction.audio_output_queue is not None
        await self._interaction.audio_output_queue.put(
            OutputAudioRawFrame(
                audio=audio,
                sample_rate=OJIN_PERSONA_SAMPLE_RATE,
                num_channels=1,
            )
        )


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

    mirror_frame_idx: int
    frame_idx: int
    image: bytes


class PersonaPlaybackLoop:
    """Manages a complete idle animation loop with synchronized audio and video."""

    id: int = 0
    duration: int = 0  # seconds
    frames: list[AnimationKeyframe] = []  # Keyframes of the idle animation
    playback_time: float = 0  # Total elapsed playback time in seconds
    is_mirrored_loop: bool = False  # whether to mirror the idle loop

    def __init__(
        self,
        duration: int,
        fps: int = 25,
    ):
        """Initialize the persona idle loop animation.

        Args:
            duration (int): The total duration of the animation in seconds
            fps (int, optional): Frames per second for the animation. Defaults to 25.

        """
        self.duration = duration
        self.fps = fps

    def num_frames(self) -> int:
        return len(self.frames)

    def add_frame(self, image: bytes) -> AnimationKeyframe:
        """Get an existing keyframe or create a new one at the specified frame index.

        Args:
            image (bytes): The image data for the frame

        Returns:
            AnimationKeyframe: The existing or newly created keyframe

        """
        frame_idx = len(self.frames)
        expected_frames = self.duration * self.fps
        keyframe = AnimationKeyframe(
            mirror_frame_idx=mirror_index(frame_idx, expected_frames, 2 if self.is_mirrored_loop else 1),
            frame_idx=frame_idx,
            image=image,
        )
        self.frames.append(keyframe)
        return keyframe

    def get_frame(self, frame_idx: int) -> AnimationKeyframe | None:
        """Get an existing keyframe or create a new one at the specified frame index.

        Args:
            frame_idx (int): The frame index to retrieve or create

        Returns:
            AnimationKeyframe: The existing or newly created keyframe

        """
        for keyframe in self.frames:
            if keyframe.frame_idx == frame_idx:
                return keyframe

        logger.error(f"Couldn't find idle frame frame_idx: {frame_idx}")
        return None

    def get_frame_at_time(self, playback_time: float) -> AnimationKeyframe:
        """Retrieve the animation keyframe at a specific playback time.

        Args:
            playback_time (float): The time in seconds to get the frame for

        Returns:
            AnimationKeyframe: The keyframe corresponding to the given playback time

        """
        # Get total frames passed
        current_frame_idx = math.floor(playback_time * self.fps)

        mirror_frame_idx = mirror_index(current_frame_idx, len(self.frames), 2 if self.is_mirrored_loop else 1)

        return self.frames[mirror_frame_idx]

    def get_current_idle_frame(self) -> AnimationKeyframe | None:
        """Get the keyframe at the current playback time.

        Returns:
            AnimationKeyframe | None: The current keyframe or None if no keyframes exist

        """
        return self.get_frame_at_time(self.playback_time)

    def get_current_frame_idx(self) -> int:
        """Get the absolute key frame idx at the current playback time

        Returns:
            int: The current keyframe idx

        """
        current_frame_idx = math.floor(self.playback_time * self.fps)
        return current_frame_idx

    def get_playback_time(self) -> float:
        return self.playback_time

    def seek(self, playback_time: float):
        """Set the playback time to a specific position.

        Args:
            playback_time (float): The time in seconds to set as the current playback position

        """
        self.playback_time = playback_time

    def seek_frame(self, frame_idx: int):
        """Set the playback position to a specific frame.

        Args:
            frame_idx (int): The frame index to seek to

        """
        self.playback_time = frame_idx / self.fps

    def step(self, delta_time: float):
        """Advance the animation by the specified time delta.

        Args:
            delta_time (float): The time in seconds to advance the animation

        """
        self.playback_time += delta_time


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
