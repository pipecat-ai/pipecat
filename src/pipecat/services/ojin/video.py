"""Ojin Avatar implementation for Pipecat."""

import asyncio
import time
import math
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Optional, Tuple, Callable, Awaitable

# Will use numpy when implementing avatar-specific processing
from loguru import logger
from ojin.ojin_avatar_client import OjinAvatarClient
from ojin.ojin_avatar_messages import (
    IOjinAvatarClient,
    OjinAvatarCancelInteractionMessage,
    OjinAvatarInteractionInputMessage,
    OjinAvatarInteractionResponseMessage,
    OjinAvatarSessionReadyMessage,
    StartInteractionMessage,
    StartInteractionResponseMessage,
)
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
from pydantic import BaseModel

class OjinAvatarInitializedFrame(Frame):
    """Frame indicating that the avatar has been initialized and can now output frames."""

    pass


class InteractionState(Enum):
    """Enum representing the possible states of an avatar interaction.

    These states track the lifecycle of an interaction from creation to completion.
    """

    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    ENDING = "ending"
    WAITING_FOR_LAST_FRAME = "waiting_for_last_frame"

@dataclass
class OjinAvatarSettings:
    """Settings for Ojin Avatar service.

    This class encapsulates all configuration parameters for the OjinAvatarService.
    """

    api_key: str = field(default="") # api key for Ojin platform
    ws_url: str = field(default="wss://models.ojin.ai/realtime") # websocket url for Ojin platform
    avatar_config_id: str = field(default="")    # config id of the avatar to use from Ojin platform
    image_size: Tuple[int, int] = field(default=(1920, 1080))
    cache_idle_sequence: bool = field(default=False) # whether to cache the idle sequence loop to avoid doing inference while avatar is not speaking
    idle_sequence_duration: int = field(default=30) # length of the idle sequence loop in seconds.
    idle_to_speech_seconds: float = field(default=0.75) # seconds to wait before starting speech, recommended not less than 0.75 to avoid missing frames. This ensures smooth transition between idle frames and speech frames
    tts_audio_passthrough: bool = field(default=False) # whether to pass through TTS audio to the output
    push_bot_stopped_speaking_frames: bool = field(default=True) # whether to push bot stopped speaking frames to the output



class ConversationSignal(Enum):
    """Possible states of the conversation."""

    SPEECH_AUDIO_STARTED_PROCESSING = "speech_audio_started_processing"
    USER_INTERRUPTED_AI = "user_interrupted_ai"
    NO_MORE_IMAGE_FRAMES_EXPECTED = "no_more_image_frames_expected"


class AvatarState(Enum):
    """Enum representing the possible states of the avatar conversation FSM."""

    INVALID = "invalid"
    INITIALIZING = "initializing"
    IDLE = "idle"
    IDLE_TO_SPEECH = "idle_to_speech"
    SPEECH = "speech"


class OjinAvatarFSM:
    """Finite State Machine for managing avatar conversational states.

    This class manages the different states of an avatar during a conversation,
    including initialization, idle animations, speech, and transitions between states.
    It also handles the caching of idle frames for efficient playback.

    Attributes:
        idle_loop (AvatarIdleLoop): The idle animation loop for the avatar
        current_state (AvatarState): The current state of the avatar
        fps (int): Frames per second for animations

    """

    def __init__(
        self,
        frame_processor: FrameProcessor,
        settings: OjinAvatarSettings,
        on_state_changed_callback: Callable[
            [AvatarState, AvatarState], Awaitable[None]
        ],
    ):
        self._settings = settings
        self._frame_processor = frame_processor
        self._state = AvatarState.INVALID
        self._num_frames_missed = 0
        self._playback_loop = AvatarPlaybackLoop(settings.idle_sequence_duration, 25)
        self._speech_frames: asyncio.Queue[OutputImageRawFrame] = asyncio.Queue()
        self._transition_time: float = -1
        self._last_frame_idx: int = -1
        self._playback_task: Optional[asyncio.Task] = None
        self._waiting_for_image_frames = False
        self._last_frame: Optional[OutputImageRawFrame] = None
        self.on_state_changed_callback = on_state_changed_callback

    async def start(self):
        await self.set_state(AvatarState.INITIALIZING)

    async def close(self):
        await self._stop_playback()

    async def set_state(self, new_state: AvatarState):
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        logger.debug(f"set_state from {old_state} to {new_state}")
        self.on_state_changed(old_state, new_state)
        await self.on_state_changed_callback(old_state, new_state)

    def get_transition_frame_idx(self) -> int:
        logger.debug(f"get_transition_frame_idx: {self._transition_time} frame {int(self._transition_time * 25)}")
        return int(self._transition_time * 25)

    def get_state(self) -> AvatarState:
        """Get the current state of the avatar FSM.

        Returns:
            The current avatar state from the AvatarState enum

        """
        return self._state

    def receive_image_frame(self, image_frame: OutputImageRawFrame):
        """Process incoming image frames based on the current avatar state.

        During initialization, frames are added to the idle animation loop.
        In other states, frames are queued as speech animation frames.

        Args:
            image_frame: The raw image frame to process

        """
        # While initializing we consider frames as idle frames
        if self._state == AvatarState.INITIALIZING:
            self._playback_loop.add_frame(image_frame.image)

        # On any other state these would be speech frames
        else:
            self._speech_frames.put_nowait(image_frame)

    async def on_conversation_signal(self, signal: ConversationSignal):
        """Handle conversation signals to update the avatar state.

        Processes signals such as user interruptions, speech starting,
        and frame completion to trigger appropriate state transitions.

        Args:
            signal: The conversation signal to process

        """
        logger.debug(f"{signal}")
        match signal:
            case ConversationSignal.USER_INTERRUPTED_AI:
                await self.interrupt()
                pass

            case ConversationSignal.SPEECH_AUDIO_STARTED_PROCESSING:
                await self.set_state(AvatarState.IDLE_TO_SPEECH)
                pass

            case ConversationSignal.NO_MORE_IMAGE_FRAMES_EXPECTED:
                self._waiting_for_image_frames = False
                if self._state == AvatarState.INITIALIZING:
                    await self.set_state(AvatarState.IDLE)

                pass

    async def interrupt(self):
        """Interrupt the current speech animation.

        Clears any queued speech frames and transitions the avatar back to
        the IDLE state if it was in a speaking state.

        """
        if self._state in (
            AvatarState.SPEECH,
            AvatarState.IDLE_TO_SPEECH,
        ):
            while not self._speech_frames.empty():
                self._speech_frames.get_nowait()

            await self.set_state(AvatarState.IDLE)  # Corrected from DittoState

    def _start_playback(self):
        """Start the animation playback loop.

        Creates an asynchronous task to run the animation loop if one doesn't
        already exist. Initializes the timing for frame rate control.

        """
        logger.debug("Starting playback loop")

        if self._playback_task is not None:
            return
        self.last_update_time = time.perf_counter()
        self._playback_task = asyncio.create_task(self._run())

    async def _stop_playback(self):
        """Stop the animation playback loop.

        Cancels the running playback task if one exists and waits for it to
        complete before cleaning up references.

        """
        logger.debug("Stopping playback loop")
        if self._playback_task is None:
            return

        self._playback_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._playback_task
        self._playback_task = None

    async def _run(self):
        """Run the main animation loop continuously while playback is active.

        Handle frame timing, state transitions, and frame processing based on
        the current avatar state. Manage transitions between idle and speech states,
        detect when speech has ended, and push frames to the output pipeline.

        """
        while self._playback_task is not None:
            delta_time = time.perf_counter() - self.last_update_time
            self._playback_loop.step(delta_time)
            self.last_update_time = time.perf_counter()

            if (
                self._state == AvatarState.IDLE_TO_SPEECH
                and self._playback_loop.get_playback_time() >= self._transition_time
            ):
                await self.set_state(AvatarState.SPEECH)

            if (
                not self._waiting_for_image_frames
                and self._speech_frames.empty()
                and (
                    self._state == AvatarState.SPEECH
                    or self._state == AvatarState.IDLE_TO_SPEECH
                )
            ):
                logger.debug("Speech ended!!!")
                await self.set_state(AvatarState.IDLE)

            # Process output image frames
            if self.should_process_output():
                frame = await self.get_next_avatar_frame()
                if frame is not None:
                    await self._frame_processor.push_frame(frame)

            await asyncio.sleep(0.005)

        logger.debug("Playback loop stopped")

    def should_process_output(self) -> bool:
        """Determine if the avatar should process and output frames.

        Returns:
            True if the avatar is in a state where it should output frames,
            False otherwise

        """
        return self._state in (
            AvatarState.IDLE,
            AvatarState.SPEECH,
            AvatarState.IDLE_TO_SPEECH,
        )

    def on_state_changed(self, old_state: AvatarState, new_state: AvatarState):
        """Handle state transitions in the avatar FSM.

        This method is called whenever the avatar's state changes, allowing for
        state-specific behavior to be implemented.

        Args:
            new_state: The new state that the avatar has transitioned to

        """
        match new_state:
            case AvatarState.INITIALIZING:
                self._waiting_for_image_frames = True

            case AvatarState.IDLE:
                if old_state == AvatarState.INITIALIZING:
                    self._start_playback()

            case AvatarState.SPEECH:
                pass

            case AvatarState.IDLE_TO_SPEECH:
                self._waiting_for_image_frames = True
                self._transition_time = (
                    self._playback_loop.get_playback_time()
                    + self._settings.idle_to_speech_seconds
                )

            case _:
                logger.debug(f"State: {self._state} - Unknown state")

    async def get_next_avatar_frame(self) -> OutputImageRawFrame | None:
        """Get the next frame to display based on the current avatar state.

        Retrieves either an idle animation frame or a speech animation frame
        depending on the current state. If a speech frame is expected but not
        available, falls back to an idle frame and tracks missed frames.

        Returns:
            The next image frame to display, or None if no new frame is available

        """
        # Wait until current frame idx is different than the last one (frame steps of 25 fps)
        if self._last_frame_idx == self._playback_loop.get_current_frame_idx():
            return None

        self._last_frame_idx = self._playback_loop.get_current_frame_idx()

        match self._state:
            case AvatarState.IDLE | AvatarState.IDLE_TO_SPEECH:
                if self._last_frame_idx % 25 == 0:
                    logger.debug(f"Pushing idle frame: {self._last_frame_idx}")
                    
                idle_frame = self._playback_loop.get_current_frame()
                image_frame = OutputImageRawFrame(
                    image=idle_frame.image,
                    size=self._settings.image_size,
                    format="BGR",
                )
                image_frame.pts = idle_frame.frame_idx
                return image_frame
            case AvatarState.SPEECH:
                try:
                    frame = self._speech_frames.get_nowait()
                except asyncio.QueueEmpty:
                    frame = None

                if frame is None:
                    self._num_frames_missed += 1
                    logger.warning(f"Frames missed {self._num_frames_missed}")

                    if self._last_frame is not None:
                        return self._last_frame

                    # For now we push an idle frame instead
                    idle_frame = self._playback_loop.get_current_frame()
                    assert idle_frame is not None
                    image_frame = OutputImageRawFrame(
                        image=idle_frame.image,
                        size=self._settings.image_size,
                        format="BGR",
                    )
                    image_frame.pts = idle_frame.frame_idx

                    return image_frame

                else:
                    self._last_frame = frame
                    if frame.pts % 25 == 0:
                        logger.debug(
                            f"Pushing speech frame: {frame.pts} ==? {self._last_frame_idx}"
                        )
                    self._num_frames_missed = 0

                return frame
            case _:
                return None


OJIN_AVATAR_SAMPLE_RATE=16000
SPEECH_FILTER_AMOUNT = 0.0
IDLE_FILTER_AMOUNT = 1.0
IDLE_MOUTH_OPENING_SCALE = 0.0
SPEECH_MOUTH_OPENING_SCALE = 1.0

@dataclass
class OjinAvatarInteraction:
    """Represents an interaction session between a user and the Ojin avatar.

    This class maintains the state of an ongoing interaction, including audio queues,
    frame tracking for animations, and interaction lifecycle state. It handles the
    buffering of audio inputs and manages the interaction's state transitions.
    """

    interaction_id: str = ""
    avatar_id: str = ""
    audio_input_queue: asyncio.Queue[OjinAvatarInteractionInputMessage] | None = None
    audio_output_queue: asyncio.Queue[OutputAudioRawFrame] | None = None
    pending_first_input: bool = True
    start_frame_idx: int | None = None
    frame_idx: int = 0
    filter_amount: float = 0.0
    mirrored_frame_idx: int = 0
    num_loop_frames: int = 0
    state: InteractionState = InteractionState.INACTIVE
    mouth_opening_scale: float = 0.0

    def __post_init__(self):
        """Initialize queues after instance creation."""
        if self.audio_input_queue is None:
            self.audio_input_queue = asyncio.Queue()
        if self.audio_output_queue is None:
            self.audio_output_queue = asyncio.Queue()

    def next_frame(self):
        """Advance to the next frame in the animation sequence.

        Updates the frame index and applies mirroring for smooth looping animations.
        """
        self.frame_idx += 1
        self.mirrored_frame_idx = mirror_index(self.frame_idx, self.num_loop_frames)

    def close(self):
        """Close the interaction."""
        if self.audio_input_queue is not None:
            while not self.audio_input_queue.empty():
                self.audio_input_queue.get_nowait()
                self.audio_input_queue.task_done()

        if self.audio_output_queue is not None:
            while not self.audio_output_queue.empty():
                self.audio_output_queue.get_nowait()
                self.audio_output_queue.task_done()

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

        old_state = self.state
        self.state = new_state        


class OjinAvatarService(FrameProcessor):
    """Ojin Avatar integration for Pipecat.

    This class provides integration between Ojin avatars and the Pipecat framework.
    """

    def __init__(
        self,
        settings: OjinAvatarSettings,
        client: IOjinAvatarClient | None = None,
    ) -> None:
        super().__init__()
        logger.debug(
            f"OjinAvatarService initialized with settings {settings}"
        )
        # Use provided settings or create default settings
        self._settings = settings
        if client is None:
            self._client = OjinAvatarClient(
                ws_url=settings.ws_url,
                api_key=settings.api_key,
                config_id=settings.avatar_config_id,
            )
        else:
            self._client = client

        self._fsm = OjinAvatarFSM(
            self,
            settings,
            on_state_changed_callback=self._on_state_changed,
        )

        # Generate a UUID if avatar_id is not provided
        assert self._settings.avatar_config_id is not None

        self._audio_input_task: Optional[asyncio.Task] = None
        self._audio_output_task: Optional[asyncio.Task] = None

        self._interaction: Optional[OjinAvatarInteraction] = None
        self._pending_interaction: Optional[OjinAvatarInteraction] = None

        self._resampler = create_default_resampler()

    async def _on_state_changed(
        self, old_state: AvatarState, new_state: AvatarState
    ) -> None:
        """Handle state transitions in the avatar FSM.

        This method is called when the avatar's state changes and performs
        state-specific initialization actions.

        Args:
            old_state: The previous state of the avatar
            new_state: The new state that the avatar has transitioned to

        """
        if new_state == AvatarState.INITIALIZING:
            # Send silence to avatar with idle_sequence_duration
            silence_duration = self._settings.idle_sequence_duration
            num_samples = silence_duration * OJIN_AVATAR_SAMPLE_RATE
            silence_audio = b"\x00\x00" * num_samples
            logger.debug(f"Sending {silence_duration}s of silence to initialize avatar")
            await self._start_interaction(is_speech=False)
            assert self._interaction is not None
            self._interaction.set_state(InteractionState.WAITING_FOR_LAST_FRAME)

            # Since start interaction happens locally it should be ready here to push input
            if self._interaction is None or self._interaction.interaction_id is None:
                return

            assert (
                self._interaction is not None
                and self._interaction.audio_input_queue is not None
            )
            await self._interaction.audio_input_queue.put(
                OjinAvatarInteractionInputMessage(
                    audio_int16_bytes=silence_audio,
                    interaction_id=self._interaction.interaction_id,
                    is_last_input=True,
                    params={
                        "start_frame_idx": self._interaction.start_frame_idx,
                        "filter_amount": self._interaction.filter_amount,
                        "mouth_opening_scale": self._interaction.mouth_opening_scale,
                    },
                )
            )

        if old_state == AvatarState.INITIALIZING and new_state == AvatarState.IDLE:
            await self.push_frame(
                OjinAvatarInitializedFrame(), direction=FrameDirection.DOWNSTREAM
            )
            await self.push_frame(
                OjinAvatarInitializedFrame(), direction=FrameDirection.UPSTREAM
            )

        if new_state == AvatarState.SPEECH and self._audio_output_task is None:
            await self._start_pushing_audio_output()

        if new_state == AvatarState.IDLE and self._audio_output_task is not None:
            if self._settings.push_bot_stopped_speaking_frames:
                await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            await self._stop_pushing_audio_output()

    async def _start(self):
        """Initialize the avatar service and start processing.

        Authenticates with the proxy and creates tasks for processing
        audio and receiving messages.
        """
        assert self._client is not None
        await self._client.connect()
        # Create tasks to process audio and video
        self._audio_input_task = self.create_task(self._process_queued_audio())
        self._receive_task = self.create_task(self._receive_messages())

    async def _stop(self):
        """Stop the avatar service and clean up resources.

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

        logger.debug(f"OjinAvatarService {self._settings.avatar_config_id} stopped")

    async def _start_pushing_audio_output(self):
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
            await self._handle_ojin_message(message)

    async def push_ojin_message(self, message: BaseModel):
        """Send a message to the proxy.

        Args:
            message: The message to send to the proxy

        """
        assert self._client is not None
        await self._client.send_message(message)

    async def _handle_ojin_message(self, message: BaseModel):
        """Process incoming messages from the proxy.

        Handles different message types including authentication responses,
        interaction ready notifications, and interaction responses containing
        video frames.

        Args:
            message: The message received from the proxy

        """
        if isinstance(message, OjinAvatarInteractionResponseMessage):
            # logger.debug(f"Video frame received: {self._interaction.frame_idx}")
            # Create and push the image frame
            image_frame = OutputImageRawFrame(
                image=message.video_frame_bytes,
                size=self._settings.image_size,
                format="BGR",
            )
            assert self._interaction is not None
            image_frame.pts = self._interaction.mirrored_frame_idx
            # Push the image frame to the FSM if it exists for advanced processing or directly to the output to outsource the processing to the client
            if self._fsm is not None:
                self._fsm.receive_image_frame(image_frame)
            else:
                logger.debug(
                    f"Video frame pushed (no fsm): {self._interaction.mirrored_frame_idx}"
                )
                if self._audio_output_task is None:
                    await self._start_pushing_audio_output()

                await self.push_frame(image_frame)

            self._interaction.next_frame()
            if message.is_final_response:
                logger.debug("No more video frames expected")
                self._close_interaction()
                if self._fsm is not None:
                    await self._fsm.on_conversation_signal(
                        ConversationSignal.NO_MORE_IMAGE_FRAMES_EXPECTED
                    )
                if self._pending_interaction is not None:
                    await self._start_interaction(
                        self._pending_interaction, is_speech=True
                    )
                    self._pending_interaction = None

        elif isinstance(message, OjinAvatarSessionReadyMessage):
            if self._fsm is not None:
                await self._fsm.start()

        elif isinstance(message, StartInteractionResponseMessage):
            assert self._interaction is not None
            self._interaction.interaction_id = message.interaction_id
            self._interaction.set_state(InteractionState.ACTIVE)

    def get_fsm_state(self) -> AvatarState:
        """Get the current state of the avatar's finite state machine.

        Returns:
            The current state of the avatar FSM or INVALID if no FSM exists

        """
        if self._fsm is not None:
            return self._fsm.get_state()
        return AvatarState.INVALID

    def is_tts_input_allowed(self) -> bool:
        """Check if the avatar is ready to receive TTS input.

        Returns:
            True if the avatar is in a state that can accept TTS input, False otherwise

        """
        return self.get_fsm_state() not in [
            AvatarState.INITIALIZING,
            AvatarState.INVALID,
        ]

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
            # TODO(@JM): Avoid ending interaction here since some TTS services continue to send audio frames
            if self._pending_interaction:
                self._pending_interaction = None
            else:
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
            if self.is_tts_input_allowed():
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
            return

        if self._interaction.state == InteractionState.ACTIVE:
            logger.debug("Interrupting interaction")
            await self.push_ojin_message(
                OjinAvatarCancelInteractionMessage(
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
        new_interaction: Optional[OjinAvatarInteraction] = None,
        is_speech: bool = False,
    ):
        """Start a new interaction with the avatar.

        Creates a new interaction or uses the provided one, initializes it with
        the appropriate state and parameters, and sends a start message to the backend.

        Args:
            new_interaction: Optional existing interaction to use instead of creating a new one
            is_speech: Whether this interaction is speech-based (True) or idle (False)

        """
        self._interaction = new_interaction or OjinAvatarInteraction(
            avatar_id=self._settings.avatar_config_id,
        )
        self._interaction.num_loop_frames = (
            self._settings.idle_sequence_duration * 25
        )  # 25 fps
        self._interaction.set_state(InteractionState.STARTING)
        if is_speech:
            self._interaction.filter_amount = SPEECH_FILTER_AMOUNT
            self._interaction.mouth_opening_scale = SPEECH_MOUTH_OPENING_SCALE
            if self._fsm is not None:
                await self._fsm.on_conversation_signal(
                    ConversationSignal.SPEECH_AUDIO_STARTED_PROCESSING
                )
                self._interaction.start_frame_idx = self._fsm.get_transition_frame_idx()
                self._interaction.frame_idx = self._fsm.get_transition_frame_idx()
        else:
            self._interaction.filter_amount = IDLE_FILTER_AMOUNT
            self._interaction.mouth_opening_scale = IDLE_MOUTH_OPENING_SCALE            
            self._interaction.start_frame_idx = 0
            self._interaction.frame_idx = 0

        await self.push_ojin_message(StartInteractionMessage())

        # immediately receive the ready message for now
        assert self._client is not None
        message = await self._client.receive_message()
        await self._handle_ojin_message(message)

    async def _end_interaction(self):
        """End the current interaction.

        Updates the interaction state to ENDING, which will trigger cleanup
        once all queued audio has been processed.
        """
        # TODO Handle possible race conditions i.e. when _interaction.state == STARTING
        if self._interaction is None:
            return

        self._interaction.set_state(InteractionState.ENDING)

    async def _handle_input_audio(self, frame: TTSAudioRawFrame):
        """Process incoming audio frames from the TTS service.

        Handles audio frames based on the current avatar state. If the avatar is not
        ready to receive input, the audio is queued for later processing. Otherwise,
        it starts or continues an active interaction.

        Resamples the audio to the target sample rate and either sends it directly
        to the backend if an interaction is running, or queues it for later processing.

        Args:
            frame: The audio frame to process

        """
        resampled_audio = await self._resampler.resample(
            frame.audio, frame.sample_rate, OJIN_AVATAR_SAMPLE_RATE
        )

        if not self.is_tts_input_allowed():
            if self._interaction is None or self._interaction.interaction_id is None:
                logger.debug("No interaction is set")
                return

            if self._pending_interaction is None:
                self._pending_interaction = OjinAvatarInteraction(
                    avatar_id=self._settings.avatar_config_id,
                )
                self._pending_interaction.set_state(InteractionState.INACTIVE)

            assert self._pending_interaction.audio_input_queue is not None
            self._pending_interaction.audio_input_queue.put_nowait(
                OjinAvatarInteractionInputMessage(
                    interaction_id=self._interaction.interaction_id,
                    audio_int16_bytes=resampled_audio,
                )
            )
            self._pending_interaction.pending_first_input = False
            logger.debug(
                f"Audio input is still not allowed (initializing), queing to pending interaction. Queue size: {self._pending_interaction.audio_input_queue.qsize()}"
            )
        else:
            if self._interaction is None:
                await self._start_interaction(is_speech=True)

            assert (
                self._interaction is not None
                and self._interaction.audio_input_queue is not None
            )
            # Queue the audio for later processing
            if self._interaction.pending_first_input:
                start_frame_idx = self._interaction.start_frame_idx
            else:
                start_frame_idx = None

            await self._interaction.audio_input_queue.put(
                OjinAvatarInteractionInputMessage(
                    interaction_id=self._interaction.interaction_id,
                    audio_int16_bytes=resampled_audio,
                    params={
                        "start_frame_idx": start_frame_idx,
                        "filter_amount": self._interaction.filter_amount,
                        "mouth_opening_scale": self._interaction.mouth_opening_scale,
                    }
                )
            )
            self._interaction.pending_first_input = False
            logger.debug(
                f"Queued audio for later processing. Queue size: {self._interaction.audio_input_queue.qsize()}"
            )

    def _close_interaction(self):
        """Close and clean up the current interaction.

        Clears the interaction queue and resets the interaction state.
        """
        # Clear the interaction queue if it exists
        if self._interaction is not None:
            self._interaction.close()
            self._interaction = None

    async def _process_queued_audio(self):
        """Process audio that was queued before an interaction was ready.

        Continuously monitors the audio queue of the current interaction and
        sends audio messages to the backend when available. Handles the final
        audio chunk specially to signal the end of input.
        """
        audio_buffer = b""

        while True:
            # Wait until we have a running interaction (starts with first audio input)
            if not self._interaction or self._interaction.audio_input_queue is None:
                await asyncio.sleep(0.001)
                continue

            is_final_message = (
                self._interaction.audio_input_queue.qsize() == 1
                and self._interaction.state == InteractionState.ENDING
            )           

            # while there is more audio coming we wait for it if we don't have any to process atm
            if self._interaction.audio_input_queue.empty() and not is_final_message:
                await asyncio.sleep(0.005)
                continue

            # Get audio from the queue
            should_finish_task = False
            try:
                message: OjinAvatarInteractionInputMessage = (
                    self._interaction.audio_input_queue.get_nowait()
                )
            except asyncio.QueueEmpty:
                logger.error(f"Audio queue empty! state = {self._interaction.state} is_final_message = {is_final_message}")
                await asyncio.sleep(0.05)
                continue

            should_finish_task = True 
            if is_final_message:
                self._interaction.set_state(InteractionState.WAITING_FOR_LAST_FRAME)
                message.is_last_input = True

            logger.debug(
                f"Sending audio int16: {len(message.audio_int16_bytes)} is_final: {message.is_last_input}"
            )
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
                sample_rate=OJIN_AVATAR_SAMPLE_RATE,
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


class AvatarPlaybackLoop:
    """Manages a complete idle animation loop with synchronized audio and video."""

    id: int = 0
    duration: int = 0  # seconds
    frames: list[AnimationKeyframe] = []  # Keyframes of the idle animation
    playback_time: float = 0  # Total elapsed playback time in seconds

    def __init__(
        self,
        duration: int,
        fps: int = 25,
    ):
        """Initialize the avatar idle loop animation.

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
            mirror_frame_idx=mirror_index(frame_idx, expected_frames),
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

        mirror_frame_idx = mirror_index(current_frame_idx, len(self.frames))

        return self.frames[mirror_frame_idx]

    def get_current_frame(self) -> AnimationKeyframe | None:
        """Get the keyframe at the current playback time.

        Returns:
            AnimationKeyframe | None: The current keyframe or None if no keyframes exist

        """
        return self.get_frame_at_time(self.playback_time)

    def get_current_frame_idx(self) -> int:
        """Get the keyframe at the current playback time.

        Returns:
            AnimationKeyframe | None: The current keyframe or None if no keyframes exist

        """
        current_frame_idx = math.floor(self.playback_time * self.fps)

        mirror_frame_idx = mirror_index(current_frame_idx, len(self.frames))
        return mirror_frame_idx

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


def mirror_index(index: int, size: int) -> int:
    """Calculate a mirrored index for creating a ping-pong animation effect.

    This method maps a continuously increasing index to a back-and-forth pattern
    within the given size, creating a ping-pong effect for smooth looping animations.

    Args:
        index (int): The original frame index
        size (int): The number of available frames

    Returns:
        int: The mirrored index that creates the ping-pong effect

    """
    # Calculate period length (going up and down)
    period = (size - 1) * 2

    # Get position within one period
    normalized_idx = index % period

    # If in first half, return the index directly
    if normalized_idx < size:
        return normalized_idx
    else:
        # If in second half, return the mirrored index
        return period - normalized_idx
