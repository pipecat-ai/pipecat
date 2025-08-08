#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import collections
import json
import os
import sys
import time
from typing import Deque

from dotenv import load_dotenv
from loguru import logger
from pipecat_flows import (
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowManager,
    FlowResult,
    FlowsFunctionSchema,
    NodeConfig,
)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotInterruptionFrame,
    CancelFrame,
    EndFrame,
    EndTaskFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    LLMMessagesFrame,
    StartFrame,
    StartInterruptionFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMContext, GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

use_prebuilt = True

# Simple constants for model states
VOICEMAIL_MODE = "voicemail"
HUMAN_MODE = "human"
MUTE_MODE = "mute"

VOICEMAIL_CONFIDENCE_THRESHOLD = 0.6
HUMAN_CONFIDENCE_THRESHOLD = 0.6


def warm():
    """
    Warm up function to ensure the bot is ready to handle requests.
    This function can be called periodically to keep the bot warm.
    """
    logger.info("Warming up the bot...")
    # Perform any necessary warm-up tasks here
    # For example, you can load models, initialize connections, etc.
    # This is just a placeholder for demonstration purposes
    pass


# ------------ FLOW MANAGER SETUP ------------


# ------------ PIPECAT FLOWS FOR HUMAN CONVERSATION ------------


# Type definitions for flows
class GreetingResult(FlowResult):
    greeting_complete: bool


class ConversationResult(FlowResult):
    message: str


class EndConversationResult(FlowResult):
    status: str


# Flow function handlers - updated to return (result, next_node) tuple
async def handle_greeting(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[GreetingResult, NodeConfig]:
    """Handle initial greeting to human."""
    logger.debug("handle_greeting executing")

    result = GreetingResult(greeting_complete=True)

    # Return the next node config directly instead of using set_node
    next_node = create_conversation_node()
    return result, next_node


async def handle_conversation(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[ConversationResult, NodeConfig]:
    """Handle ongoing conversation with human."""
    message = args.get("message", "")
    logger.debug(f"handle_conversation executing with message: {message}")

    result = ConversationResult(message=message)

    # Return the same conversation node to continue the chat
    next_node = create_conversation_node()
    return result, next_node


async def handle_end_conversation(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[EndConversationResult, NodeConfig]:
    """Handle ending the conversation."""
    logger.debug("handle_end_conversation executing")

    result = EndConversationResult(status="completed")

    # Return the end node config directly
    next_node = create_end_node()
    return result, next_node


# Node configurations for human conversation flow
def create_greeting_node() -> NodeConfig:
    """Create the initial greeting node for human conversation."""
    return {
        "name": "greeting",
        "role_messages": [
            {
                "role": "system",
                "content": """You are a friendly chatbot. Your responses will be
                    converted to audio, so avoid special characters.
                    Be conversational and helpful. The user will have just replied to your greeting and question asking if they are Tim.
                    If they say yes, proceed to the conversation node.""",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": """Decide if the user is Tim based on their response.
                    If they say yes, call handle_greeting to proceed to the conversation.""",
            }
        ],
        "respond_immediately": False,
        "functions": [
            FlowsFunctionSchema(
                name="handle_greeting",
                description="Mark that greeting is complete and proceed to conversation",
                properties={},
                required=[],
                handler=handle_greeting,
            )
        ],
    }


def create_conversation_node() -> NodeConfig:
    """Create the main conversation node."""
    return {
        "name": "conversation",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "You are having a friendly conversation with a human. "
                    "Listen to what they say and respond helpfully. "
                    "Keep your responses brief and conversational. "
                    "If they indicate they want to end the conversation (saying goodbye, "
                    "thanks, that's all, etc.), call handle_end_conversation. "
                    "Otherwise, use handle_conversation to continue the chat."
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="handle_conversation",
                description="Continue the conversation with the human",
                properties={"message": {"type": "string", "description": "The response message"}},
                required=["message"],
                handler=handle_conversation,
            ),
            FlowsFunctionSchema(
                name="handle_end_conversation",
                description="End the conversation when the human is ready to finish",
                properties={},
                required=[],
                handler=handle_end_conversation,
            ),
        ],
    }


def create_end_node() -> NodeConfig:
    """Create the final conversation end node."""
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "Thank the person for the conversation and say goodbye. "
                    "Keep it brief and friendly."
                ),
            }
        ],
        "functions": [],  # Required by FlowManager, even if empty
        "post_actions": [{"type": "end_conversation"}],
    }


# ------------ SIMPLIFIED CLASSES ------------


class DebugClass(FrameProcessor):
    """A simple debug class to log frames."""

    def __init__(self):
        super().__init__()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        logger.debug(f"DebugClass received frame: {frame} in direction: {direction}")
        await self.push_frame(frame, direction)


class VoicemailDetectionObserver(BaseObserver):
    """Observes voicemail speaking patterns to know when voicemail is done."""

    def __init__(self, timeout: float = 5.0):
        super().__init__()
        self._processed_frames = set()
        self._timeout = timeout
        self._last_turn_time = 0
        self._voicemail_speaking = False

    async def on_push_frame(self, data: FramePushed):
        if data.frame.id in self._processed_frames:
            return
        self._processed_frames.add(data.frame.id)

        if isinstance(data.frame, UserStartedSpeakingFrame):
            self._voicemail_speaking = True
            self._last_turn_time = 0
        elif isinstance(data.frame, UserStoppedSpeakingFrame):
            self._last_turn_time = time.time()

    async def wait_for_voicemail(self):
        """Wait for voicemail to finish speaking."""
        while self._voicemail_speaking:
            logger.debug("📩️ Waiting for voicemail to finish")
            if self._last_turn_time:
                diff_time = time.time() - self._last_turn_time
                self._voicemail_speaking = diff_time < self._timeout
            if self._voicemail_speaking:
                await asyncio.sleep(0.5)


class VADPrebufferProcessor(FrameProcessor):
    """
    This processor buffers a specified number of audio frames before speech is
    detected.

    When a VADUserStartedSpeakingFrame is received, it first replays the
    buffered audio frames in the correct order, ensuring that the very
    beginning of the user's speech is not missed. After replaying the buffer,
    all subsequent frames are passed through immediately.

    This is useful for preventing the initial part of a user's utterance from
    being cut off by the Voice Activity Detection (VAD).

    Args:
        prebuffer_frame_count (int): The number of InputAudioRawFrames to buffer before speech.
                                     Defaults to 33.
        direction (FrameDirection): The direction of frames to process (UPSTREAM or DOWNSTREAM).
                                    Defaults to DOWNSTREAM.
    """

    def __init__(
        self,
        prebuffer_frame_count: int = 33,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        super().__init__()
        self._direction = direction
        self._speech_started = False
        self._prebuffer_frame_count = prebuffer_frame_count

        # A deque with a maxlen is a highly efficient fixed-size buffer.
        # When it's full, adding a new item automatically discards the oldest item.
        self._audio_buffer: Deque[InputAudioRawFrame] = collections.deque(
            maxlen=prebuffer_frame_count
        )

    def _should_passthrough_frame(self, frame: Frame, direction: FrameDirection) -> bool:
        """Determines if a frame should bypass the buffering logic entirely."""
        return isinstance(frame, (SystemFrame, EndFrame)) or direction != self._direction

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Let system/end frames and frames in the wrong direction pass through immediately.
        if self._should_passthrough_frame(frame, direction):
            await self.push_frame(frame, direction)
            return

        # If speech has already started, the gate is open. Let all frames through.
        if self._speech_started:
            await self.push_frame(frame, direction)
            return

        # --- Speech has NOT started yet ---

        # The VAD frame is the trigger to release the buffered audio.
        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug(
                f"Initial VAD Detected. Replaying {len(self._audio_buffer)} buffered audio frames."
            )
            # 1. Set the flag so all future frames pass through immediately.
            self._speech_started = True

            # 2. Push all the buffered audio frames downstream in order.
            for buffered_frame in self._audio_buffer:
                await self.push_frame(buffered_frame, direction)

            # 3. Clear the buffer now that it's been sent.
            self._audio_buffer.clear()

            # 4. Finally, push the VAD frame itself so downstream processors know speech has started.
            await self.push_frame(frame, direction)

        # If it's an audio frame, add it to our buffer. It won't be pushed downstream yet.
        elif isinstance(frame, InputAudioRawFrame):
            self._audio_buffer.append(frame)

        # Any other frames that arrive before speech (e.g., TextFrame) will be
        # ignored by this processor, as they don't match the conditions above.


class BlockAudioFrames(FrameProcessor):
    """Blocks audio frames from being processed further, conditionally based on mode."""

    def __init__(self, mode_checker, allowed_modes):
        super().__init__()
        self._mode_checker = mode_checker
        self._allowed_modes = allowed_modes if isinstance(allowed_modes, list) else [allowed_modes]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Block audio frames based on current mode
        if (
            isinstance(frame, InputAudioRawFrame)
            or isinstance(frame, UserStartedSpeakingFrame)
            or isinstance(frame, UserStoppedSpeakingFrame)
            or isinstance(frame, TranscriptionFrame)
        ):
            current_mode = self._mode_checker()
            # logger.debug(f"Current mode: {current_mode}, allowed modes: {self._allowed_modes}")
            if current_mode in self._allowed_modes:
                await self.push_frame(frame, direction)
            # If current mode is not in allowed modes, just return (block the frame)
            return

        # Pass all other frames through
        await self.push_frame(frame, direction)


class OutputGate(FrameProcessor):
    """Simple gate that opens when notified."""

    def __init__(self, notifier, start_open: bool = False):
        super().__init__()
        self._gate_open = start_open
        self._frames_buffer = []
        self._notifier = notifier
        self._gate_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Always pass system frames and function call frames
        if isinstance(
            frame,
            (
                SystemFrame,
                EndFrame,
                FunctionCallInProgressFrame,
                FunctionCallResultFrame,
            ),
        ):
            if isinstance(frame, StartFrame):
                await self._start()
            elif isinstance(frame, (CancelFrame, EndFrame)):
                await self._stop()
            elif isinstance(frame, StartInterruptionFrame):
                self._frames_buffer = []
                self._gate_open = False
            await self.push_frame(frame, direction)
            return

        # Only gate downstream frames
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if self._gate_open:
            await self.push_frame(frame, direction)
        else:
            # Buffer frames until gate opens
            self._frames_buffer.append((frame, direction))

    async def _start(self):
        self._frames_buffer = []
        if not self._gate_task:
            self._gate_task = self.create_task(self._gate_task_handler())

    async def _stop(self):
        if self._gate_task:
            await self.cancel_task(self._gate_task)
            self._gate_task = None

    async def _gate_task_handler(self):
        """Wait for notification to open gate."""
        while True:
            try:
                await self._notifier.wait()
                self._gate_open = True
                # Flush buffered frames
                for frame, direction in self._frames_buffer:
                    await self.push_frame(frame, direction)
                self._frames_buffer = []
                break  # Gate stays open
            except asyncio.CancelledError:
                break


class UserAudioCollector(FrameProcessor):
    """Collects audio frames for the LLM context."""

    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = 0.2
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            return
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._context.add_audio_frames_message(audio_frames=self._audio_frames)
            await self._user_context_aggregator.push_frame(
                self._user_context_aggregator.get_context_frame()
            )
        elif isinstance(frame, InputAudioRawFrame):
            if self._user_speaking:
                self._audio_frames.append(frame)
            else:
                # Maintain rolling buffer
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


# ------------ MAIN FUNCTION ------------


async def run_bot(room_url: str, token: str, body: dict) -> None:
    """Run the voice bot with parallel pipeline architecture."""

    # ------------ SETUP ------------
    logger.info(f"Starting bot with room: {room_url}")

    body_data = json.loads(body)
    dialout_settings = body_data["dialout_settings"]
    phone_number = dialout_settings["phone_number"]
    caller_id = dialout_settings.get("caller_id")

    # Simple state tracking
    current_mode = MUTE_MODE
    is_voicemail = False

    # Notifier for human conversation gate
    human_notifier = EventNotifier()

    # Observer for voicemail detection
    voicemail_observer = VoicemailDetectionObserver()

    # ------------ FUNCTION HANDLERS ------------

    async def voicemail_detected(params: FunctionCallParams):
        nonlocal current_mode, is_voicemail

        confidence = params.arguments["confidence"]
        reasoning = params.arguments["reasoning"]

        logger.info(f"Voicemail detected - confidence: {confidence}, reasoning: {reasoning}")

        if confidence >= VOICEMAIL_CONFIDENCE_THRESHOLD and current_mode == MUTE_MODE:
            logger.info(f"🔄 MODE CHANGE: {current_mode} -> {VOICEMAIL_MODE}")
            current_mode = VOICEMAIL_MODE
            is_voicemail = True

            await voicemail_observer.wait_for_voicemail()

            # Generate voicemail message
            message = "Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you."
            logger.info(f"🎤 SENDING VOICEMAIL MESSAGE: {message}")
            await voicemail_tts.queue_frame(TTSSpeakFrame(text=message))
            await voicemail_tts.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

        await params.result_callback({"confidence": f"{confidence}", "reasoning": reasoning})

    async def human_detected(params: FunctionCallParams):
        nonlocal current_mode, is_voicemail

        confidence = params.arguments["confidence"]
        reasoning = params.arguments["reasoning"]

        logger.info(f"Human detected - confidence: {confidence}, reasoning: {reasoning}")

        if confidence >= HUMAN_CONFIDENCE_THRESHOLD and current_mode == MUTE_MODE:
            logger.info(f"🔄 MODE CHANGE: {current_mode} -> {HUMAN_MODE}")
            current_mode = HUMAN_MODE
            is_voicemail = False

            await human_notifier.notify()
            message = "Hello, this is virtual agent John. Am I speaking to Tim?"
            logger.info(f"🎤 SENDING HUMAN MESSAGE: {message}")
            await voicemail_tts.queue_frame(TTSSpeakFrame(text=message))

        await params.result_callback({"confidence": f"{confidence}", "reasoning": reasoning})

    # async def terminate_call(params: FunctionCallParams):
    #     logger.info("Terminating call")
    #     await asyncio.sleep(3)  # Brief delay before termination
    #     await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
    #     await params.result_callback({"status": "call terminated"})

    # ------------ TRANSPORT & SERVICES ------------

    transport = DailyTransport(
        room_url,
        token,
        "Voicemail Detection Bot",
        DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16000,
                params=VADParams(start_secs=0.1, confidence=0.4, min_volume=0.4),
            ),
        ),
    )

    # TTS services
    voicemail_tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",
    )

    human_tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # ------------ LLM SETUP ------------

    detection_tools = [
        {
            "function_declarations": [
                {
                    "name": "voicemail_detected",
                    "description": "Signals that a voicemail greeting has been detected by the LLM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "The LLM's confidence score (ranging from 0.0 to 1.0) that a voicemail greeting was detected.",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "The LLM's textual explanation for why it believes a voicemail was detected, often citing specific phrases from the transcript.",
                            },
                        },
                        "required": ["confidence", "reasoning"],
                    },
                },
                {
                    "name": "human_detected",
                    "description": "Signals that a human attempting to communicate has been detected by the LLM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confidence": {
                                "type": "number",
                                "description": "The LLM's confidence score (ranging from 0.0 to 1.0) that a human conversation has been detected.",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "The LLM's textual explanation for why it believes a human communication was detected, often citing specific phrases from the transcript.",
                            },
                        },
                        "required": ["confidence", "reasoning"],
                    },
                },
            ]
        }
    ]

    detection_system_instructions = """
    You are an AI Call Analyzer. Your primary function is to determine if the initial audio from an incoming call is a voicemail system/answering machine or a live human attempting to engage in conversation.

    You will be provided with a transcript of the first few seconds of an audio interaction.

    Based on your analysis of this transcript, you MUST decide to call ONE of the following two functions:

    1.  voicemail_detected
        *   Call this function if the transcript strongly indicates a pre-recorded voicemail greeting, an answering machine message, or instructions to leave a message.
        *   Keywords and phrases to look for: "you've reached," "not available," "leave a message," "at the tone/beep," "sorry I missed your call," "please leave your name and number."
        *   Also consider if the speech sounds like a monologue without expecting an immediate response.
        *   Keep in mind that the beep noise from a typical pre-recorded voicemail greeting comes after the greeting and not before.

    2.  human_detected
        *   Call this function if the transcript indicates a human is present and actively trying to communicate or expecting an immediate response.
        *   Keywords and phrases to look for: "Hello?", "Hi," "[Company Name], how can I help you?", "Speaking.", or any direct question aimed at initiating a dialogue.
        *   Consider if the speech sounds like the beginning of a two-way conversation.

    **Decision Guidelines:**

    *   **Prioritize Human:** If there's ambiguity but a slight indication of a human trying to speak (e.g., a simple "Hello?" followed by a pause, which could be either), err on the side of `human_detected` to avoid missing a live interaction. Only call `voicemail_detected` if there are clear, strong indicators of a voicemail system.
    *   **Focus on Intent:** Is the speaker *delivering information* (likely voicemail) or *seeking interaction* (likely human)?
    *   **Brevity:** Voicemail greetings are often concise and formulaic. Human openings can be more varied."""

    detection_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-lite",
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=detection_system_instructions,
        tools=detection_tools,
    )

    human_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-001",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # ------------ CONTEXTS & FUNCTIONS ------------

    detection_context = GoogleLLMContext()
    detection_context_aggregator = detection_llm.create_context_aggregator(detection_context)

    human_context = GoogleLLMContext()
    human_context_aggregator = human_llm.create_context_aggregator(human_context)

    # Register functions
    detection_llm.register_function("voicemail_detected", voicemail_detected)
    detection_llm.register_function("human_detected", human_detected)

    # ------------ PROCESSORS ------------

    def get_current_mode():
        """Get the current conversation mode."""
        return current_mode

    audio_collector = UserAudioCollector(detection_context, detection_context_aggregator.user())
    voicemail_audio_blocker = BlockAudioFrames(get_current_mode, [VOICEMAIL_MODE, MUTE_MODE])
    human_audio_blocker = BlockAudioFrames(get_current_mode, [HUMAN_MODE])

    _VADPrebufferProcessor = VADPrebufferProcessor()

    # Filter functions
    async def voicemail_filter(frame) -> bool:
        result = current_mode == VOICEMAIL_MODE or current_mode == MUTE_MODE
        if hasattr(frame, "text") and frame.text:
            logger.debug(
                f"🎯 VOICEMAIL FILTER: mode={current_mode}, allowing={result}, frame={type(frame).__name__}"
            )
        return result

    async def human_filter(frame) -> bool:
        result = current_mode == HUMAN_MODE
        if hasattr(frame, "text") and frame.text:
            logger.debug(
                f"🎯 HUMAN FILTER: mode={current_mode}, allowing={result}, frame={type(frame).__name__}"
            )
        return result

    debug_processor = DebugClass()

    transcript = TranscriptProcessor()

    @transcript.event_handler("on_transcript_update")
    async def handle_update(processor, frame):
        for message in frame.messages:
            logger.info(f"📝 TRANSCRIPT {message.role}: {message.content}")

    # Add debug logging for TTS frames
    class TTSDebugProcessor(FrameProcessor):
        """Debug processor to track TTS frames."""

        def __init__(self, name):
            super().__init__()
            self._name = name

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            # Log all frame types for comprehensive debugging
            frame_type = type(frame).__name__
            if isinstance(frame, TTSTextFrame):
                logger.info(f"🔊 TTS DEBUG ({self._name}): {frame_type} - {frame.text}")
            await self.push_frame(frame, direction)

    voicemail_tts_debug = TTSDebugProcessor("VOICEMAIL")
    human_tts_debug = TTSDebugProcessor("HUMAN")

    # Debug processor to see what makes it past transport.output()
    post_transport_debug = TTSDebugProcessor("POST_TRANSPORT")

    # ------------ PIPELINE ------------

    pipeline = Pipeline(
        [
            transport.input(),
            ParallelPipeline(
                # Voicemail detection branch
                [
                    voicemail_audio_blocker,  # Allows audio at the start to detect voicemail, and while in voicemail mode. Is blocked when LLM detects human.
                    _VADPrebufferProcessor,
                    audio_collector,
                    detection_context_aggregator.user(),
                    detection_llm,
                    FunctionFilter(voicemail_filter),
                ],
                [
                    voicemail_tts,
                    transcript.assistant(),  # Capture voicemail TTS frames
                ],
                [
                    # Human conversation branch
                    human_audio_blocker,  # Allows audio when in human mode, blocks when voicemail is detected or when deciding if human or voicemail.
                    stt,
                    transcript.user(),  # Place after STT
                    human_context_aggregator.user(),
                    human_llm,
                    FunctionFilter(human_filter),
                    human_tts,
                    transcript.assistant(),  # Capture human TTS frame
                    human_context_aggregator.assistant(),
                ],
            ),
            transport.output(),
        ]
    )

    pipeline_task = PipelineTask(
        pipeline,
        idle_timeout_secs=90,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
        ),
        cancel_on_idle_timeout=False,
        observers=[voicemail_observer],
    )

    flow_manager = FlowManager(
        task=pipeline_task,
        tts=human_tts,
        llm=human_llm,
        context_aggregator=human_context_aggregator,
        transport=transport,
    )

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        await flow_manager.initialize(create_greeting_node())

        if not use_prebuilt:
            dialout_params = {"phoneNumber": phone_number}
            if caller_id:
                dialout_params["callerId"] = caller_id
            await transport.start_dialout(dialout_params)

    @transport.event_handler("on_participant_updated")
    async def on_participant_updated(transport, participant):
        logger.debug(f"Participant updated: {participant}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        logger.debug(f"Call answered: {data}")
        await transport.capture_participant_transcription(data["sessionId"])

    @transport.event_handler("on_dialout_error")
    async def on_dialout_error(transport, data):
        logger.error(f"Dialout error: {data}")
        await pipeline_task.queue_frame(EndFrame())

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await pipeline_task.queue_frame(EndFrame())

    # Remove the problematic on_pipeline_started handler
    # The context will be initialized naturally when frames flow through the pipeline

    # ------------ RUN ------------

    runner = PipelineRunner()
    logger.info("Starting simplified parallel pipeline bot")

    try:
        await runner.run(pipeline_task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback

        logger.error(traceback.format_exc())


# ------------ ENTRY POINT ------------


async def main():
    parser = argparse.ArgumentParser(description="Simplified Parallel Pipeline Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON config")

    args = parser.parse_args()
    if not all([args.url, args.token, args.body]):
        logger.error("All arguments required")
        parser.print_help()
        sys.exit(1)

    await run_bot(args.url, args.token, args.body)


if __name__ == "__main__":
    asyncio.run(main())
