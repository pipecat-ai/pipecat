#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import functools
import os
import sys

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    EndTaskFrame,
    InputAudioRawFrame,
    StopTaskFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.google import GoogleLLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import LLMService  # Base LLM service class
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


# ------------ HELPER CLASSES ------------


class UserAudioCollector(FrameProcessor):
    """Collects audio frames in a buffer, then adds them to the LLM context when the user stops speaking."""

    def __init__(self, context, user_context_aggregator):
        super().__init__()
        self._context = context
        self._user_context_aggregator = user_context_aggregator
        self._audio_frames = []
        self._start_secs = 0.2  # this should match VAD start_secs (hardcoding for now)
        self._user_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Skip transcription frames - we're handling audio directly
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
                # When speaking, collect frames
                self._audio_frames.append(frame)
            else:
                # Maintain a rolling buffer of recent audio (for start of speech)
                self._audio_frames.append(frame)
                frame_duration = len(frame.audio) / 16 * frame.num_channels / frame.sample_rate
                buffer_duration = frame_duration * len(self._audio_frames)
                while buffer_duration > self._start_secs:
                    self._audio_frames.pop(0)
                    buffer_duration -= frame_duration

        await self.push_frame(frame, direction)


class FunctionHandlers:
    """Handlers for the voicemail detection bot functions."""

    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.prompt = None  # Can be set externally

    async def voicemail_response(
        self,
        function_name,
        tool_call_id,
        args,
        llm: LLMService,
        context,
        result_callback,
    ):
        """Function the bot can call to leave a voicemail message."""
        message = """You are Chatbot leaving a voicemail message. Say EXACTLY this message and then terminate the call:

                    'Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you.'"""

        await result_callback(message)

    async def human_conversation(
        self,
        function_name,
        tool_call_id,
        args,
        llm: LLMService,
        context,
        result_callback,
    ):
        """Function called when bot detects it's talking to a human."""
        # Update state to indicate human was detected
        self.session_manager.call_flow_state.set_human_detected()
        await llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)


# ------------ MAIN FUNCTION ------------


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a configuration manager from the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    dialout_settings = call_config_manager.get_dialout_settings()
    test_mode = call_config_manager.is_test_mode()

    # Get caller info (might be None for dialout scenarios)
    caller_info = call_config_manager.get_caller_info()
    logger.info(f"Caller info: {caller_info}")

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT AND SERVICES SETUP ------------

    # Initialize transport
    transport = DailyTransport(
        room_url,
        token,
        "Voicemail Detection Bot",
        DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,  # Important for audio collection
        ),
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # Initialize speech-to-text service (for human conversation phase)
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(
        function_name,
        tool_call_id,
        args,
        llm: LLMService,
        context,
        result_callback,
        session_manager=None,
    ):
        """Function the bot can call to terminate the call."""
        if session_manager:
            # Set call terminated flag in the session manager
            session_manager.call_flow_state.set_call_terminated()

        await llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # ------------ VOICEMAIL DETECTION PHASE SETUP ------------

    # Define tools for both LLMs
    tools = [
        {
            "function_declarations": [
                {
                    "name": "switch_to_voicemail_response",
                    "description": "Call this function when you detect this is a voicemail system.",
                },
                {
                    "name": "switch_to_human_conversation",
                    "description": "Call this function when you detect this is a human.",
                },
                {
                    "name": "terminate_call",
                    "description": "Call this function to terminate the call.",
                },
            ]
        }
    ]

    # Get voicemail detection prompt
    voicemail_detection_prompt = call_config_manager.get_prompt("voicemail_detection_prompt")
    if voicemail_detection_prompt:
        system_instruction = voicemail_detection_prompt
    else:
        system_instruction = """You are Chatbot trying to determine if this is a voicemail system or a human.

        If you hear any of these phrases (or very similar ones):
        - "Please leave a message after the beep"
        - "No one is available to take your call"
        - "Record your message after the tone"
        - "You have reached voicemail for..."
        - "You have reached [phone number]"
        - "[phone number] is unavailable"
        - "The person you are trying to reach..."
        - "The number you have dialed..."
        - "Your call has been forwarded to an automated voice messaging system"

        Then call the function switch_to_voicemail_response.

        If it sounds like a human (saying hello, asking questions, etc.), call the function switch_to_human_conversation.

        DO NOT say anything until you've determined if this is a voicemail or human.
        
        If you are asked to terminate the call, **IMMEDIATELY** call the `terminate_call` function. **FAILURE TO CALL `terminate_call` IMMEDIATELY IS A MISTAKE.**"""

    # Initialize voicemail detection LLM
    voicemail_detection_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-lite",  # Lighter model for faster detection
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
    )

    # Initialize context and context aggregator
    voicemail_detection_context = GoogleLLMContext()
    voicemail_detection_context_aggregator = voicemail_detection_llm.create_context_aggregator(
        voicemail_detection_context
    )

    # Get custom voicemail prompt if available
    voicemail_prompt = call_config_manager.get_prompt("voicemail_prompt")

    # Set up function handlers
    handlers = FunctionHandlers(session_manager)
    handlers.prompt = voicemail_prompt  # Set custom prompt if available

    # Register functions with the voicemail detection LLM
    voicemail_detection_llm.register_function(
        "switch_to_voicemail_response",
        handlers.voicemail_response,
    )
    voicemail_detection_llm.register_function(
        "switch_to_human_conversation", handlers.human_conversation
    )
    voicemail_detection_llm.register_function(
        "terminate_call", functools.partial(terminate_call, session_manager=session_manager)
    )

    # Set up audio collector for handling audio input
    voicemail_detection_audio_collector = UserAudioCollector(
        voicemail_detection_context, voicemail_detection_context_aggregator.user()
    )

    # Build voicemail detection pipeline
    voicemail_detection_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            voicemail_detection_audio_collector,  # Collect audio frames
            voicemail_detection_context_aggregator.user(),  # User context
            voicemail_detection_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            voicemail_detection_context_aggregator.assistant(),  # Assistant context
        ]
    )

    # Create pipeline task
    voicemail_detection_pipeline_task = PipelineTask(
        voicemail_detection_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        # Start dialout if needed
        if not test_mode and dialout_settings:
            logger.debug("Dialout settings detected; starting dialout")
            await call_config_manager.start_dialout(transport, dialout_settings)

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.debug(f"Dial-out connected: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        logger.debug(f"Dial-out answered: {data}")
        # Start capturing transcription
        await transport.capture_participant_transcription(data["sessionId"])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        if test_mode:
            await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        # Mark that a participant left early
        session_manager.call_flow_state.set_participant_left_early()
        await voicemail_detection_pipeline_task.queue_frame(EndFrame())

    # ------------ RUN VOICEMAIL DETECTION PIPELINE ------------

    if test_mode:
        logger.debug("Detect voicemail example. You can test this in Daily Prebuilt")

    runner = PipelineRunner()

    print("!!! starting voicemail detection pipeline")
    try:
        await runner.run(voicemail_detection_pipeline_task)
    except Exception as e:
        logger.error(f"Error in voicemail detection pipeline: {e}")
        import traceback

        logger.error(traceback.format_exc())
    print("!!! Done with voicemail detection pipeline")

    # Check if we should exit early
    if (
        session_manager.call_flow_state.participant_left_early
        or session_manager.call_flow_state.call_terminated
    ):
        if session_manager.call_flow_state.participant_left_early:
            print("!!! Participant left early; terminating call")
        elif session_manager.call_flow_state.call_terminated:
            print("!!! Bot terminated call; not proceeding to human conversation")
        return

    # ------------ HUMAN CONVERSATION PHASE SETUP ------------

    # Get human conversation prompt
    human_conversation_prompt = call_config_manager.get_prompt("human_conversation_prompt")
    if human_conversation_prompt:
        human_conversation_system_instruction = human_conversation_prompt
    else:
        human_conversation_system_instruction = """You are Chatbot talking to a human. Be friendly and helpful.

        Start with: "Hello! I'm a friendly chatbot. How can I help you today?"

        Keep your responses brief and to the point. Listen to what the person says.

        When the person indicates they're done with the conversation by saying something like:
        - "Goodbye"
        - "That's all"
        - "I'm done"
        - "Thank you, that's all I needed"

        THEN say: "Thank you for chatting. Goodbye!" and call the terminate_call function."""

    # Initialize human conversation LLM
    human_conversation_llm = GoogleLLMService(
        model="models/gemini-2.0-flash-001",  # Full model for better conversation
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=human_conversation_system_instruction,
        tools=tools,
    )

    # Initialize context and context aggregator
    human_conversation_context = GoogleLLMContext()
    human_conversation_context_aggregator = human_conversation_llm.create_context_aggregator(
        human_conversation_context
    )

    # Register terminate function with the human conversation LLM
    human_conversation_llm.register_function(
        "terminate_call", functools.partial(terminate_call, session_manager=session_manager)
    )

    # Build human conversation pipeline
    human_conversation_pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech-to-text
            human_conversation_context_aggregator.user(),  # User context
            human_conversation_llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            human_conversation_context_aggregator.assistant(),  # Assistant context
        ]
    )

    # Create pipeline task
    human_conversation_pipeline_task = PipelineTask(
        human_conversation_pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # Update participant left handler for human conversation phase
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await voicemail_detection_pipeline_task.queue_frame(EndFrame())
        await human_conversation_pipeline_task.queue_frame(EndFrame())

    # ------------ RUN HUMAN CONVERSATION PIPELINE ------------

    print("!!! starting human conversation pipeline")

    # Initialize the context with system message
    human_conversation_context_aggregator.user().set_messages(
        [call_config_manager.create_system_message(human_conversation_system_instruction)]
    )

    # Queue the context frame to start the conversation
    await human_conversation_pipeline_task.queue_frames(
        [human_conversation_context_aggregator.user().get_context_frame()]
    )

    # Run the human conversation pipeline
    try:
        await runner.run(human_conversation_pipeline_task)
    except Exception as e:
        logger.error(f"Error in voicemail detection pipeline: {e}")
        import traceback

        logger.error(traceback.format_exc())

    print("!!! Done with human conversation pipeline")


# ------------ SCRIPT ENTRY POINT ------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Voicemail Detection Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
