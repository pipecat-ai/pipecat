#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import json
import os
import sys
from typing import Any

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
from pipecat.services.llm_service import FunctionCallParams
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


class CallFlowState:
    """State for tracking call flow operations and state transitions."""

    def __init__(self):
        # Voicemail detection state
        self.voicemail_detected = False
        self.human_detected = False

        # Call termination state
        self.call_terminated = False
        self.participant_left_early = False

    # Voicemail detection methods
    def set_voicemail_detected(self):
        """Mark that a voicemail system has been detected."""
        self.voicemail_detected = True
        self.human_detected = False

    def set_human_detected(self):
        """Mark that a human has been detected (not voicemail)."""
        self.human_detected = True
        self.voicemail_detected = False

    # Call termination methods
    def set_call_terminated(self):
        """Mark that the call has been terminated by the bot."""
        self.call_terminated = True

    def set_participant_left_early(self):
        """Mark that a participant left the call early."""
        self.participant_left_early = True


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

    def __init__(self, call_flow_state: CallFlowState):
        self.call_flow_state = call_flow_state

    async def voicemail_response(self, params: FunctionCallParams):
        """Function the bot can call to leave a voicemail message."""
        message = """You are Chatbot leaving a voicemail message. Say EXACTLY this message and then terminate the call:

                    'Hello, this is a message for Pipecat example user. This is Chatbot. Please call back on 123-456-7891. Thank you.'"""

        await params.result_callback(message)

    async def human_conversation(self, params: FunctionCallParams):
        """Function called when bot detects it's talking to a human."""
        # Update state to indicate human was detected
        self.call_flow_state.set_human_detected()
        await params.llm.push_frame(StopTaskFrame(), FrameDirection.UPSTREAM)


# ------------ MAIN FUNCTION ------------


async def run_bot(
    room_url: str,
    token: str,
    body: dict,
) -> None:
    """Run the voice bot with the given parameters.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        body: Body passed to the bot from the webhook

    """
    # ------------ CONFIGURATION AND SETUP ------------
    logger.info(f"Starting bot with room: {room_url}")
    logger.info(f"Token: {token}")
    logger.info(f"Body: {body}")
    # Parse the body to get the dial-in settings
    body_data = json.loads(body)

    # Check if the body contains dial-in settings
    logger.debug(f"Body data: {body_data}")

    if not body_data.get("dialout_settings"):
        logger.error("Dial-out settings not found in the body data")
        return

    dialout_settings = body_data["dialout_settings"]

    if not dialout_settings.get("phone_number"):
        logger.error("Dial-out phone number not found in the dial-out settings")
        return

    # Extract dial-out phone number
    phone_number = dialout_settings["phone_number"]
    caller_id = dialout_settings.get("caller_id")  # Use .get() to handle optional field

    if caller_id:
        logger.info(f"Dial-out caller ID specified: {caller_id}")
    else:
        logger.info("Dial-out caller ID not specified; proceeding without it")

    # ------------ TRANSPORT SETUP ------------

    transport_params = DailyParams(
        api_url=daily_api_url,
        api_key=daily_api_key,
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        vad_analyzer=SileroVADAnalyzer(),
        transcription_enabled=True,
    )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Voicemail Detection Bot",
        transport_params,
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
        params: FunctionCallParams,
        call_flow_state: CallFlowState = None,
    ):
        """Function the bot can call to terminate the call."""
        if call_flow_state:
            # Set call terminated flag in the session manager
            call_flow_state.set_call_terminated()

        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

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

    # Set up function handlers
    call_flow_state = CallFlowState()
    handlers = FunctionHandlers(call_flow_state)

    # Register functions with the voicemail detection LLM
    voicemail_detection_llm.register_function(
        "switch_to_voicemail_response",
        handlers.voicemail_response,
    )
    voicemail_detection_llm.register_function(
        "switch_to_human_conversation", handlers.human_conversation
    )
    voicemail_detection_llm.register_function(
        "terminate_call", lambda params: terminate_call(params, call_flow_state)
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
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ------------ RETRY LOGIC VARIABLES ------------
    max_retries = 5
    retry_count = 0
    dialout_successful = False

    # Build dialout parameters conditionally
    dialout_params = {"phoneNumber": phone_number}
    if caller_id:
        dialout_params["callerId"] = caller_id
        logger.debug(f"Including caller ID in dialout: {caller_id}")

    logger.debug(f"Dialout parameters: {dialout_params}")

    async def attempt_dialout():
        """Attempt to start dialout with retry logic."""
        nonlocal retry_count, dialout_successful

        if retry_count < max_retries and not dialout_successful:
            retry_count += 1
            logger.info(
                f"Attempting dialout (attempt {retry_count}/{max_retries}) to: {phone_number}"
            )
            await transport.start_dialout(dialout_params)
        else:
            logger.error(f"Maximum retry attempts ({max_retries}) reached. Giving up on dialout.")

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        # Start initial dialout attempt
        logger.debug(f"Dialout settings detected; starting dialout to number: {phone_number}")
        await attempt_dialout()

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.debug(f"Dial-out connected: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        nonlocal dialout_successful
        logger.debug(f"Dial-out answered: {data}")
        dialout_successful = True  # Mark as successful to stop retries
        # Automatically start capturing transcription for the participant
        await transport.capture_participant_transcription(data["sessionId"])
        # The bot will wait to hear the user before the bot speaks

    @transport.event_handler("on_dialout_error")
    async def on_dialout_error(transport, data: Any):
        logger.error(f"Dial-out error (attempt {retry_count}/{max_retries}): {data}")

        if retry_count < max_retries:
            logger.info(f"Retrying dialout")
            await attempt_dialout()
        else:
            logger.error(f"All {max_retries} dialout attempts failed. Stopping bot.")
            await voicemail_detection_pipeline_task.queue_frame(EndFrame())

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        # Mark that a participant left early
        call_flow_state.set_participant_left_early()
        await voicemail_detection_pipeline_task.queue_frame(EndFrame())

    # ------------ RUN VOICEMAIL DETECTION PIPELINE ------------

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
    if call_flow_state.participant_left_early or call_flow_state.call_terminated:
        if call_flow_state.participant_left_early:
            print("!!! Participant left early; terminating call")
        elif call_flow_state.call_terminated:
            print("!!! Bot terminated call; not proceeding to human conversation")
        return

    # ------------ HUMAN CONVERSATION PHASE SETUP ------------

    # Get human conversation prompt
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
        "terminate_call", lambda params: terminate_call(params, call_flow_state)
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
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
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
        [
            {
                "role": "system",
                "content": human_conversation_system_instruction,
            }
        ]
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


async def main():
    """Parse command line arguments and run the bot."""
    parser = argparse.ArgumentParser(description="Simple Dial-out Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    logger.debug(f"url: {args.url}")
    logger.debug(f"token: {args.token}")
    logger.debug(f"body: {args.body}")
    if not all([args.url, args.token, args.body]):
        logger.error("All arguments (-u, -t, -b) are required")
        parser.print_help()
        sys.exit(1)

    await run_bot(args.url, args.token, args.body)


if __name__ == "__main__":
    asyncio.run(main())
