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

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, EndTaskFrame, StopTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection
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
    llm = GoogleLLMService(
        model="models/gemini-2.0-flash-001",  # Full model for better conversation        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=system_instruction,
        tools=tools,
    )

    # Initialize context and context aggregator
    context = GoogleLLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    # Set up function handlers
    call_flow_state = CallFlowState()
    handlers = FunctionHandlers(call_flow_state)

    # Register functions with the voicemail detection LLM
    llm.register_function(
        "switch_to_voicemail_response",
        handlers.voicemail_response,
    )
    llm.register_function("switch_to_human_conversation", handlers.human_conversation)
    llm.register_function("terminate_call", lambda params: terminate_call(params, call_flow_state))

    # Build voicemail detection pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User context
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant context
        ]
    )

    # Create pipeline task
    pipeline_task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        # Start dialout with conditional caller_id
        logger.debug(f"Dialout settings detected; starting dialout to number: {phone_number}")

        # Build dialout parameters conditionally
        dialout_params = {"phoneNumber": phone_number}
        if caller_id:
            dialout_params["callerId"] = caller_id
            logger.debug(f"Including caller ID in dialout: {caller_id}")

        logger.debug(f"Dialout parameters: {dialout_params}")
        await transport.start_dialout(dialout_params)

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.debug(f"Dial-out connected: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        logger.debug(f"Dial-out answered: {data}")
        # Automatically start capturing transcription for the participant
        await transport.capture_participant_transcription(data["sessionId"])
        # The bot will wait to hear the user before the bot speaks

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        # Mark that a participant left early
        call_flow_state.set_participant_left_early()
        await pipeline_task.queue_frame(EndFrame())

    # ------------ RUN VOICEMAIL DETECTION PIPELINE ------------

    runner = PipelineRunner()

    print("!!! starting voicemail detection pipeline")
    try:
        await runner.run(pipeline_task)
    except Exception as e:
        logger.error(f"Error in voicemail detection pipeline: {e}")
        import traceback

        logger.error(traceback.format_exc())
    print("!!! Done with voicemail detection pipeline")


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
