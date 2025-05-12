#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
import time

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport
from pipecat.processors.user_idle_processor import UserIdleProcessor
# from pipecat.frames.frames import LLMMessagesFrame, TTSSpeakFrame
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndTaskFrame,
    Frame,
    LLMMessagesFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    EndFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.filters.function_filter import FunctionFilter

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")



class TranscriptionModifierProcessor(FrameProcessor):
    """Processor that modifies transcription frames before they reach the context aggregator."""

    def __init__(self, operator_session_id_ref):
        """Initialize with a reference to the operator_session_id variable.

        Args:
            operator_session_id_ref: A reference or container holding the operator's session ID
        """
        super().__init__()
        self.operator_session_id_ref = operator_session_id_ref

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only process frames that are moving downstream
        if direction == FrameDirection.DOWNSTREAM:
            # Check if the frame is a transcription frame
            if isinstance(frame, TranscriptionFrame):
                # Check if this frame is from the operator
                if (
                    self.operator_session_id_ref[0] is not None
                    and hasattr(frame, "user_id")
                    and frame.user_id == self.operator_session_id_ref[0]
                ):
                    # Modify the text to include operator prefix
                    frame.text = f"[OPERATOR]: {frame.text}"
                    logger.debug(f"++++ Modified Operator Transcription: {frame.text}")

        # Push the (potentially modified) frame downstream
        await self.push_frame(frame, direction)

async def main(
    room_url: str,
    token: str,
    body: dict,
):

    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    num_idle_events = 0

    user_idled_too_much = False


    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
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
        "Simple Dial-in Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(  # Pipeline task reference
        params: FunctionCallParams
                              ):
        """Function the bot can call to terminate the call upon completion of a voicemail message."""
        content = f"""The user wants to end the call. The user has been silent for {UserIdleProcessor.retry_count} events"""
        message = call_config_manager.create_system_message(content)
        messages.append(message)
        await task.queue_frames([LLMMessagesFrame(messages)])


        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. 
    Your output will be converted to audio so don't include special characters in your answers. 
    Respond to what the user said in a creative and helpful way, but keep your responses brief. 

    When the person indicates they're done with the conversation by saying something like:
        - "Goodbye"
        - "That's all"
        - "I'm done"
        - "Thank you, that's all I needed"
        
     THEN say: "Thank you for chatting. Goodbye!" and call the terminate_call function.
     
    Start by saying how can I help you?
    """

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    async def detect_user_idle(user_idle: UserIdleProcessor, retry_count: int) -> bool:
        if retry_count == 1:
            # First attempt: Add a gentle prompt to the conversation
            messages.append(
                {
                    "role": "system",
                    "content": "The user has been quiet. Politely and briefly ask if they're still there.",
                }
            )
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        elif retry_count == 2:
            # Second attempt: More direct prompt
            messages.append(
                {
                    "role": "system",
                    "content": "The user is still inactive. Ask if they'd like to continue our conversation.",
                }
            )
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        elif retry_count == 3:
            # Third attempt: End the conversation
            await user_idle.push_frame(
                TTSSpeakFrame("It seems like you're busy right now. Have a nice day!")
            )
            await user_idle.push_frame(LLMMessagesFrame(messages))
            return True
        elif retry_count == 4:
            content = f"""User has been silent after {retry_count-1} prompts, ending conversation."""
            logger.info(content)
            await user_idle.push_frame(
                TTSSpeakFrame(content)
            )
            return True


    user_idle = UserIdleProcessor(callback=detect_user_idle, timeout=3.0)

    transcription_modifier = TranscriptionModifierProcessor(
        session_manager.get_session_id_ref("operator")
    )

    # Define function to determine if bot should speak
    # async def should_speak(self) -> bool:
    #     result = (
    #             not user_idled_too_much #and not session_manager.call_flow_state.summary_finished
    #     )
    #     return result

    # ------------ PIPELINE SETUP ------------

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            user_idle,  # Idle user check-in
            context_aggregator.user(),  # User responses
            # FunctionFilter(should_speak),
            llm,  # LLM
            tts,  # TTS
            transport.output()  # Transport bot output
            # context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")

        await task.cancel()

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)

# class SummaryFinished(FrameProcessor):
#     """Frame processor that monitors when summary has been finished."""
#
#     def __init__(self, dial_operator_state):
#         super().__init__()
#         # Store reference to the shared state object
#         self.dial_operator_state = dial_operator_state
#
#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#
#         # Check if operator is connected and this is the end of bot speaking
#         if self.dial_operator_state.operator_connected and isinstance(
#             frame, BotStoppedSpeakingFrame
#         ):
#             logger.debug("Summary finished, bot will stop speaking")
#             self.dial_operator_state.set_summary_finished()
#
#         await self.push_frame(frame, direction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
