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
from datetime import datetime, timedelta

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    LLMMessagesFrame,
    TTSSpeakFrame,
    MetricsFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import (
    TTFBMetricsData,
    ProcessingMetricsData,
    LLMUsageMetricsData,
    TTSUsageMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


class CallStatistics:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.silence_events = 0
        self.user_messages = 0
        self.bot_messages = 0
        self.llm_tokens_used = 0
        self.tts_characters_used = 0

    def record_silence_event(self):
        self.silence_events += 1

    def record_user_message(self):
        self.user_messages += 1

    def record_bot_message(self):
        self.bot_messages += 1

    def record_llm_usage(self, tokens):
        self.llm_tokens_used += tokens

    def record_tts_usage(self, characters):
        self.tts_characters_used += characters

    def end_call(self):
        self.end_time = time.time()

    def get_duration_seconds(self):
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def log_summary(self):
        duration = self.get_duration_seconds()
        duration_str = str(timedelta(seconds=int(duration)))

        logger.info("===== CALL SUMMARY =====")
        logger.info(f"Call Duration: {duration_str}")
        logger.info(f"Silence Events: {self.silence_events}")
        logger.info(f"User Messages: {self.user_messages}")
        logger.info(f"Bot Messages: {self.bot_messages}")
        logger.info(f"LLM Tokens Used: {self.llm_tokens_used}")
        logger.info(f"TTS Characters Used: {self.tts_characters_used}")
        logger.info("=======================")


class MetricsProcessor(FrameProcessor):
    def __init__(self, call_stats):
        super().__init__()
        self.call_stats = call_stats

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, LLMUsageMetricsData):
                    tokens = d.value
                    self.call_stats.record_llm_usage(tokens.total_tokens)
                elif isinstance(d, TTSUsageMetricsData):
                    self.call_stats.record_tts_usage(d.value)

        # Track user and bot messages
        if isinstance(frame, TranscriptionFrame) and frame.final:
            self.call_stats.record_user_message()
        elif isinstance(frame, TTSSpeakFrame):
            self.call_stats.record_bot_message()

        await self.push_frame(frame, direction)


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Initialize call statistics
    call_stats = CallStatistics()

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

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

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call upon completion of a voicemail message."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Then end the call
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
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. If the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function. """

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    # Create a user idle processor to detect silence
    async def handle_user_idle(processor, retry_count):
        # Record silence event
        call_stats.record_silence_event()

        if retry_count <= 2:  # First and second attempts (retry_count starts at 1)
            messages = [
                "I notice you've been quiet for a while. Is there anything I can help you with?",
                "I haven't heard from you. Are you still there?",
                "Since I haven't heard from you in a while, I'll be ending the call soon.",
            ]
            await pipeline.push_frame(
                TTSSpeakFrame(
                    text=messages[retry_count - 1],
                    interrupt=False,
                )
            )
            return True  # Continue monitoring for idle events
        else:  # Third attempt (retry_count = 3)
            # Final message before terminating
            await pipeline.push_frame(
                TTSSpeakFrame(
                    text="I'll be ending our call now. Feel free to call back if you need assistance later.",
                    interrupt=False,
                )
            )
            # Wait for the message to be spoken before terminating
            await asyncio.sleep(5)

            # Mark that the call was terminated by the bot
            if session_manager:
                session_manager.call_flow_state.set_call_terminated()

            # End the call
            await task.queue_frame(EndTaskFrame())
            # End call statistics and log summary
            call_stats.end_call()
            call_stats.log_summary()
            return False  # Stop monitoring for idle events

    user_idle_processor = UserIdleProcessor(
        callback=handle_user_idle,
        timeout=10.0,  # Trigger after 10 seconds of silence
    )

    # Create metrics processor
    metrics_processor = MetricsProcessor(call_stats)

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
            user_idle_processor,  # Add the user idle processor to detect silence
            metrics_processor,  # Add metrics processor
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        # End call statistics and log summary
        call_stats.end_call()
        call_stats.log_summary()
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)


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
