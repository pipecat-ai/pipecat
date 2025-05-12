import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from typing import Optional

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

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

class CallStats:
    def __init__(self):
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.silence_events = 0
        self.unanswered_prompts = 0
        self.last_speech_time = datetime.now()
        self.total_duration = 0

    def end_call(self):
        self.end_time = datetime.now()
        self.total_duration = (self.end_time - self.start_time).total_seconds()

    def to_dict(self):
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration,
            "silence_events": self.silence_events,
            "unanswered_prompts": self.unanswered_prompts
        }

async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()
    test_mode = call_config_manager.is_test_mode()
    dialin_settings = call_config_manager.get_dialin_settings()
    session_manager = SessionManager()
    call_stats = CallStats()

    # ------------ TRANSPORT SETUP ------------
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

    transport = DailyTransport(
        room_url,
        token,
        "Enhanced Dial-in Bot",
        transport_params,
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",
    )

    # ------------ FUNCTION DEFINITIONS ------------
    async def terminate_call(params: FunctionCallParams):
        """Function to terminate the call and log call statistics."""
        if session_manager:
            session_manager.call_flow_state.set_call_terminated()
        
        call_stats.end_call()
        logger.info(f"Call statistics: {call_stats.to_dict()}")
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def handle_silence(params: FunctionCallParams):
        """Function to handle silence detection and prompt the user."""
        current_time = datetime.now()
        silence_duration = (current_time - call_stats.last_speech_time).total_seconds()
        
        if silence_duration >= 10:  # 10 seconds of silence
            call_stats.silence_events += 1
            call_stats.unanswered_prompts += 1
            
            if call_stats.unanswered_prompts >= 3:
                await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
            else:
                prompt = "I notice you've been quiet. Are you still there? Please let me know if you need anything."
                await params.llm.queue_frame(tts.synthesize(prompt), FrameDirection.DOWNSTREAM)
                call_stats.last_speech_time = current_time

    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call and log statistics.",
        properties={},
        required=[],
    )

    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. If the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function."""

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm.register_function("terminate_call", terminate_call)

    messages = [call_config_manager.create_system_message(system_instruction)]
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------
    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        call_stats.last_speech_time = datetime.now()

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        call_stats.end_call()
        logger.info(f"Call statistics: {call_stats.to_dict()}")
        await task.cancel()

    @transport.event_handler("on_speech_detected")
    async def on_speech_detected(transport, participant):
        call_stats.last_speech_time = datetime.now()

    # Start silence detection task
    async def silence_detection_task():
        while True:
            await handle_silence(FunctionCallParams(llm=llm))
            await asyncio.sleep(1)  # Check every second

    # ------------ RUN PIPELINE ------------
    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    silence_task = asyncio.create_task(silence_detection_task())
    try:
        await runner.run(task)
    finally:
        silence_task.cancel()
        try:
            await silence_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body)) 