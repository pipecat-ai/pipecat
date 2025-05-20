

import argparse
import asyncio
import os
import sys

from call_connection_manager import CallConfigManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

# New imports for Azure STT and Smallest.ai TTS
from pipecat.services.azure.stt import AzureSpeechService
from pipecat.services.smallest.tts import SmallestTTSService

# Load environment
load_dotenv(override=True)

# Main bot entrypoint
async def main(room_url: str, token: str, body: dict):
    # ------------ CONFIGURATION AND SETUP ------------
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()
    dialout_settings = call_config_manager.get_dialout_settings()
    test_mode = call_config_manager.is_test_mode()

    # Environment-driven API keys / URLs
    daily_api_url = os.getenv("DAILY_API_URL", "")
    daily_api_key = os.getenv("DAILY_API_KEY", "")

    # ------------ TRANSPORT SETUP ------------
    transport_params = DailyParams(
        api_url=daily_api_url,
        api_key=daily_api_key,
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        vad_analyzer=SileroVADAnalyzer(),
        transcription_enabled=False,   # disable built-in ASR, using Azure instead
    )
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-out Bot",
        transport_params,
    )

    # ------------ SERVICE INSTANTIATION ------------
    # 1️⃣ LLM: OpenAI GPT-4o Mini (or gpt-4.1-nano)
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    # 2️⃣ STT: Azure Speech (Indian English)
    stt = AzureSpeechService(
        api_key=os.getenv("AZURE_SPEECH_KEY", ""),
        region=os.getenv("AZURE_SPEECH_REGION", ""),
        language="en-IN",
    )

    # 3️⃣ TTS: Smallest.ai
    tts = SmallestTTSService(
        api_key=os.getenv("SMALLEST_API_KEY", ""),
        voice_id=os.getenv("SMALLEST_VOICE_ID", ""),
    )

    # ------------ FUNCTION DEFINITIONS ------------
    async def terminate_call(params: FunctionCallParams):
        # Signal end of call by queuing an EndTaskFrame upstream
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # Setup LLM with function support
    llm.register_function("terminate_call", terminate_call)

    # System prompt from environment or default
    system_instruction = os.getenv("BOT_SYSTEM_PROMPT", "You are a helpful phone agent.")
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------
    pipeline = Pipeline([
        transport.input(),             # capture raw audio
        stt,                           # Azure STT → text
        context_aggregator.user(),     # user text → LLM context
        llm,                           # generate response text
        tts,                           # Smallest.ai TTS → audio
        transport.output(),            # send audio back to room
        context_aggregator.assistant() # update context with assistant output
    ])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------
    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        if not test_mode and dialout_settings:
            logger.debug("Dialout settings detected; starting dialout")
            await call_config_manager.start_dialout(transport, dialout_settings)

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.debug(f"Dial-out connected: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        logger.debug(f"Dial-out answered: {data}")
        # Begin capturing quality transcription via Azure STT
        await transport.capture_participant_transcription(data["sessionId"])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        if test_mode:
            logger.debug(f"First participant joined: {participant['id']}")
            await transport.capture_participant_transcription(participant["id"])

    # ------------ RUNNER ------------
    runner = PipelineRunner(transport, task)
    await runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-out Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    load_dotenv(override=True)
    asyncio.run(main(args.url, args.token, args.body))
