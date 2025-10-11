#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Moss-Powered Customer Support Voice Agent with Pipecat.

This example demonstrates how to create an intelligent customer support voice assistant
that leverages semantic search capabilities using InferEdge Moss integration. It shows
how to build an agent that can search through knowledge bases and provide contextual
responses based on semantic similarity rather than keyword matching.

The example:
    1. Sets up a real-time voice conversation between customers and an AI support agent
    2. Uses Moss semantic search to retrieve relevant context from knowledge bases
    3. Enhances LLM responses with retrieved context for accurate, informed answers
    4. Processes audio input/output for natural voice interactions
    5. Intercepts LLM messages to inject relevant knowledge base context
    6. Supports both Daily and WebRTC transports for flexible deployment

Example usage:
    $ pip install -r requirements.txt # Install dependencies including inferedge-moss
    $ python bot.py
    # or
    $ uv run python bot.py

Requirements:
    - Deepgram API key (for speech-to-text)
    - Cartesia API key (for text-to-speech with natural voices)
    - OpenAI API key (for LLM)
    - Moss Project ID, Key, and Index Name (for semantic search)
      See https://github.com/usemoss/moss-samples for examples on how to set up indexes

    Environment variables (set in .env or in your terminal using `export`):
        DEEPGRAM_API_KEY=your_deepgram_api_key
        CARTESIA_API_KEY=your_cartesia_api_key
        OPENAI_API_KEY=your_openai_api_key
        OPENAI_MODEL=your_openai_model  # e.g., gpt-4
        MOSS_PROJECT_ID=your_moss_project_id
        MOSS_PROJECT_KEY=your_moss_project_key
        MOSS_INDEX_NAME=your_knowledge_base_index

The bot runs as part of a pipeline that processes voice frames, performs semantic search
on user queries, and provides contextually-aware responses for customer support scenarios.
Open http://localhost:7860/client after starting to begin voice conversations.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame

print("🚀 Starting Customer Support Voice AI Bot...")
print("⏳ Loading models and imports (20 seconds first run only)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from inferedge_moss import MossContextRetriever

logger.info("✅ All components loaded successfully!")

# Add project root to path for our imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv(override=True)


def create_llm_service():
    """Create LLM service with standard OpenAI."""
    logger.info("🔧 Using standard OpenAI configuration")
    
    # Get configuration for OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    return OpenAILLMService(
        api_key=api_key,
        model=model
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.debug(f"Starting customer support bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # Create and initialize Moss context retriever
    moss_context_retriever = MossContextRetriever(
        project_id=os.getenv("MOSS_PROJECT_ID"),
        project_key=os.getenv("MOSS_PROJECT_KEY"),
        index_name=os.getenv("MOSS_INDEX_NAME")
    )
    logger.debug("⏳ Initializing Moss semantic search...")
    await moss_context_retriever.initialize_index()
    logger.debug("✅ Moss semantic search ready")

    llm = create_llm_service()

    # System prompt with semantic retrieval support
    system_content = """You are a helpful customer support voice assistant. Your role is to assist customers with their questions about orders, shipping, returns, payments, and general inquiries.

Guidelines:
- Be friendly, professional, and concise in your responses
- Keep responses conversational since this is a voice interface
- Use any provided knowledge base context to give accurate, helpful answers
- If you don't have specific information, acknowledge this and offer to connect them with a human agent
- Ask clarifying questions if the customer's request is unclear
- Always prioritize customer satisfaction and be empathetic

When relevant knowledge base information is provided, use it to give accurate and detailed responses."""

    messages = [
        {
            "role": "system",
            "content": system_content,
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    
    pipeline = Pipeline([
        transport.input(),  # Transport user input
        rtvi,  # RTVI processor  
        stt,  # Speech-to-text
        context_aggregator.user(),  # User responses
        moss_context_retriever,  # Moss context retriever (intercepts LLM messages)
        llm,  # LLM (receives enhanced context)
        tts,  # Text-to-speech
        transport.output(),  # Transport bot output
        context_aggregator.assistant(),  # Assistant spoken responses
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Customer connected to support")
        # Kick off the conversation with a customer support greeting
        greeting = "A customer has just connected to customer support. Greet them warmly and ask how you can help them today."
        messages.append({"role": "system", "content": greeting})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Customer disconnected from support")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the customer support bot."""

    # Check required environment variables
    required_vars = ["DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "OPENAI_API_KEY", "OPENAI_MODEL", "MOSS_PROJECT_ID", "MOSS_PROJECT_KEY", "MOSS_INDEX_NAME"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error("❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        logger.error("\n🔧 Please update your .env file with the required API keys")
        return

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()