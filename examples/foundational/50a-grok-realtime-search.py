#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Grok Voice Agent with Built-in Search Tools

This example demonstrates Grok's unique built-in tools:
- web_search: Search the web for current information
- x_search: Search X/Twitter for posts and discussions

These tools are unique to Grok and don't require any function handlers -
they're processed entirely by the Grok API.

Requirements:
    - XAI_API_KEY environment variable set
    - pip install pipecat-ai[grok]

Usage:
    python 50a-grok-realtime-search.py --transport webrtc
"""

import os

from dotenv import load_dotenv
from loguru import logger

# Note: Grok has built-in server-side VAD, so we don't need local VAD
# from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TranscriptionMessage
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.grok.realtime.events import (
    AudioConfiguration,
    AudioInputFormat,
    AudioOutputFormat,
    PCMAudioFormat,
    SessionProperties,
    TurnDetection,
    WebSearchTool,
    XSearchTool,
)
from pipecat.services.grok.realtime.llm import GrokRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


# Grok uses 24kHz audio by default, so we configure the transports to match.
# Note: We don't need local VAD since Grok has built-in server-side VAD.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_in_sample_rate=24000,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_in_sample_rate=24000,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=24000,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Grok Voice Agent with Search Tools")

    # Configure session with Grok's built-in search tools
    session_properties = SessionProperties(
        voice="Rex",  # Using Rex for a confident, clear voice
        turn_detection=TurnDetection(type="server_vad"),
        audio=AudioConfiguration(
            input=AudioInputFormat(format=PCMAudioFormat(rate=24000)),
            output=AudioOutputFormat(format=PCMAudioFormat(rate=24000)),
        ),
        instructions="""You are a knowledgeable AI assistant with access to real-time information.

You have two powerful search capabilities:
1. Web Search - Use this to find current news, facts, and information from the web
2. X Search - Use this to find what people are saying on X (Twitter)

When users ask about:
- Current events, news, or recent developments → Use web search
- Public opinions, trending topics, or what people are saying → Use X search
- Technical questions or facts → Use web search

Be conversational and summarize search results naturally. Don't just read out URLs.
Keep responses concise for voice interaction.""",
        # Enable Grok's built-in tools
        tools=[
            {"type": "web_search"},
            {"type": "x_search"},
        ],
    )

    llm = GrokRealtimeLLMService(
        api_key=os.getenv("XAI_API_KEY"),
        session_properties=session_properties,
        sample_rate=24000,
    )

    transcript = TranscriptProcessor()

    context = LLMContext(
        [
            {
                "role": "user",
                "content": "Hello! I want to know what's happening in the world today.",
            }
        ],
    )

    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            transcript.user(),
            llm,
            transport.output(),
            transcript.assistant(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - Starting Grok with search tools")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                logger.info(f"Transcript: {timestamp}{msg.role}: {msg.content}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
