#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat.services.mcp_run.mcp_run import MCPRun

from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService

load_dotenv(override=True)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Bot with MCP tools",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-7-sonnet-latest")
        # llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")
        # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        mcp_run = MCPRun(llm)
        tools = mcp_run.register_mcp_tools(llm)

        system = """
                You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities 
                in a succinct way. You have access to various tools provided by mcp.run that you can use to help users.
                Your output will be converted to audio so don't include special characters in your answers. 
                Respond to what the user said in a creative and helpful way. Don't overexplain what you are doing.
                Just respond with short sentences when you are carrying out tool calls.
                """

        messages = [{"role": "system","content": system}]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                context_aggregator.user(),  # User spoken responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses and tool context
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info("First participant joined: {}", participant["id"])
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
