#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example showing how to use MCP resources with Pipecat.

MCP resources are read-only data exposed by MCP servers (e.g., files,
database records, application state). This example fetches resources
from an MCP server and injects their content into the LLM system prompt,
giving the bot awareness of external context.

Resources are different from tools:
- Tools are actions the LLM can invoke (function calls).
- Resources are data the LLM can read (context injection).
"""

import os

from dotenv import load_dotenv
from loguru import logger
from mcp.client.session_group import StreamableHttpParameters

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.mcp_service import MCPClient
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def build_system_prompt(resources):
    """Build the system prompt with resource content injected."""
    base_prompt = """\
You are a helpful LLM in a voice call.
Your goal is to answer questions using the context provided to you.
Your output will be spoken aloud, so avoid special characters that can't easily be spoken.
Respond with short, conversational sentences.
"""

    if not resources:
        return base_prompt

    resource_section = "\n\nYou have access to the following context:\n\n"
    for r in resources:
        resource_section += f"### {r.name}\n"
        if r.description:
            resource_section += f"{r.description}\n"
        if r.text:
            resource_section += f"\n{r.text}\n\n"

    return base_prompt + resource_section


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    # Connect to an MCP server that exposes resources.
    # Replace the URL with your MCP server's address.
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

    async with MCPClient(
        server_params=StreamableHttpParameters(
            url=mcp_url,
            headers={"Accept": "application/json, text/event-stream"},
        )
    ) as mcp:
        # Fetch tools (if any)
        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
        tools = await mcp.register_tools(llm)

        # Fetch resources and inject into system prompt
        resources = await mcp.read_all_resources()
        logger.info(f"Loaded {len(resources)} MCP resources into context")
        for r in resources:
            logger.info(f"  - {r.name} ({r.uri})")

        system_prompt = build_system_prompt(resources)

        llm.settings.system_instruction = system_prompt

        context = LLMContext(
            messages=[{"role": "user", "content": "Please introduce yourself."}],
            tools=tools,
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,
                user_aggregator,  # User spoken responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                assistant_aggregator,  # Assistant spoken responses and tool context
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {client}")
            # Kick off the conversation.
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
