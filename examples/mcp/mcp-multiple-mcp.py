#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os
import shutil

from dotenv import load_dotenv
from loguru import logger
from mcp import StdioServerParameters
from mcp.client.session_group import StreamableHttpParameters

from pipecat.adapters.schemas.tools_schema import ToolsSchema
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
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.mcp_service import MCPClient
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

load_dotenv(override=True)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    system_prompt = f"""
    You are a helpful LLM in a voice call.
    Your goal is to demonstrate your capabilities in a succinct way.
    You have access to memory tools that let you store and recall information,
    and tools to answer questions about the user's GitHub repositories and account.
    Offer to remember things for the user, like their name, preferences, or anything they'd like.
    You can also recall things you've previously stored.
    You can also offer to answer users questions about their GitHub repositories and account.
    Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points.
    Respond to what the user said in a creative and helpful way.
    Don't overexplain what you are doing.
    Just respond with short sentences when you are carrying out tool calls.
    """

    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        settings=AnthropicLLMService.Settings(
            system_instruction=system_prompt,
        ),
    )

    async with (
        # https://github.com/modelcontextprotocol/servers/tree/main/src/memory
        MCPClient(
            server_params=StdioServerParameters(
                command=shutil.which("npx"),
                args=["-y", "@modelcontextprotocol/server-memory"],
                # env={"MEMORY_FILE_PATH": "/tmp/pipecat_memory.jsonl"}, # Optional: specify MEMORY_FILE_PATH
            ),
        ) as memory_mcp,
        # Github MCP docs: https://github.com/github/github-mcp-server
        # Enable Github Copilot on your GitHub account. Free tier is ok. (https://github.com/settings/copilot)
        # Generate a personal access token. It must be a Fine-grained token, classic tokens are not supported. (https://github.com/settings/personal-access-tokens)
        # Set permissions you want to use (eg. "all repositories", "profile: read/write", etc)
        MCPClient(
            server_params=StreamableHttpParameters(
                url="https://api.githubcopilot.com/mcp/",
                headers={"Authorization": f"Bearer {os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')}"},
            ),
        ) as github_mcp,
    ):
        memory_tools = await memory_mcp.register_tools(llm)
        github_tools = await github_mcp.register_tools(llm)

        all_standard_tools = memory_tools.standard_tools + github_tools.standard_tools
        all_tools = ToolsSchema(standard_tools=all_standard_tools)

        context = LLMContext(
            messages=[{"role": "user", "content": "Please introduce yourself."}],
            tools=all_tools,
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
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        logger.error(f"Please set `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable.")
        import sys

        sys.exit(1)

    from pipecat.runner.run import main

    main()
