#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A LLM switching flow example for Pipecat Flows.

This example demonstrates how to dynamically switch between different LLM providers
during a conversation using Pipecat's LLMSwitcher.

Multi-LLM Support:
This example requires API keys for all supported LLM providers:
- OpenAI (default), Google, Anthropic, and AWS Bedrock
- Users can switch between providers in real-time during conversation

Requirements:
- CARTESIA_API_KEY (required for TTS)
- DEEPGRAM_API_KEY (required for STT)
- DAILY_API_KEY (optional for transport)
- OPENAI_API_KEY (for OpenAI LLM)
- GOOGLE_API_KEY (for Google LLM)
- ANTHROPIC_API_KEY (for Anthropic LLM)
- AWS credentials configured (for AWS Bedrock LLM)
"""

import os
import sys
from typing import TypedDict

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import FlowManager, NodeConfig
from pipecat.flows.types import ContextStrategy, ContextStrategyConfig
from pipecat.frames.frames import ManuallySwitchServiceFrame
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import ServiceSwitcherStrategyManual
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

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
    # Behavioral evals: run with `-t eval` to drive this bot via `pipecat eval`.
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


class SwitchLLMResult(TypedDict):
    """Result of switching the LLM service."""

    status: str
    message: str


async def switch_llm(flow_manager: FlowManager, llm: str) -> tuple[SwitchLLMResult, None]:
    """Switch the current LLM service.

    Args:
        llm: The name of the LLM service to switch to. Must be one of "OpenAI", "Google", "Anthropic", or "AWS".
    """
    if llm == "OpenAI":
        new_llm = llm_openai
    elif llm == "Google":
        new_llm = llm_google
    elif llm == "Anthropic":
        new_llm = llm_anthropic
    elif llm == "AWS":
        new_llm = llm_aws

    if llm_switcher.active_llm == new_llm:
        return SwitchLLMResult(status="success", message=f"Already using {llm} LLM service."), None

    # Typically, you would just switch LLMs like this:
    # await flow_manager.worker.queue_frames([ManuallySwitchServiceFrame(service=new_llm)])

    # But because we're in a tool call, and tool calls result in upstream
    # updates from the assistant context aggregator, we're pushing the
    # LLM-switching frame upstream from the aggregator to guarantee that the
    # switch happens before the LLM is run with the tool call result.
    await context_aggregator.assistant().push_frame(
        ManuallySwitchServiceFrame(service=new_llm), FrameDirection.UPSTREAM
    )

    return SwitchLLMResult(status="success", message=f"Switched to {llm} LLM service."), None


class WeatherResult(TypedDict):
    """Result of getting the current weather."""

    status: str
    conditions: str
    temperature: int


async def get_current_weather(
    flow_manager: FlowManager, location: str, format: str
) -> tuple[WeatherResult, None]:
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    # This is a placeholder for the actual implementation
    # In a real scenario, you would call an API to get the weather data
    return WeatherResult(
        status="success", conditions="sunny", temperature=75 if format == "fahrenheit" else 24
    ), None


async def summarize_conversation(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Summarize the conversation so far."""
    return None, create_main_node(summarize=True)


def create_main_node(summarize: bool = False) -> NodeConfig:
    return NodeConfig(
        name="main",
        role_message="You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY,
            summary_prompt="Summarize the conversation so far in a concise way.",
        )
        if summarize
        else ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        task_messages=[
            {
                "role": "developer",
                "content": "Say the conversation summary, which was already retrieved (do not invoke the summarize_conversation function again)."
                if summarize
                else "Say a brief hello.",
            }
        ],
        functions=[switch_llm, get_current_weather, summarize_conversation],
    )


# Main setup
async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the LLM switching bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    # Shared context and aggregators for LLM services
    context = LLMContext()
    global context_aggregator
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=True,
        ),
    )

    # LLM services
    global llm_openai, llm_google, llm_anthropic, llm_aws, llm_switcher
    llm_openai = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm_google = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY", ""))
    llm_anthropic = AnthropicLLMService(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    llm_aws = AWSBedrockLLMService(
        aws_region="us-west-2",
        model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        params=AWSBedrockLLMService.InputParams(temperature=0.8, latency="optimized"),
    )
    llm_switcher = LLMSwitcher(
        llms=[llm_openai, llm_google, llm_anthropic, llm_aws],
        strategy_type=ServiceSwitcherStrategyManual,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm_switcher,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm_switcher,
        context_aggregator=context_aggregator,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.debug("Initializing flow manager")
        await flow_manager.initialize(create_main_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
