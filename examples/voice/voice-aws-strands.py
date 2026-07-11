#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.strands_agents import StrandsAgentsProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

# Strands agent setup
try:
    from strands import Agent, tool
    from strands.models import BedrockModel
except ImportError:
    logger.warning("Strands not installed. Please install with: uv add strands-agents")
    Agent = None
    BedrockModel = None

load_dotenv(override=True)

# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
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


def build_agent(model_id: str, max_tokens: int):
    """Create and configure a Strands agent for NAB customer service coaching.

    Args:
        model_id: The AWS Bedrock model ID to use
        max_tokens: Maximum tokens for the model

    Returns:
        Configured Strands Agent
    """

    @tool
    def check_weather(location: str) -> str:
        return "The weather is nice and sunny with a temperature of 75 degrees."

    agent = Agent(
        model=BedrockModel(
            model_id=model_id,
            max_tokens=max_tokens,
        ),
        tools=[check_weather],
        system_prompt="You are a helpful assistant that can check the weather in a given location.",
    )

    return agent


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = AWSTranscribeSTTService()

    tts = AWSPollyTTSService(
        region="us-west-2",  # only specific regions support generative TTS
        settings=AWSPollyTTSService.Settings(
            voice="Joanna",
            engine="generative",
            rate="1.1",
        ),
    )

    # Create Strands agent processor
    try:
        agent = build_agent(model_id="us.anthropic.claude-sonnet-4-6", max_tokens=8000)
        llm = StrandsAgentsProcessor(agent=agent)
        logger.info("Successfully created Strands agent for NAB customer service coaching")
    except Exception as e:
        logger.error(f"Failed to create Strands agent: {e}")
        raise ValueError(
            "Unable to create Strands processor. Please ensure you have properly "
            "installed strands-agents and configured your AWS credentials."
        )

    # Setup context aggregators for message handling
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech-to-text
            user_aggregator,  # User responses
            llm,  # Strands Agents processor
            tts,  # Text-to-speech
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await worker.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": f"Greet the user and introduce yourself. Don't use emojis.",
                        }
                    ],
                    run_llm=True,
                )
            ]
        )

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
