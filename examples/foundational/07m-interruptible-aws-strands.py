#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# Strands agent setup
try:
    from strands import Agent, tool
    from strands.models import BedrockModel
except ImportError:
    logger.warning("Strands not installed. Please install with: pip install strands-agents")
    Agent = None
    BedrockModel = None

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
        if location.lower() == "san francisco":
            return "The weather in San Francisco is sunny and 75 degrees."
        elif location.lower() == "sydney":
            return "The weather in Sydney is cloudy and 60 degrees."
        else:
            return "I'm not sure about the weather in that location."

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
        voice_id="Joanna",
        params=AWSPollyTTSService.InputParams(engine="generative", rate="1.1"),
    )

    # Create Strands agent processor
    try:
        agent = build_agent(model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0", max_tokens=8000)
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
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
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
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Greet the user and introduce yourself.",
                        }
                    ],
                    run_llm=True,
                )
            ]
        )

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
    from pipecat.runner.run import main

    main()
