#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger

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
from pipecat.services.aws.llm import AWSBedrockLLMService, AWSBedrockLLMSettings
from pipecat.services.deepgram.flux.sagemaker.stt import DeepgramFluxSageMakerSTTService
from pipecat.services.deepgram.sagemaker.tts import DeepgramSageMakerTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Initialize Deepgram Flux SageMaker STT Service
    # This requires:
    # - AWS credentials configured (via environment variables or AWS CLI)
    # - A deployed SageMaker endpoint with Deepgram Flux model
    stt = DeepgramFluxSageMakerSTTService(
        endpoint_name=os.getenv("SAGEMAKER_STT_ENDPOINT_NAME"),
        region=os.getenv("AWS_REGION"),
        settings=DeepgramFluxSageMakerSTTService.Settings(
            min_confidence=0.3,
        ),
    )

    # Initialize Deepgram SageMaker TTS Service
    # This requires:
    # - AWS credentials configured (via environment variables or AWS CLI)
    # - A deployed SageMaker endpoint with Deepgram TTS model
    tts = DeepgramSageMakerTTSService(
        endpoint_name=os.getenv("SAGEMAKER_TTS_ENDPOINT_NAME"),
        region=os.getenv("AWS_REGION"),
        settings=DeepgramSageMakerTTSService.Settings(
            voice="aura-2-andromeda-en",
        ),
    )

    llm = AWSBedrockLLMService(
        aws_region=os.getenv("AWS_REGION"),
        settings=AWSBedrockLLMSettings(
            model="us.amazon.nova-pro-v1:0",
            temperature=0.8,
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    context = LLMContext()
    # Use ExternalUserTurnStrategies since Flux handles turn detection natively
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=ExternalUserTurnStrategies(),
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
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
        context.add_message({"role": "user", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    @stt.event_handler("on_update")
    async def on_deepgram_flux_update(stt, transcript):
        logger.debug(f"On deepgram flux update: {transcript}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
