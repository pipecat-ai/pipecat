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
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_turn_processor import UserTurnProcessor
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

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    openai_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    openai_messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        },
    ]

    groq_llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-maverick-17b-128e-instruct"
    )

    groq_messages = [
        {
            "role": "system",
            "content": "You are a very helpful assistant. Your goal is to demonstrate your capabilities in detail in a creative and helpful way.",
        },
    ]

    openai_context = LLMContext(openai_messages)
    groq_context = LLMContext(groq_messages)

    # We use an external VADProcessor because the UserTurnProcessor is shared
    # across multiple parallel aggregators. The VADProcessor emits
    # VADUserStartedSpeakingFrame and VADUserStoppedSpeakingFrame which the
    # UserTurnProcessor needs to manage turn lifecycle.
    vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    # We use this external user turn processor. This processor will push
    # UserStartedSpeakingFrame and UserStoppedSpeakingFrame as well as
    # interruptions. This can be used in advanced cases when there are multiple
    # aggregators in the pipeline.
    user_turn_processor = UserTurnProcessor()

    # We use external user turn strategies for both aggregators since the turn
    # management is done by the common UserTurnProcessor.
    openai_context_aggregator = LLMContextAggregatorPair(
        openai_context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )
    groq_context_aggregator = LLMContextAggregatorPair(
        groq_context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            vad_processor,
            user_turn_processor,
            ParallelPipeline(
                [
                    openai_context_aggregator.user(),  # User responses
                    openai_llm,  # LLM
                    tts,  # TTS (bot will speak the chosen language)
                    transport.output(),  # Transport bot output
                    openai_context_aggregator.assistant(),  # Assistant spoken responses
                ],
                [
                    groq_context_aggregator.user(),  # User responses
                    groq_llm,  # LLM
                    groq_context_aggregator.assistant(),  # Assistant responses
                ],
            ),
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
        openai_messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."}
        )
        groq_messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."}
        )
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
    from pipecat.runner.run import main

    main()
