#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVIObserver ignored sources example.

This example shows how to suppress RTVI messages from a specific pipeline
processor so that secondary branches don't leak events to the client.

"""

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
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
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
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # Main LLM — drives the conversation. Its RTVI events reach the client.
    main_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    main_messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        },
    ]

    # Evaluator LLM — silently grades the user's message in the background.
    # Its RTVI events will be suppressed so the client is unaware of this branch.
    evaluator_llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        name="EvaluatorLLM",
    )

    evaluator_messages = [
        {
            "role": "system",
            "content": (
                "You are a silent quality evaluator. When given a user message, "
                "respond with a single JSON object: "
                '{"score": <1-5>, "reason": "<brief reason>"}. '
                "Do not respond conversationally."
            ),
        },
    ]

    main_context = LLMContext(main_messages)
    evaluator_context = LLMContext(evaluator_messages)

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
    main_context_aggregator = LLMContextAggregatorPair(
        main_context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )
    evaluator_context_aggregator = LLMContextAggregatorPair(
        evaluator_context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=ExternalUserTurnStrategies()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            vad_processor,
            user_turn_processor,
            ParallelPipeline(
                # Main branch: speaks to the user.
                [
                    main_context_aggregator.user(),
                    main_llm,
                    tts,
                    transport.output(),
                    main_context_aggregator.assistant(),
                ],
                # Evaluator branch: silent background scoring, no audio output.
                [
                    evaluator_context_aggregator.user(),
                    evaluator_llm,
                    evaluator_context_aggregator.assistant(),
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
        rtvi_observer_params=RTVIObserverParams(ignored_sources=[evaluator_llm]),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        main_messages.append(
            {"role": "system", "content": "Please introduce yourself to the user."}
        )
        evaluator_messages.append({"role": "system", "content": "Ready to evaluate user messages."})
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
