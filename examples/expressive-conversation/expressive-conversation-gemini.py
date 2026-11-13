#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.expressive.expression import ExpressionDelivery, ExpressionSound
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.tts import GeminiTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

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
    "livekit": lambda: LiveKitParams(
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

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # Gemini 3.8 TTS models, on the Gemini API, take each sentence's delivery as
    # a style and perform sounds and pauses written in the text.
    tts = GeminiTTSService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiTTSService.Settings(model="gemini-3.8-flash-lite-tts", voice="Kore"),
    )

    # With expressive conversation on, the LLM is taught the expression tags the TTS
    # performs. Gemini 3.8 performs every universal delivery, so the policy offers
    # the ones this conversation calls for.
    system_instruction = "You are a warm, expressive assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Keep your responses brief."

    expression_policy = ExpressionPolicy(
        guidance=[
            ExpressionGuidance.MATCH_STYLE,
            ExpressionGuidance.SLOW_FOR_DETAILS,
            ExpressionGuidance.BREATHE,
        ],
        delivery=[
            ExpressionDelivery.SLOW,
            ExpressionDelivery.HAPPY,
            ExpressionDelivery.EXCITED,
            ExpressionDelivery.CURIOUS,
            ExpressionDelivery.THOUGHTFUL,
            ExpressionDelivery.WARM,
            ExpressionDelivery.SYMPATHETIC,
            ExpressionDelivery.REASSURING,
            ExpressionDelivery.NERVOUS,
            ExpressionDelivery.SAD,
            ExpressionDelivery.FRUSTRATED,
            ExpressionDelivery.SARCASTIC,
            ExpressionDelivery.DEADPAN,
            ExpressionDelivery.MISCHIEVOUS,
            ExpressionDelivery.WHISPERING,
        ],
        sounds=[
            ExpressionSound.LAUGH,
            ExpressionSound.CHUCKLE,
            ExpressionSound.GIGGLE,
            ExpressionSound.SIGH,
            ExpressionSound.LIGHT_BREATH,
            ExpressionSound.CLEAR_THROAT,
            ExpressionSound.GULP,
        ],
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(system_instruction=system_instruction),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            # The LLM writes expression tags into its responses, and the TTS
            # performs them.
            enable_expressive_conversation=True,
            expression_policy=expression_policy,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
