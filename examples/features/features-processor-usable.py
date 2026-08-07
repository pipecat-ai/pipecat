#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example demonstrating processor health and what a bot does about it.

Every processor reports whether it can still do its job, through ``is_usable``.
Most failures leave it True: a dropped websocket, a provider hiccup, anything
worth retrying. Some don't — when a provider rejects an API key, an unknown
model or an unsupported voice, or when a service has failed enough times to
stop trying, retrying will keep failing. Those flip ``is_usable`` to False,
which stops the processor from reconnecting or accepting more work.

Three things to watch, all wired up below:

- ``on_usable_changed`` fires on each processor whose health changes, so you can
  tell *which* one is in trouble.
- ``on_pipeline_error`` fires for every error. Reading
  ``frame.processor.is_usable`` in the handler is what separates an error the
  processor will recover from and one that ended its usefulness — the verdict is
  always in before the error reaches you.
- ``PipelineWorker(processor_unusable_policy=...)`` decides what happens next.
  ``END`` stops the bot, ``CANCEL`` stops it immediately, and ``CONTINUE`` (the
  default) leaves the decision to your handlers — use it when the application
  can recover on its own, for example by failing over to another provider with a
  ``ServiceSwitcher``, or by calling ``set_usable(True)`` once the underlying
  problem is dealt with.

To see it work, run with a deliberately wrong key for any of the services::

    OPENAI_API_KEY=not-a-real-key python features-processor-usable.py

The provider rejects the key, the service reports ``authentication`` and stops
being usable, both handlers below fire, and the bot ends — instead of retrying a
key that will keep being rejected for as long as it runs.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import ErrorFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.ai_service import AIService
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
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

    stt = CartesiaSTTService(api_key=os.environ["CARTESIA_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
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
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        # End the bot once a service can no longer do its job. Without this the
        # pipeline keeps running with a dead service, which is rarely what you
        # want in development.
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    # Watch each service, so you can tell which one is in trouble.
    for service in (stt, tts, llm):

        @service.event_handler("on_usable_changed")
        async def on_usable_changed(service: AIService, is_usable: bool):
            if not is_usable:
                logger.error(
                    f"{service} needs attention: check its API key, model and voice settings"
                )

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker: PipelineWorker, frame: ErrorFrame):
        """Report an error, saying whether the processor survived it.

        ``frame.category`` says what kind of failure it was, which is what
        distinguishes "wrong API key" from "this voice doesn't exist", while
        ``frame.processor.is_usable`` says whether it is worth trying again.
        """
        if frame.processor and not frame.processor.is_usable:
            logger.error(f"{frame.processor} is spent ({frame.category.value}): {frame.error}")
        else:
            logger.warning(f"{frame.processor} hit a problem it can recover from: {frame.error}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

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
