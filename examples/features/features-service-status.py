#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example demonstrating service health status and configuration errors.

Services report their health as a ``ServiceStatus``. Most failures are
transient — a dropped websocket, a provider hiccup — and the service keeps
retrying. Some are not: when a provider rejects an API key, an unknown model or
an unsupported voice, retrying will keep failing until the configuration
changes. Those move the service to ``ServiceStatus.MISCONFIGURED``, which stops
it from reconnecting or accepting more work.

Three things to watch, all wired up below:

- ``on_status_changed`` fires on each service whose health changes, so you can
  tell *which* service is in trouble and how.
- ``on_pipeline_configuration_error`` fires once per misconfigured service, with
  the ``ErrorFrame`` that identified it. Once per service, not once per failed
  request — a rejected API key reports a single time rather than on every chunk
  of audio.
- ``PipelineWorker(on_configuration_error=...)`` decides what happens next.
  ``END`` stops the bot, ``CANCEL`` stops it immediately, and ``CONTINUE`` (the
  default) leaves the decision to your handlers — use it when the application
  can recover on its own, for example by failing over to another provider.

To see it work, run with a deliberately wrong key for any of the services::

    OPENAI_API_KEY=not-a-real-key python features-service-status.py

The provider rejects the key, the service reports ``authentication`` and moves
to ``misconfigured``, both handlers below fire once, and the bot ends — instead
of retrying a key that will keep being rejected for as long as it runs.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import ErrorFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import ConfigurationErrorPolicy, PipelineParams, PipelineWorker
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
from pipecat.services.status import ServiceStatus
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
        # End the bot when a service can't work with its current configuration.
        # Without this the pipeline keeps running with a dead service, which is
        # rarely what you want in development.
        on_configuration_error=ConfigurationErrorPolicy.END,
    )

    # Watch every service's health, not just the ones that fail fatally. A
    # service that reconnects successfully reports DEGRADED and then READY, so
    # this is also how you spot a provider that is flapping.
    for service in (stt, tts, llm):

        @service.event_handler("on_status_changed")
        async def on_status_changed(
            service: AIService, previous: ServiceStatus, current: ServiceStatus
        ):
            logger.info(f"{service}: {previous.value} -> {current.value}")

            if not current.is_usable:
                logger.error(
                    f"{service} needs attention: check its API key, model and voice settings"
                )

    @worker.event_handler("on_pipeline_configuration_error")
    async def on_pipeline_configuration_error(worker: PipelineWorker, frame: ErrorFrame):
        """Report a service the provider has rejected.

        Fires once per service, carrying the error that identified the problem.
        ``frame.category`` says what kind of rejection it was, which is what
        distinguishes "wrong API key" from "this voice doesn't exist".
        """
        logger.error(f"Configuration error from {frame.processor}: {frame.error}")
        logger.error(f"  category: {frame.category.value}")

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
