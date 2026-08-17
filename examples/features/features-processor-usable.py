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

The TTS here is a ``ServiceSwitcher`` over two providers, which is the useful
thing to do about it: when Cartesia stops being usable, the switcher moves the
work to ElevenLabs and the bot keeps talking. Nothing upstream even hears about
it — the switcher recovered, so there is no error left to act on. Only when the
last provider is gone does the switcher report itself unusable, and that is what
ends the bot.

Four things to watch, all wired up below:

- ``on_usable_changed`` fires on each processor whose health changes, so you can
  tell *which* one is in trouble. A switcher raises it for itself too, once none
  of its services can work.
- ``on_service_switched`` fires when the switcher moves to another provider.
- ``on_pipeline_error`` fires for every error that isn't recovered from. Reading
  ``frame.processor.is_usable`` in the handler is what separates an error the
  processor will carry on from and one that ended its usefulness — the verdict
  is always in before the error reaches you.
- ``PipelineWorker(processor_unusable_policy=...)`` decides what happens next.
  ``END`` stops the bot, ``CANCEL`` stops it immediately, and ``CONTINUE`` (the
  default) leaves the decision to your handlers.

To watch the failover, run with a Cartesia key the provider will reject::

    CARTESIA_API_KEY=not-a-real-key python features-processor-usable.py

Cartesia is rejected, the switcher moves to ElevenLabs, and the conversation
carries on. Break both keys and the switcher runs out of providers, reports
itself unusable, and the bot ends — instead of retrying keys that will keep
being rejected for as long as it runs.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import ErrorFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import (
    ServiceSwitcher,
    ServiceSwitcherStrategy,
    ServiceSwitcherStrategyFailover,
)
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
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
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts_cartesia = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    tts_elevenlabs = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID", ""),
        ),
    )

    # The failover strategy moves to the next provider that can still work, and
    # only once the active one can't. An error Cartesia can carry on from — a
    # dropped websocket it reconnects — costs no switch.
    tts_switcher = ServiceSwitcher(
        services=[tts_cartesia, tts_elevenlabs],
        strategy_type=ServiceSwitcherStrategyFailover,
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
            tts_switcher,  # TTS, with a second provider to fall back on
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
        # End the bot once a processor can no longer do its job. For the TTS
        # that means both providers are gone, since the switcher answers for
        # them: losing one is something it recovers from on its own.
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    # Watch each processor, so you can tell which one is in trouble, and what to
    # look at when it is. The switcher is in the list on its own account:
    # losing one of its providers doesn't show here, but running out of them does.
    what_to_check = {
        stt: "check DEEPGRAM_API_KEY and the model",
        tts_cartesia: "check CARTESIA_API_KEY and the voice id",
        tts_elevenlabs: "check ELEVENLABS_API_KEY and the voice id",
        tts_switcher: "both TTS providers are gone, so there is nothing left to speak with",
        llm: "check OPENAI_API_KEY and the model",
    }

    for processor in what_to_check:

        @processor.event_handler("on_usable_changed")
        async def on_usable_changed(processor: FrameProcessor, is_usable: bool):
            if not is_usable:
                logger.error(f"{processor} can no longer do its job: {what_to_check[processor]}")

    @tts_switcher.strategy.event_handler("on_service_switched")
    async def on_service_switched(strategy: ServiceSwitcherStrategy, service: FrameProcessor):
        logger.info(f"TTS failed over to {service.name}; the bot keeps talking")

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker: PipelineWorker, frame: ErrorFrame):
        """Report an error, saying whether the processor survived it.

        ``frame.category`` says what kind of failure it was, which is what
        distinguishes "wrong API key" from "this voice doesn't exist", while
        ``frame.processor.is_usable`` says whether it is worth trying again.
        """
        if frame.processor and not frame.processor.is_usable:
            logger.error(
                f"{frame.processor} can no longer do its job "
                f"({frame.category.value}): {frame.error}"
            )
        else:
            logger.warning(f"{frame.processor} hit a problem it can recover from: {frame.error}")

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
