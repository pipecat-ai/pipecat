#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.gated_llm_context import GatedLLMContextAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.filters.null_filter import NullFilter
from pipecat.processors.filters.wake_notifier_filter import WakeNotifierFilter
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAIContextAggregatorPair, OpenAILLMService
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


class TurnDetectionLLM(Pipeline):
    def __init__(self, llm: LLMService, context_aggregator: OpenAIContextAggregatorPair):
        # This is the LLM that will be used to detect if the user has finished a
        # statement. This doesn't really need to be an LLM, we could use NLP
        # libraries for that, but it was easier as an example because we
        # leverage the context aggregators.
        statement_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        statement_messages = [
            {
                "role": "system",
                "content": "Determine if the user's statement is a complete sentence or question, ending in a natural pause or punctuation. Return 'YES' if it is complete and 'NO' if it seems to leave a thought unfinished.",
            },
        ]

        statement_context = LLMContext(statement_messages)
        statement_context_aggregator = LLMContextAggregatorPair(statement_context)

        # We have instructed the LLM to return 'YES' if it thinks the user
        # completed a sentence. So, if it's 'YES' we will return true in this
        # predicate which will wake up the notifier.
        async def wake_check_filter(frame):
            logger.debug(f"Completeness check frame: {frame}")
            return frame.text == "YES"

        # This is a notifier that we use to synchronize the two LLMs.
        notifier = EventNotifier()

        # This a filter that will wake up the notifier if the given predicate
        # (wake_check_filter) returns true.
        completness_check = WakeNotifierFilter(
            notifier, types=(TextFrame,), filter=wake_check_filter
        )

        # This processor keeps the last context and will let it through once the
        # notifier is woken up. We start with the gate open because we send an
        # initial context frame to start the conversation.
        gated_context_aggregator = GatedLLMContextAggregator(notifier=notifier, start_open=True)

        # Notify if the user hasn't said anything.
        async def user_idle_notifier(frame):
            await notifier.notify()

        # Sometimes the LLM will fail detecting if a user has completed a
        # sentence, this will wake up the notifier if that happens.
        user_idle = UserIdleProcessor(callback=user_idle_notifier, timeout=3.0)

        # The ParallePipeline input are the user transcripts. We have two
        # contexts. The first one will be used to determine if the user finished
        # a statement and if so the notifier will be woken up. The second
        # context is simply the regular context but it's gated waiting for the
        # notifier to be woken up.
        super().__init__(
            [
                ParallelPipeline(
                    [
                        statement_context_aggregator.user(),
                        statement_llm,
                        completness_check,
                        NullFilter(),
                    ],
                    [context_aggregator.user(), gated_context_aggregator, llm],
                ),
                user_idle,
            ]
        )


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    # This is the regular LLM.
    llm_main = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # LLM + turn detection (with an extra LLM as a judge)
    llm = TurnDetectionLLM(llm_main, context_aggregator)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            llm,  # LLM with turn detection
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
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
