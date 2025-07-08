#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.gated_openai_llm_context import GatedOpenAILLMContextAggregator
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.null_filter import NullFilter
from pipecat.processors.filters.wake_notifier_filter import WakeNotifierFilter
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


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


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

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

    statement_context = OpenAILLMContext(statement_messages)
    statement_context_aggregator = statement_llm.create_context_aggregator(statement_context)

    # This is the regular LLM.
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # We have instructed the LLM to return 'YES' if it thinks the user
    # completed a sentence. So, if it's 'YES' we will return true in this
    # predicate which will wake up the notifier.
    async def wake_check_filter(frame):
        return frame.text == "YES"

    # This is a notifier that we use to synchronize the two LLMs.
    notifier = EventNotifier()

    # This a filter that will wake up the notifier if the given predicate
    # (wake_check_filter) returns true.
    completness_check = WakeNotifierFilter(notifier, types=(TextFrame,), filter=wake_check_filter)

    # This processor keeps the last context and will let it through once the
    # notifier is woken up. We start with the gate open because we send an
    # initial context frame to start the conversation.
    gated_context_aggregator = GatedOpenAILLMContextAggregator(notifier=notifier, start_open=True)

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
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
