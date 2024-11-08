#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import aiohttp
import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.gated_openai_llm_context import GatedOpenAILLMContextAggregator
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.null_filter import NullFilter
from pipecat.processors.filters.wake_notifier_filter import WakeNotifierFilter
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.sync.event_notifier import EventNotifier
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        # This is the LLM that will be used to detect if the user has finished a
        # statement. This doesn't really need to be an LLM, we could use NLP
        # libraries for that, but it was easier as an example because we
        # leverage the context aggregators.
        statement_llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        statement_messages = [
            {
                "role": "system",
                "content": "Determine if the user's statement is a complete sentence or question, ending in a natural pause or punctuation. Return 'YES' if it is complete and 'NO' if it seems to leave a thought unfinished.",
            },
        ]

        statement_context = OpenAILLMContext(statement_messages)
        statement_context_aggregator = statement_llm.create_context_aggregator(statement_context)

        # This is the regular LLM.
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

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
        completness_check = WakeNotifierFilter(
            notifier, types=(TextFrame,), filter=wake_check_filter
        )

        # This processor keeps the last context and will let it through once the
        # notifier is woken up.
        gated_context_aggregator = GatedOpenAILLMContextAggregator(notifier)

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
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
