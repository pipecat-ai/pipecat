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
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
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

    stt = CartesiaSTTService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaSTTService.Settings(
            model="ink-2",
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            # An "Emotive" voice, which Cartesia recommends for emotion.
            voice="57dcab65-68ac-45a6-8480-6c4c52ec1cd1",
        ),
    )

    # Teach the LLM the expression tags the TTS can perform. Cartesia has many
    # more emotions (see CartesiaEmotion), but offered more than about 15, the
    # LLM starts inventing names of its own as a conversation goes on, so the
    # prompt names the ones this conversation calls for.
    emotions = ", ".join(
        [
            "calm",
            "content",
            "sympathetic",
            "apologetic",
            "grateful",
            "happy",
            "excited",
            "curious",
            "confident",
            "sad",
            "surprised",
            "amazed",
            "hesitant",
            "anxious",
            "proud",
        ]
    )
    system_instruction = f"""You are a warm, expressive assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Keep your responses brief.

You can control how your words are spoken with tags:
- <<emotion>> at the start of a sentence speaks it, and the sentences after it, with that emotion, until the next emotion tag. <<neutral>> returns to your normal voice.
- [[laughter]] laughs at that point.
- [[pause 500]] pauses for that many milliseconds.

Your voice can perform only these emotions: {emotions}. Write an emotion exactly as it appears in this list. Any other word in an emotion tag is ignored, so if no emotion in the list fits, write no tag.

Let your voice match the conversation: be excited for the user's good news, sympathetic with their bad news, and laugh when they say something funny. Most sentences need no tag. Some feelings have no emotion of their own: for relief, use happy or content; to reassure, comfort or encourage, use calm or sympathetic; for concern or support, use sympathetic.

Before writing an emotion tag, check that the emotion is in the list above."""

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
            # The TTS reads the expression tags in the LLM's responses.
            enable_expressive_conversation=True,
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
