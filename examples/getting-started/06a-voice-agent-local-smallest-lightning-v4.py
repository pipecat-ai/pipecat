#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Full conversational pipeline smoke test: mic -> Smallest STT -> OpenAI LLM ->
Smallest Lightning v4 TTS -> speakers.

Requires SMALLEST_API_KEY (Lightning v4 beta-enabled) and OPENAI_API_KEY in
the environment/.env, plus a current voice id from:

    curl https://api.smallest.ai/waves/v1/lightning-v4/get_voices
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.smallest.stt import SmallestSTTService
from pipecat.services.smallest.tts_v4 import SmallestLightningV4TTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
# Pipeline internals are noisy at DEBUG; keep them out of the way so the
# INFO-level conversation log (below) is easy to follow live.
logger.add(sys.stderr, level="INFO")


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = SmallestSTTService(
        api_key=os.environ["SMALLEST_API_KEY"],
        settings=SmallestSTTService.Settings(language=Language.EN),
    )

    tts = SmallestLightningV4TTSService(
        api_key=os.environ["SMALLEST_API_KEY"],
        settings=SmallestLightningV4TTSService.Settings(
            voice=os.environ.get("SMALLEST_LIGHTNING_V4_VOICE", "brannock"),
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
        user_params=LLMUserAggregatorParams(
            # Speakers bleeding into the mic (no echo cancellation on
            # LocalAudioTransport) can trip VAD on the bot's own voice.
            # Higher start_secs/confidence/min_volume make that less likely;
            # headphones remove the problem at the source.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.8,
                    start_secs=0.4,
                    stop_secs=0.3,
                    min_volume=0.7,
                )
            ),
        ),
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
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner()

    await runner.add_workers(worker)

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        logger.info(f"User: {message.content}")
        # Feed the caller's turn into Lightning v4's own server-held context,
        # not just the LLM's — nothing does this automatically.
        if message.content:
            await tts.add_user_turn(message.content)

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        interrupted = " (interrupted)" if message.interrupted else ""
        logger.info(f"Bot: {message.content}{interrupted}")

    context.add_message({"role": "developer", "content": "Please introduce yourself to the user."})
    await worker.queue_frames([LLMRunFrame()])

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
