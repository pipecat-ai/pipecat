#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.lemonslice.transport import (
    LemonSliceNewSessionRequest,
    LemonSliceParams,
    LemonSliceTransport,
)
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        transport = LemonSliceTransport(
            bot_name="Pipecat",
            api_key=os.environ["LEMONSLICE_API_KEY"],
            session=session,
            session_request=LemonSliceNewSessionRequest(
                agent_id=os.getenv("LEMONSLICE_AGENT_ID"),
            ),
            params=LemonSliceParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                microphone_out_enabled=False,
            ),
        )

        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

        llm = GroqLLMService(
            api_key=os.environ["GROQ_API_KEY"],
            settings=GroqLLMService.Settings(
                system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
            ),
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            settings=ElevenLabsTTSService.Settings(
                voice=os.getenv("ELEVENLABS_VOICE_ID", ""),
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
                stt,  # STT
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
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            logger.info("Client connected")
            # Kick off the conversation.
            context.add_message(
                {
                    "role": "developer",
                    "content": "Start by greeting the user and ask how you can help.",
                }
            )
            await worker.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, participant):
            logger.info("Client disconnected")
            await worker.cancel()

        @transport.event_handler("on_avatar_connected")
        async def on_avatar_connected(transport, participant):
            logger.info("Avatar connected")

        @transport.event_handler("on_avatar_disconnected")
        async def on_avatar_disconnected(transport, participant, reason):
            logger.info(f"Avatar disconnected. Reason: {reason}")

        runner = WorkerRunner()

        await runner.add_workers(worker)
        await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
