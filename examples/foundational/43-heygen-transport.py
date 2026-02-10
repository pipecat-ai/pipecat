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
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.heygen.api_liveavatar import LiveAvatarNewSessionRequest
from pipecat.transports.heygen.transport import HeyGenParams, HeyGenTransport, ServiceType

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        transport = HeyGenTransport(
            api_key=os.getenv("HEYGEN_LIVE_AVATAR_API_KEY"),
            service_type=ServiceType.LIVE_AVATAR,
            session=session,
            params=HeyGenParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
            session_request=LiveAvatarNewSessionRequest(
                is_sandbox=True,
                # Sandbox mode only works with this specific avatar
                # https://docs.liveavatar.com/docs/developing-in-sandbox-mode#sandbox-mode-behaviors
                avatar_id="dd73ea75-1218-4ef3-92ce-606d5f7fbc0a",
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="00967b2f-88a6-4a31-8153-110a92134b9f",
        )

        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Be succinct and respond to what the user said in a creative and helpful way.",
            },
        ]

        context = LLMContext(messages)
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
            messages.append(
                {
                    "role": "system",
                    "content": "Start by saying 'Hello' and then a short greeting.",
                }
            )
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
