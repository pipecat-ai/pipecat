#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from typing import Any, Mapping

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.tavus.video import TavusVideoService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        tavus = TavusVideoService(
            api_key=os.getenv("TAVUS_API_KEY"),
            replica_id=os.getenv("TAVUS_REPLICA_ID"),
            session=session,
        )

        # get persona, look up persona_name, set this as the bot name to ignore
        persona_name = await tavus.get_persona_name()
        room_url = await tavus.initialize()

        transport = DailyTransport(
            room_url=room_url,
            token=None,
            bot_name="Pipecat bot",
            params=DailyParams(
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a167e0f3-df7e-4d52-a9c3-f949145efdab",
        )

        llm = OpenAILLMService(model="gpt-4o-mini")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                tavus,  # Tavus output layer
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                # We just use 16000 because that's what Tavus is expecting and
                # we avoid resampling.
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(
            transport: DailyTransport, participant: Mapping[str, Any]
        ) -> None:
            # Ignore the Tavus replica's microphone
            if participant.get("info", {}).get("userName", "") == persona_name:
                logger.debug(f"Ignoring {participant['id']}'s microphone")
                await transport.update_subscriptions(
                    participant_settings={
                        participant["id"]: {
                            "media": {"microphone": "unsubscribed"},
                        }
                    }
                )

            if participant.get("info", {}).get("userName", "") != persona_name:
                # Kick off the conversation.
                messages.append(
                    {"role": "system", "content": "Please introduce yourself to the user."}
                )
                await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
