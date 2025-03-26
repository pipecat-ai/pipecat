#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_audio_passthrough=True,
                # set stop_secs to something roughly similar to the internal setting
                # of the Multimodal Live api, just to align events. This doesn't really
                # matter because we can only use the Multimodal Live API's phrase
                # endpointing, for now.
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            # system_instruction="Talk like a pirate."
        )

        pipeline = Pipeline(
            [
                transport.input(),
                llm,
                transport.output(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[
                            {
                                "role": "user",
                                "content": "Greet the user.",
                            }
                        ]
                    )
                ]
            )

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
