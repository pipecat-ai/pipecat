#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


async def main(room_url: str, token: str):
    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        ),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    messages = [
        {
            "role": "system",
            "content": "You are Chatbot, a friendly, helpful robot. Your output will be converted to audio so don't include special characters other than '!' or '?' in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by saying hello.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.cancel()

    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        if state == "left":
            # Here we don't want to cancel, we just want to finish sending
            # whatever is queued, so we use an EndFrame().
            await task.queue_frame(EndFrame())

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot")
    parser.add_argument("-u", type=str, help="Room URL")
    parser.add_argument("-t", type=str, help="Token")
    config = parser.parse_args()

    asyncio.run(main(config.u, config.t))
