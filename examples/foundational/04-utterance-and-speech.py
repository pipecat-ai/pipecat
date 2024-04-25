#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

from pipecat.pipeline.merge_pipeline import SequentialMergePipeline
from pipecat.pipeline.pipeline import Pipeline

from pipecat.frames.frames import EndPipeFrame, LLMMessagesFrame, TextFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.services.azure import AzureLLMService, AzureTTSService
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.transport_services import TransportServiceOutput
from pipecat.services.transports.daily_transport import DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(room_url, None, "Static And Dynamic Speech")

        meeting = TransportServiceOutput(transport, mic_enabled=True)

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )
        azure_tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
        )

        elevenlabs_tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        messages = [{"role": "system",
                     "content": "tell the user a joke about llamas"}]

        # Start a task to run the LLM to create a joke, and convert the LLM
        # output to audio frames. This task will run in parallel with generating
        # and speaking the audio for static text, so there's no delay to speak
        # the LLM response.
        llm_pipeline = Pipeline([llm, elevenlabs_tts])
        llm_task = PipelineTask(llm_pipeline)
        await llm_task.queue_frames([LLMMessagesFrame(messages), EndPipeFrame()])

        simple_tts_pipeline = Pipeline([azure_tts])
        await simple_tts_pipeline.queue_frames(
            [
                TextFrame("My friend the LLM is going to tell a joke about llamas."),
                EndPipeFrame(),
            ]
        )

        merge_pipeline = SequentialMergePipeline(
            [simple_tts_pipeline, llm_pipeline])

        await asyncio.gather(
            transport.run(merge_pipeline),
            simple_tts_pipeline.run_pipeline(),
            llm_pipeline.run_pipeline(),
        )


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url))
