#
# Copyright (c) 2024, Daily
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

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.langchain import LangchainProcessor
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

load_dotenv(override=True)

try:
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError as e:
    logger.exception(
        "You need to `pip install langchain_openai` for this example. Also, be sure to set `OPENAI_API_KEY` in the environment variable."
    )
    raise Exception(f"Missing module: {e}")

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "Be nice and helpful. Answer very briefly and without special characters like `#` or `*`. Your response will be synthesized to voice and those characters will create unnatural sounds.",
                 ),
                ("human",
                 "{input}"),
            ])
        chain = prompt | llm
        lc = LangchainProcessor(chain)

        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                tma_in,  # User responses
                lc,  # Langchain
                tts,  # TTS
                transport.output(),  # Transport bot output
                tma_out,  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, allow_interruptions=True)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            # the `LLMMessagesFrame` will be picked up by the LangchainProcessor using
            # only the content of the last message to inject it in the prompt defined
            # above. So no role is required here.
            messages = [(
                {
                    "content": "Please briefly introduce yourself to the user."
                }
            )]
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
