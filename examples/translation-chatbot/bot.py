#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

from pipecat.frames.frames import Frame, LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.azure import AzureTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTranscriptionSettings,
    DailyTransport,
    DailyTransportMessageFrame,
)

from runner import configure

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


"""
This example looks a bit different than the chatbot example, because it isn't waiting on the user to stop talking to start translating.
It also isn't saving what the user or bot says into the context object for use in subsequent interactions.
"""


# We need to use a custom service here to yield LLM frames without saving
# any context
class TranslationProcessor(FrameProcessor):
    def __init__(self, language):
        super().__init__()
        self._language = language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            context = [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in English, and your task is to translate it into {self._language}.",
                },
                {"role": "user", "content": frame.text},
            ]
            await self.push_frame(LLMMessagesFrame(context))
        else:
            await self.push_frame(frame)


class TranslationSubtitles(FrameProcessor):
    def __init__(self, language):
        super().__init__()
        self._language = language

    #
    # This doesn't do anything unless the receiver recognizes the message being
    # sent. For example, in this case, we are sending a message to the transport
    # so an application running at the other end of the transport could display
    # subtitles.
    #
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            message = {"language": self._language, "text": frame.text}
            await self.push_frame(DailyTransportMessageFrame(message))

        await self.push_frame(frame)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Translator",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                transcription_settings=DailyTranscriptionSettings(extra={"interim_results": False}),
            ),
        )

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="es-ES-AlvaroNeural",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        sa = SentenceAggregator()
        tp = TranslationProcessor("Spanish")
        lfra = LLMFullResponseAggregator()
        ts = TranslationSubtitles("spanish")

        pipeline = Pipeline([transport.input(), sa, tp, llm, lfra, ts, tts, transport.output()])

        task = PipelineTask(pipeline)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
