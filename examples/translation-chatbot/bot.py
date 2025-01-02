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

from pipecat.frames.frames import Frame, LLMMessagesFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.azure import AzureSTTService, AzureTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
    DailyTransportMessageFrame,
)

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
    def __init__(self, source_language, language):
        super().__init__()
        self._language = language
        self._source_language = source_language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            logger.debug(f"Translating {self._source_language}: {frame.text} to {self._language}")
            context = [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in {self._source_language}, and your task is to only translate it into {self._language}.",
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
            print(f"TranslationSubtitles: {frame.text}")
            message = {"event": "translation", "language": self._language, "text": frame.text}
            await self.push_frame(DailyTransportMessageFrame(message))

        await self.push_frame(frame)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Translator bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_audio_passthrough=True,
            ),
        )

        stt = AzureSTTService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            language="ja-JP",
        )

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            # Use Japanese Voice from Azure,
            # https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#text-to-speech
            voice="ja-JP-KeitaNeural",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        sa = SentenceAggregator()
        tp = TranslationProcessor(source_language="English", language="Japanese")
        lfra = LLMFullResponseAggregator()
        ts = TranslationSubtitles("japanese")

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                sa,
                tp,
                llm,
                lfra,
                ts,
                tts,
                transport.output(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info("First participant joined")

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
