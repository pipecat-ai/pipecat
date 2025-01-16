#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from typing import List

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    TextFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
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
    """A processor that translates text frames from a source language to a target language."""

    def __init__(self, in_language, out_language):
        """Initialize the TranslationProcessor with source and target languages.

        Args:
            in_language (str): The language of the input text.
            out_language (str): The language to translate the text into.
        """
        super().__init__()
        self._language = out_language
        self._in_language = in_language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and translate text frames.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction of the frame.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            logger.debug(f"Translating {self._in_language}: {frame.text} to {self._language}")
            context = [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in {self._in_language}, and your task is to only translate it into {self._language}.",
                },
                {"role": "user", "content": frame.text},
            ]
            await self.push_frame(LLMMessagesFrame(context))
        else:
            await self.push_frame(frame)


class TranscriptHandler:
    """Simple handler to demonstrate transcript processing.

    Maintains a list of conversation messages and logs them with timestamps.
    """

    def __init__(self, in_language="English", out_language="Spanish"):
        """Initialize the TranscriptHandler with an empty list of messages."""
        self.messages: List[TranscriptionMessage] = []
        self.in_language = in_language
        self.out_language = out_language

    async def on_transcript_update(
        self, processor: TranscriptProcessor, frame: TranscriptionUpdateFrame
    ):
        """Handle new transcript messages.

        Args:
            processor: The TranscriptProcessor that emitted the update
            frame: TranscriptionUpdateFrame containing new messages
        """
        self.messages.extend(frame.messages)

        # Log the new messages
        logger.info("New transcript messages:")
        for msg in frame.messages:
            timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
            message = {
                "event": "translation",
                "timestamp": msg.timestamp,
                "role": msg.role,
                "language": self.out_language if msg.role == "assistant" else self.in_language,
                "text": msg.content,
            }
            await processor.push_frame(DailyTransportMessageFrame(message))
            logger.info(f"{timestamp}{msg.role}: {msg.content}")


async def main():
    """Main function to set up and run the translation chatbot pipeline."""
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Translator",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_audio_passthrough=True,
            ),
        )

        stt = AzureSTTService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            language="en-US",
        )

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            # Use Spanish voice from Azure,
            # https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/language-support#text-to-speech
            voice="es-ES-AlvaroNeural",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        tp = TranslationProcessor(in_language="English", out_language="Spanish")
        lfra = LLMFullResponseAggregator()

        transcript = TranscriptProcessor()
        transcript_handler = TranscriptHandler(in_language="English", out_language="Spanish")

        # Register event handler for transcript updates
        @transcript.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            await transcript_handler.on_transcript_update(processor, frame)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                transcript.user(),  # User transcripts
                tp,
                llm,
                lfra,
                tts,
                context_aggregator.assistant(),
                transcript.assistant(),  # Assistant transcripts
                transport.output(),
            ]
        )

        task = PipelineTask(pipeline)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info("First participant joined")

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
