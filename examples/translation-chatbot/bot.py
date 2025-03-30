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

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMMessagesFrame,
    TranscriptionFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

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
        self._out_language = out_language
        self._in_language = in_language

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and translate text frames.

        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction of the frame.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.debug(f"Translating {self._in_language}: {frame.text} to {self._out_language}")
            context = [
                {
                    "role": "system",
                    "content": f"You will be provided with a sentence in {self._in_language}, and your task is to only translate it into {self._out_language}.",
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
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="34dbb662-8e98-413c-a1ef-1a3407675fe7",  # Spanish Narrator Man
            model="sonic-2",
        )

        in_language = "English"
        out_language = "Spanish"

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        tp = TranslationProcessor(in_language=in_language, out_language=out_language)

        transcript = TranscriptProcessor()
        transcript_handler = TranscriptHandler(in_language=in_language, out_language=out_language)

        # Register event handler for transcript updates
        @transcript.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            await transcript_handler.on_transcript_update(processor, frame)

        rtvi = RTVIProcessor()

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                stt,
                transcript.user(),  # User transcripts
                tp,
                llm,
                tts,
                transport.output(),
                transcript.assistant(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=False,  # We don't want to interrupt the translator bot
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info("First participant joined")

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
