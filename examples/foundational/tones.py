#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
import numpy as np
from dotenv import load_dotenv
from dtmf import detect, generate, model, parse
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    LLMMessagesFrame,
    OutputAudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class DebugProcessor(FrameProcessor):
    def __init__(self, name, **kwargs):
        self._name = name
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not (
            isinstance(frame, InputAudioRawFrame)
            or isinstance(frame, BotSpeakingFrame)
            or isinstance(frame, UserStoppedSpeakingFrame)
            or isinstance(frame, TTSAudioRawFrame)
            or isinstance(frame, TextFrame)
        ):
            logger.debug(f"--- DebugProcessor {self._name}: {frame} {direction}")
        await self.push_frame(frame, direction)


class DTMFProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        tones = model.String(
            [
                model.Tone("1"),
                model.Tone("2"),
                model.Tone("3"),
                model.Tone("4"),
                model.Pause(),
                model.Tone("5"),
                model.Tone("6"),
                model.Tone("7"),
                model.Tone("8"),
                model.Tone("9"),
            ]
        )

        tone_audio = generate(tones)

        # Convert the generated audio to a numpy array (assuming the generate function returns an iterable of floats)
        audio_data = np.array(list(tone_audio), dtype=np.float32)

        # Create an AudioRawFrame with the audio data
        audio_frame = OutputAudioRawFrame(audio_data, sample_rate=8000, num_channels=1)

        await self.push_frame(audio_frame)


async def main():
    print(detect, generate, parse)
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

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

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        dtmf = DTMFProcessor()
        dp = DebugProcessor("dp")

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                dtmf,
                dp,
                # llm,
                # tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
