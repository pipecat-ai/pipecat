#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from dataclasses import dataclass

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.frames.frames import AudioRawFrame, EndFrame, OutputAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


@dataclass
class SilenceFrame(OutputAudioRawFrame):
    def __init__(
        self,
        *,
        sample_rate: int,
        duration: float,
    ):
        # Initialize the parent class with the silent frame's data
        super().__init__(
            audio=self.create_silent_audio_frame(sample_rate, 1, duration).audio,
            sample_rate=sample_rate,
            num_channels=1,
        )

    @staticmethod
    def create_silent_audio_frame(
        sample_rate: int, num_channels: int, duration: float
    ) -> AudioRawFrame:
        """Create an AudioRawFrame containing silence."""
        frame_size = num_channels * 2  # 2 bytes per sample for 16-bit audio
        total_frames = int(sample_rate * duration)
        total_bytes = total_frames * frame_size
        silent_audio = bytes(total_bytes)  # Create a byte array filled with zeros
        return AudioRawFrame(audio=silent_audio, sample_rate=sample_rate, num_channels=num_channels)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url, None, "Say One Thing", DailyParams(audio_out_enabled=True)
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        runner = PipelineRunner()

        task = PipelineTask(Pipeline([tts, transport.output()]))

        # Register an event handler so we can play the audio when we receive a specific message
        @transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            logger.debug(f"Received app message: {message} - {sender}")
            if "playable" not in message:
                return
            await task.queue_frames(
                [
                    SilenceFrame(
                        sample_rate=task.params.audio_out_sample_rate,
                        duration=0.5,
                    ),
                    TTSSpeakFrame(f"Hello there, how are you doing today ?"),
                    EndFrame(),
                ]
            )

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
