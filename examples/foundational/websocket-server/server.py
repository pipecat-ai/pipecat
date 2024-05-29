#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

from loguru import logger
from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.whisper import WhisperSTTService
from pipecat.transports.network.websocket_server import WebsocketServerTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class WhisperTranscriber(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TranscriptionFrame):
            print(f"Transcribed: {frame.text}")
        else:
            await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        transport = WebsocketServerTransport()

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        pipeline = Pipeline([
            transport.input(),
            WhisperSTTService(),
            WhisperTranscriber(),
            tts,
            transport.output(),
        ])

        task = PipelineTask(pipeline)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await task.queue_frame(TextFrame("Hello there!"))

        runner = PipelineRunner()

        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
