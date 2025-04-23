#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.kokoro import KokoroTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))

    tts = KokoroTTSService(
        model_path="assets/kokoro-v1.0.onnx",
        voices_path="assets/voices-v1.0.bin",
        voice_id="am_fenrir",
    )

    pipeline = Pipeline([tts, transport.output()])

    task = PipelineTask(pipeline)

    async def say_something():
        await asyncio.sleep(1)
        await task.queue_frames([TTSSpeakFrame("Hello there, how is it going!"), TTSSpeakFrame("I'm Marmik, nice to meet you!"), EndFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    await asyncio.gather(runner.run(task), say_something())


if __name__ == "__main__":
    asyncio.run(main())
