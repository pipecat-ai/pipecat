#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Manual smoke test for SmallestLightningV4TTSService against the real API.

Requires:
- SMALLEST_API_KEY for a Lightning v4 beta-enabled account.
- SMALLEST_LIGHTNING_V4_VOICE, fetched from the get_voices endpoint (the
  catalogue is still changing during the beta, so don't hardcode a voice):

    curl https://api.smallest.ai/waves/v1/lightning-v4/get_voices
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.services.smallest.tts_v4 import SmallestLightningV4TTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))

    tts = SmallestLightningV4TTSService(
        api_key=os.environ["SMALLEST_API_KEY"],
        settings=SmallestLightningV4TTSService.Settings(
            voice=os.environ["SMALLEST_LIGHTNING_V4_VOICE"],
        ),
    )

    pipeline = Pipeline([tts, transport.output()])

    worker = PipelineWorker(pipeline, processor_unusable_policy=ProcessorUnusablePolicy.END)

    runner = WorkerRunner(handle_sigint=False if sys.platform == "win32" else True)

    await runner.add_workers(worker)

    async def say_something():
        await asyncio.sleep(1)
        await worker.queue_frames(
            [TTSSpeakFrame("Hello there, this is Lightning v4."), EndFrame()]
        )

    await asyncio.gather(runner.run(), say_something())


if __name__ == "__main__":
    asyncio.run(main())
