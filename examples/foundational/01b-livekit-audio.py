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

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.runner.livekit import configure
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    (url, token, room_name) = await configure()

    transport = LiveKitTransport(
        url=url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(audio_out_enabled=True),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    runner = PipelineRunner()

    task = PipelineTask(Pipeline([tts, transport.output()]))

    # Register an event handler so we can play the audio when the
    # participant joins.
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        await asyncio.sleep(1)
        await task.queue_frame(
            TTSSpeakFrame(
                "Hello there! How are you doing today? Would you like to talk about the weather?"
            )
        )

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
