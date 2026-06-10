#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Repro for cloud recording corruption with odd-height bot video.

Publishes an animated 300x169 video track (odd height, not a multiple of 16)
into a Daily room, starts a cloud recording for RECORD_SECONDS, then stops and
leaves. With the new GST VCS render pipeline the recorded MP4 shows red border
artifacts and contrast shifts; the live call looks fine.

Run:

    export DAILY_API_KEY=...   # a domain WITHOUT enable_legacy_compositor=true

    python bot.py                                     # 300x169, expect corruption
    VIDEO_WIDTH=320 VIDEO_HEIGHT=176 python bot.py    # control, expect clean

Then download the recording from the Daily dashboard and compare.
"""

import asyncio
import os
import time

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image, ImageDraw

from pipecat.frames.frames import OutputImageRawFrame, SpriteFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.daily import configure
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.daily.utils import DailyMeetingTokenProperties, DailyRoomProperties
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

WIDTH = int(os.getenv("VIDEO_WIDTH", "300"))
HEIGHT = int(os.getenv("VIDEO_HEIGHT", "169"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "30"))


def make_frames(num: int = 16) -> list[OutputImageRawFrame]:
    """Generate animation frames: gray gradient, white border, moving green box."""
    frames = []
    for i in range(num):
        img = Image.new("RGB", (WIDTH, HEIGHT))
        draw = ImageDraw.Draw(img)
        for x in range(WIDTH):
            v = int(255 * x / WIDTH)
            draw.line([(x, 0), (x, HEIGHT)], fill=(v, v, v))
        draw.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], outline=(255, 255, 255), width=2)
        box_x = int((WIDTH - 40) * i / (num - 1))
        draw.rectangle([box_x, HEIGHT // 2 - 20, box_x + 40, HEIGHT // 2 + 20], fill=(0, 255, 0))
        frames.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.mode))
    return frames


async def main():
    async with aiohttp.ClientSession() as session:
        room_url, token = await configure(
            session,
            room_properties=DailyRoomProperties(
                enable_recording="cloud",
                exp=time.time() + 3600,
            ),
            token_properties=DailyMeetingTokenProperties(is_owner=True),
        )

        transport = DailyTransport(
            room_url,
            token,
            f"Repro bot {WIDTH}x{HEIGHT}",
            DailyParams(
                video_out_enabled=True,
                video_out_width=WIDTH,
                video_out_height=HEIGHT,
            ),
        )

        worker = PipelineWorker(
            Pipeline([transport.output()]),
            params=PipelineParams(),
        )

        await worker.queue_frame(SpriteFrame(images=make_frames()))

        async def record_then_leave():
            # Give the video track a moment to start flowing.
            await asyncio.sleep(3)
            stream_id, error = await transport.start_recording()
            if error:
                logger.error(f"Could not start recording: {error}")
                await worker.cancel()
                return
            logger.info(f"Recording started (stream_id={stream_id}), capturing {RECORD_SECONDS}s")
            await asyncio.sleep(RECORD_SECONDS)
            await transport.stop_recording(stream_id)
            logger.info("Recording stopped. Download it from the Daily dashboard and check it.")
            await asyncio.sleep(2)
            await worker.cancel()

        @transport.event_handler("on_joined")
        async def on_joined(transport, data):
            logger.info(f"Joined {room_url}")
            logger.info(
                f"Publishing {WIDTH}x{HEIGHT}. Open the room URL to confirm live looks clean."
            )
            asyncio.create_task(record_then_leave())

        runner = WorkerRunner()
        await runner.add_workers(worker)
        await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
