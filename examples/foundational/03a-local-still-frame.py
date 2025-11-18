#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import tkinter as tk

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.fal.image import FalImageGenService
from pipecat.transports.local.tk import TkLocalTransport, TkTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        tk_root = tk.Tk()
        tk_root.title("Picasso Cat")

        transport = TkLocalTransport(
            tk_root,
            TkTransportParams(video_out_enabled=True, video_out_width=1024, video_out_height=1024),
        )

        imagegen = FalImageGenService(
            params=FalImageGenService.InputParams(image_size="square_hd"),
            aiohttp_session=session,
            key=os.getenv("FAL_KEY"),
        )

        pipeline = Pipeline([imagegen, transport.output()])

        task = PipelineTask(pipeline)
        await task.queue_frames([TextFrame("a cat in the style of picasso")])

        runner = PipelineRunner()

        async def run_tk():
            while not task.has_finished():
                tk_root.update()
                tk_root.update_idletasks()
                await asyncio.sleep(0.1)

        await asyncio.gather(runner.run(task), run_tk())


if __name__ == "__main__":
    asyncio.run(main())
