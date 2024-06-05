#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

import tkinter as tk

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.tk import TkLocalTransport
from pipecat.transports.services.daily import DailyParams, DailyTransport

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url, token):
    tk_root = tk.Tk()
    tk_root.title("Local Mirror")

    daily_transport = DailyTransport(room_url, token, "Test", DailyParams(audio_in_enabled=True))

    tk_transport = TkLocalTransport(
        tk_root,
        TransportParams(
            audio_out_enabled=True,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=1280,
            camera_out_height=720))

    @daily_transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_video(participant["id"])

    pipeline = Pipeline([daily_transport.input(), tk_transport.output()])

    task = PipelineTask(pipeline)

    async def run_tk():
        while not task.has_finished():
            tk_root.update()
            tk_root.update_idletasks()
            await asyncio.sleep(0.1)

    runner = PipelineRunner()

    await asyncio.gather(runner.run(task), run_tk())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))
