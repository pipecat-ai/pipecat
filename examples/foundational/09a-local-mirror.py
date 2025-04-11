#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import tkinter as tk

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.tk import TkLocalTransport, TkTransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


class MirrorProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
            )
        elif isinstance(frame, InputImageRawFrame):
            await self.push_frame(
                OutputImageRawFrame(image=frame.image, size=frame.size, format=frame.format)
            )
        else:
            await self.push_frame(frame, direction)


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    p2p_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_in_enabled=True,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=1280,
            camera_out_height=720,
        ),
    )

    tk_root = tk.Tk()
    tk_root.title("Local Mirror")

    tk_transport = TkLocalTransport(
        tk_root,
        TkTransportParams(
            audio_out_enabled=True,
            camera_out_enabled=True,
            camera_out_is_live=True,
            camera_out_width=1280,
            camera_out_height=720,
        ),
    )

    @p2p_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")

    pipeline = Pipeline([p2p_transport.input(), MirrorProcessor(), tk_transport.output()])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(),
    )

    async def run_tk():
        while not task.has_finished():
            tk_root.update()
            tk_root.update_idletasks()
            await asyncio.sleep(0.1)

    runner = PipelineRunner(handle_sigint=False)

    await asyncio.gather(runner.run(task), run_tk())


if __name__ == "__main__":
    from run import main

    main()
