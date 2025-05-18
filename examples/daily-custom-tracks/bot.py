#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys

import aiohttp
from loguru import logger
from runner import configure

from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.services.daily import DailyParams, DailyTransport

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class CustomTrackMirrorProcessor(FrameProcessor):
    def __init__(self, transport_destination: str, **kwargs):
        super().__init__(**kwargs)
        self._transport_destination = transport_destination

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame) and frame.transport_source:
            output_frame = OutputAudioRawFrame(
                audio=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            output_frame.transport_destination = self._transport_destination
            await self.push_frame(output_frame)
        else:
            await self.push_frame(frame, direction)


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Custom tracks mirror",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                microphone_out_enabled=False,  # Disable since we just use custom tracks
                audio_out_destinations=["pipecat-mirror"],
            ),
        )

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                CustomTrackMirrorProcessor("pipecat-mirror"),
                transport.output(),  # Transport bot output
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_audio(participant["id"], audio_source="pipecat")

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
