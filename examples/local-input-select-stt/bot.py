#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys
from typing import Tuple

from dotenv import load_dotenv
from loguru import logger
from select_audio_device import AudioDevice, run_device_selector

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")


async def main(input_device: int, output_device: int):
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,
            input_device_index=input_device,
            output_device_index=output_device,
        )
    )

    stt = WhisperSTTService(device="cuda", model=Model.LARGE, no_speech_prob=0.3)

    tl = TranscriptionLogger()

    pipeline = Pipeline([transport.input(), stt, tl])

    task = PipelineTask(pipeline)

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    await asyncio.gather(runner.run(task))


if __name__ == "__main__":
    res: Tuple[AudioDevice, AudioDevice, int] = asyncio.run(
        run_device_selector()  # runs the textual app that allows to select input device
    )

    asyncio.run(main(res[0].index, res[1].index))
