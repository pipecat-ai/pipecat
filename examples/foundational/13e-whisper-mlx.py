#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys
import time

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame, UserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


STOP_SECS = 2.0


class TranscriptionLogger(FrameProcessor):
    """Measures transcription latency.

    Uses the (intentionally) long STOP_SECS parameter to give the transcription time to finish,
    then outputs the timing between when the VAD first classified audio input as not-speech and
    the delivery of the last transcription frame.
    """

    def __init__(self):
        super().__init__()
        self._last_transcription_time = time.time()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug(
                f"Transcription latency: {(STOP_SECS - (time.time() - self._last_transcription_time)):.2f}"
            )

        if isinstance(frame, TranscriptionFrame):
            self._last_transcription_time = time.time()


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Transcription bot",
            DailyParams(
                audio_in_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECS)),
                vad_audio_passthrough=True,
            ),
        )

        stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO)

        tl = TranscriptionLogger()

        pipeline = Pipeline([transport.input(), stt, tl])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                report_only_initial_ttfb=False,
            ),
        )

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
