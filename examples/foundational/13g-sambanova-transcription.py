#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import time

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TranscriptionFrame, UserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.sambanova.stt import SambaNovaSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


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

        # Push all frames through
        await self.push_frame(frame, direction)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECS)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECS)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECS)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = SambaNovaSTTService(
        model="Whisper-Large-v3",
        api_key=os.getenv("SAMBANOVA_API_KEY"),
    )

    tl = TranscriptionLogger()

    pipeline = Pipeline([transport.input(), stt, tl])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
