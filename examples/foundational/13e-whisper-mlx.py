#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
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
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

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


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO)

    tl = TranscriptionLogger()

    pipeline = Pipeline([transport.input(), stt, tl])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
