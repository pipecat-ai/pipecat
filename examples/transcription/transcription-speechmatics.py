#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"Interim transcription: {frame.text}")
        # Push all frames through
        await self.push_frame(frame, direction)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(audio_in_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run example using Speechmatics STT.

    This example will use diarization within our STT service and output the words spoken by
    each individual speaker and wrap them with XML tags.

    If you do not wish to use diarization, then set the `enable_diarization` parameter
    to `False` or omit it altogether. The `text_format` will only be used if diarization is enabled.

    By default, this example will use our ENHANCED operating point, which is optimized for
    high accuracy. You can change this by setting the `operating_point` parameter to a different
    value.

    For more information on operating points, see the Speechmatics documentation:
    https://docs.speechmatics.com/rt-api-ref
    """
    logger.info(f"Starting bot")

    stt = SpeechmaticsSTTService(
        api_key=os.environ["SPEECHMATICS_API_KEY"],
        settings=SpeechmaticsSTTService.Settings(
            language=Language.EN,
            speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
        ),
    )

    tl = TranscriptionLogger()

    vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    pipeline = Pipeline([transport.input(), vad_processor, stt, tl, transport.output()])

    worker = PipelineWorker(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
