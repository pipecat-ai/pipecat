#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcribe live audio with Inworld's realtime WebSocket STT API.

Add ``INWORLD_API_KEY`` to the repository's ``.env`` file and run:

    uv run python examples/transcription/transcription-inworld.py --transport webrtc

Open the displayed URL, allow microphone access, and speak. The example prints
interim and final transcriptions along with Inworld Voice Profile results.
Inworld manages voice activity and semantic end-of-turn detection. Set the
optional ``INWORLD_STT_LANGUAGE`` variable (for example, ``ru`` or ``pt``) to
provide a preferred language hint; Inworld may still detect another language.

For a conversational pipeline driven by Pipecat VAD, pass
``turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL``;
the service then disables Inworld VAD and sends ``endTurn`` on each local VAD stop.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.inworld.frames import InworldVoiceProfileFrame
from pipecat.services.inworld.stt import InworldRealtimeSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


class InworldTranscriptionLogger(FrameProcessor):
    """Print Inworld transcription and Voice Profile frames."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Print relevant results and pass every frame downstream."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Final: {frame.text}")
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"Interim: {frame.text}")
        elif isinstance(frame, InworldVoiceProfileFrame):
            print(f"Voice Profile: {frame.voice_profile.model_dump(exclude_defaults=True)}")

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
    """Run the Inworld transcription pipeline."""
    logger.info("Starting bot")

    stt = InworldRealtimeSTTService(
        api_key=os.environ["INWORLD_API_KEY"],
        settings=InworldRealtimeSTTService.Settings(
            language=os.getenv("INWORLD_STT_LANGUAGE") or None,
            enable_voice_profile=True,
            voice_profile_top_n=3,
            end_of_turn_confidence_threshold=0.7,
            min_end_of_turn_silence_when_confident=800,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            InworldTranscriptionLogger(),
            transport.output(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Create a transport and run the bot."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
