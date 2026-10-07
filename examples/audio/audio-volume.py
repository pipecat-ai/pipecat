#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import EndFrame, TTSSpeakFrame, VolumeFrame, VolumeGainFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "livekit": lambda: LiveKitParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    tts = DeepgramTTSService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramTTSService.Settings(
            voice="aura-2-andromeda-en",
        ),
    )

    worker = PipelineWorker(
        Pipeline([tts, transport.output()]),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await worker.queue_frames(
            [
                TTSSpeakFrame("This is my normal volume."),
                # A gain changes the volume for the audio between two gain
                # frames, and resetting it to 1.0 restores the volume.
                VolumeGainFrame(gain=0.1),
                TTSSpeakFrame("This sentence plays at a tenth of the volume."),
                VolumeGainFrame(gain=1.0),
                TTSSpeakFrame("And this one is back to normal."),
                VolumeGainFrame(gain=3.0),
                TTSSpeakFrame("This one plays three times as loud."),
                VolumeGainFrame(gain=1.0),
                # The volume holds until the next volume frame.
                VolumeFrame(volume=0.2),
                TTSSpeakFrame("Now my whole volume is turned down."),
                TTSSpeakFrame("It stays down until the volume changes again."),
                VolumeFrame(volume=1.0),
                TTSSpeakFrame("And I'm back to my normal volume. Goodbye!"),
                EndFrame(),
            ]
        )

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
