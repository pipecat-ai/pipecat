#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaLiveOptions, CartesiaSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")
            # Access word-level timestamps if available
            if frame.result and "words" in frame.result:
                words = frame.result["words"]
                print(f"  Word-level timestamps ({len(words)} words):")
                for word_data in words:
                    word = word_data.get("word", "")
                    start = word_data.get("start", 0)
                    end = word_data.get("end", 0)
                    print(f"    '{word}' [{start:.3f}s - {end:.3f}s]")

        # Push all frames through
        await self.push_frame(frame, direction)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Configure Cartesia STT with word-level timestamps enabled (default is True)
    live_options = CartesiaLiveOptions(
        model="ink-whisper",
        include_timestamps=True,  # Enable word-level timestamps
    )

    stt = CartesiaSTTService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        live_options=live_options,
    )

    tl = TranscriptionLogger()

    pipeline = Pipeline([transport.input(), stt, tl])

    task = PipelineTask(
        pipeline,
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
