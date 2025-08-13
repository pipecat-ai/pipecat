#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TranslationFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.gladia.config import (
    GladiaInputParams,
    LanguageConfig,
    RealtimeProcessingConfig,
    TranslationConfig,
)
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription ({frame.language}): {frame.text}")
        elif isinstance(frame, TranslationFrame):
            print(f"Translation ({frame.language}): {frame.text}")


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

    stt = GladiaSTTService(
        api_key=os.getenv("GLADIA_API_KEY"),
        region=os.getenv("GLADIA_REGION"),
        params=GladiaInputParams(
            language_config=LanguageConfig(
                languages=[Language.EN],  # Input in English
                code_switching=False,
            ),
            realtime_processing=RealtimeProcessingConfig(
                translation=True,  # Enable translation
                translation_config=TranslationConfig(
                    target_languages=[Language.ES],  # Translate to Spanish
                    model="enhanced",  # Use the enhanced translation model
                ),
            ),
        ),
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
