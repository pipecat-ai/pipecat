#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Live interpreter: the user's speech is transcribed, translated and spoken back
in the target language by Palabra's Speech-to-Speech Translation API. No LLM or
TTS service is needed."""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import Frame, TranscriptionFrame, TranslationFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.palabra.translation import PalabraTranslationService
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
            print(f"Transcription ({frame.language}): {frame.text}")
        elif isinstance(frame, TranslationFrame):
            print(f"Translation ({frame.language}): {frame.text}")

        # Push all frames through
        await self.push_frame(frame, direction)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    translator = PalabraTranslationService(
        api_key=os.environ["PALABRA_API_KEY"],
        settings=PalabraTranslationService.Settings(
            language=Language.EN,
            target_languages=[Language.ES],
            # Several target languages need a transport with several audio
            # tracks (Daily, LiveKit): register the track names in the
            # transport's ``audio_out_destinations`` and map each language to
            # its track, e.g. ``audio_destinations={Language.ES: "es",
            # Language.FR: "fr"}``.
            # Uncomment to speak the translation in the user's own voice.
            # voice_cloning=True,
        ),
    )

    @translator.event_handler("on_task_ready")
    async def on_task_ready(service):
        logger.info("Palabra translation task is running, start speaking")

    tl = TranscriptionLogger()

    # The user keeps talking while the translation plays, so the pipeline has
    # no user turn strategies and no interruptions.
    pipeline = Pipeline([transport.input(), translator, tl, transport.output()])

    worker = PipelineWorker(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
