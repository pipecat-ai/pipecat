#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


"""Log Rime interim and final transcripts using one standalone VAD processor.

Install ``pipecat-ai[rime-stt]`` plus the selected transport extra and set
``RIME_API_KEY``. Set ``RIME_STT_CONTRACT=append_fragments`` to accumulate
append fragments, or ``RIME_STT_SEGMENTED=1`` for completed-segment recognition.
``RIME_STT_LANGUAGE`` defaults to ``en``. Input audio is mono.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.rime.stt import RimeSegmentedSTTService, RimeSTTService
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
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
    ),
    "livekit": lambda: LiveKitParams(
        audio_in_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    if os.getenv("RIME_STT_SEGMENTED") == "1":
        stt = RimeSegmentedSTTService(
            api_key=os.environ["RIME_API_KEY"],
            settings=RimeSegmentedSTTService.Settings(
                language=os.getenv("RIME_STT_LANGUAGE", "en")
            ),
        )
    else:
        stt = RimeSTTService(
            api_key=os.environ["RIME_API_KEY"],
            settings=RimeSTTService.Settings(
                language=os.getenv("RIME_STT_LANGUAGE", "en"),
                output_contract=os.getenv("RIME_STT_CONTRACT", "revised_hypotheses"),
            ),
        )

    vad_processor = VADProcessor(vad_analyzer=SileroVADAnalyzer())

    pipeline = Pipeline([transport.input(), vad_processor, stt, transport.output()])

    worker = PipelineWorker(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
        observers=[TranscriptionLogObserver()],
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
