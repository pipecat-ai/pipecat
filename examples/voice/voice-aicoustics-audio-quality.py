#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimal audio-quality test bot for AICTytoAnalyzer.

Runs only the ai-coustics Tyto analysis model on the input audio so you can
watch its quality scores react to speech, silence, and background noise without
paying for STT/LLM/TTS API calls. The AICTytoAnalyzer is placed right after
``transport.input()`` so it scores the raw microphone signal (move it after an
AICFilter to score enhanced audio instead).

Logging:
    - INFO "audio quality" once per ``analysis_interval`` with the seven Tyto
      scores. ``risk_score`` (and ``noise`` / ``interfering_speech``) rising
      under poor conditions is the signal that the analyzer is working.
    - DEBUG init lines from AICTytoAnalyzer (run with LOGURU_LEVEL=DEBUG).

Required env vars:
    AIC_SDK_LICENSE    ai-coustics SDK license key
    Plus whatever credentials the chosen transport needs (DAILY_*, etc.)

Run:
    LOGURU_LEVEL=DEBUG uv run python examples/voice/voice-aicoustics-audio-quality.py daily
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.metrics.metrics import AICAudioQualityMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.audio.aic_tyto_analyzer import AICTytoAnalyzer
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


aic_tyto_analyzer = AICTytoAnalyzer(
    license_key=os.environ["AIC_SDK_LICENSE"],
    analysis_interval=1.0,
)


@aic_tyto_analyzer.event_handler("on_audio_analysis")
async def on_audio_analysis(_processor, scores: AICAudioQualityMetricsData) -> None:
    logger.info(
        "audio quality: "
        f"risk={scores.risk_score:.2f} noise={scores.noise:.2f} "
        f"interfering_speech={scores.interfering_speech:.2f} "
        f"media_speech={scores.media_speech:.2f} reverb={scores.speaker_reverb:.2f} "
        f"loudness={scores.speaker_loudness:.2f} packet_loss={scores.packet_loss:.2f}"
    )


transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    logger.info("Audio-quality test bot starting")
    pipeline = Pipeline([transport.input(), aic_tyto_analyzer])
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
