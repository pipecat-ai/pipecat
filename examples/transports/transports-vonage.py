#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example of using OpenAI Realtime voice LLM service with Vonage Video Connector transport."""

import asyncio
import os
import sys
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.vonage import configure
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.vonage.video_connector import (
    VonageVideoConnectorTransport,
    VonageVideoConnectorTransportParams,
)
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main() -> None:
    """Main entry point for the OpenAI Realtime vonage video connector example."""
    (application_id, session_id, token) = await configure()

    transport = VonageVideoConnectorTransport(
        application_id,
        session_id,
        token,
        VonageVideoConnectorTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            publisher_name="Bot",
        ),
    )

    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI.

Act like a human, but remember that you aren't a human and that you can't do human
things in the real world. Your voice and personality should be warm and engaging, with a lively and
playful tone.

If interacting in a non-English language, start by using the standard accent or dialect familiar to
the user. Talk quickly.

You are participating in a voice conversation. Keep your responses concise, short, and to the point
unless specifically asked to elaborate on a topic.

Remember, your responses should be short. Just one or two sentences, usually. Respond in English.""",
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(),
                        turn_detection=SemanticTurnDetection(),
                        noise_reduction=InputAudioNoiseReduction(type="near_field"),
                    )
                ),
            ),
        ),
    )

    context = LLMContext(
        [{"role": "developer", "content": "Say hello!"}],
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[TranscriptionLogObserver()],
    )

    event_handler: Callable[[str], Callable[[Any], Any]] = transport.event_handler

    @event_handler("on_client_connected")
    async def on_client_connected(transport: VonageVideoConnectorTransport, client: object) -> None:
        logger.info("Client connected")
        await worker.queue_frames([LLMRunFrame()])

    runner = WorkerRunner()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
