#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live with locally-driven turn detection.

By default Gemini Live drives the conversation with its own server-side VAD
(see `realtime-gemini-live.py`). That setup doesn't surface
``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame``, so pipeline
processors that depend on those frames (RTVI client speech events,
``TurnTrackingObserver``, ``AudioBufferProcessor`` turn recording,
``UserIdleController``, user mute strategies, voicemail detector) don't
activate.

This variant disables Gemini Live's server-side VAD
(``GeminiVADParams(disabled=True)``) and instead drives turn boundaries
locally with ``SileroVADAnalyzer`` wired into the user aggregator. Use this
variant if you need those downstream processors, or if you want a turn
analyzer like ``LocalSmartTurnV3`` to decide when the user is done speaking.

Caveat: locally-generated turn boundaries are a heuristic and may not match
the provider's actual server-side turn decisions, which is what really
drives the conversation. The two can drift apart in subtle, hard-to-debug
ways, especially around interruptions and overlapping speech. Prefer
server-emitted turn frames (i.e. the base `realtime-gemini-live.py` example)
unless you have a specific reason to drive turn detection locally.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    RealtimeServiceModeConfig,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiVADParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GeminiLiveLLMService.Settings(
            voice="Aoede",  # Puck, Charon, Kore, Fenrir, Aoede
            vad=GeminiVADParams(disabled=True),
        ),
        # inference_on_context_initialization=False,
    )

    context = LLMContext(
        [
            {
                "role": "user",
                "content": "Say hello. Then ask if I want to hear a joke.",
            },
        ],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=RealtimeServiceModeConfig(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    # The *_message_added events fire when messages are written to context
    # and carry the finalized content. In realtime mode the turn-stopped
    # events fire before the message text is finalized.
    @user_aggregator.event_handler("on_user_message_added")
    async def on_user_message_added(aggregator, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_message_added")
    async def on_assistant_message_added(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

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
