#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Grok Realtime with locally-driven turn detection.

By default Grok Realtime drives the conversation with its own server-side
VAD (see `realtime-grok.py`). This variant disables server-side turn
detection (``turn_detection=None``, the "manual" mode in Grok's session
properties) and instead drives turn boundaries locally with
``SileroVADAnalyzer`` wired into the user aggregator. Use this variant if
you want a turn analyzer like ``LocalSmartTurnV3`` to decide when the user
is done speaking, or if you need ``UserStartedSpeakingFrame`` /
``UserStoppedSpeakingFrame`` to fire from the same source as
``InterruptionFrame``.

Caveat: locally-generated turn boundaries are a heuristic and may not match
the provider's actual server-side turn decisions. Prefer server-emitted
turn frames (i.e. the base `realtime-grok.py` example) unless you have a
specific reason to drive turn detection locally.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnMessageAddedMessage,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.xai.realtime.events import SessionProperties
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_current_time(params: FunctionCallParams):
    """Get the current time."""
    await params.result_callback(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timezone": "local",
        }
    )


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback(
        {
            "name": "The Golden Dragon",
            "cuisine": "Chinese",
            "location": location,
            "rating": 4.5,
        }
    )


transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
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
    logger.info("Starting Grok Voice Agent bot")

    session_properties = SessionProperties(
        voice="Ara",
        # Disable Grok's server-side turn detection (manual mode). This
        # example drives turn boundaries locally via the SileroVADAnalyzer
        # wired into the user aggregator below.
        turn_detection=None,
    )

    llm = GrokRealtimeLLMService(
        api_key=os.environ["XAI_API_KEY"],
        settings=GrokRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI assistant powered by Grok.

    You have access to several tools:
    - Weather information
    - Current time
    - Restaurant recommendations
    - Web search (built-in)
    - X/Twitter search (built-in)

    Your voice and personality should be warm and engaging. Keep your responses
    concise and conversational since this is a voice interaction.

    If the user asks about current events or news, use web search.
    If they ask about what people are saying on social media, use X search.

    Always be helpful and proactive in offering assistance.""",
            session_properties=session_properties,
        ),
    )

    context = LLMContext(
        [{"role": "developer", "content": "Say hello and introduce yourself!"}],
        [get_current_weather, get_current_time, get_restaurant_recommendation],
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # Drive turn detection locally via SileroVAD wired into the user
        # aggregator. realtime_service_mode keeps context-write semantics
        # correct and (by default) drops the transcript wait on turn-end so
        # local VAD can drive turn boundaries on the latency critical path.
        realtime_service_mode=True,
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    # In realtime mode the user transcript isn't finalized at turn-stop
    # time, so on_user_turn_stopped carries no content; subscribe to
    # on_user_turn_message_added below for the finalized text.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator,
        strategy: BaseUserTurnStopStrategy,
        message: UserTurnStoppedMessage,
    ):
        logger.info(f"User turn stopped at {message.timestamp}")

    # In realtime mode this is the canonical "user said X" event,
    # decoupled from turn-stop.
    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message: UserTurnMessageAddedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
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
