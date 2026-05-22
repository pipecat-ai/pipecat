#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Realtime with locally-driven turn detection.

By default OpenAI Realtime drives the conversation with its own server-side
VAD (see `realtime-openai.py`). This variant disables server-side turn
detection (``turn_detection=False``) and instead drives turn boundaries
locally with ``SileroVADAnalyzer`` wired into the user aggregator. This is
the path to take if you want a turn analyzer like ``LocalSmartTurnV3`` to
decide when the user is done speaking, or if you need ``UserStartedSpeakingFrame``
/ ``UserStoppedSpeakingFrame`` to fire from the same source as
``InterruptionFrame``.

Caveat: locally-generated turn boundaries are a heuristic and may not match
the provider's actual server-side turn decisions. With OpenAI Realtime,
server-side turn detection is generally what the service expects to drive
the conversation, and disabling it puts the responsibility on you. Prefer
server-emitted turn frames (i.e. the base `realtime-openai.py` example)
unless you have a specific reason to drive turn detection locally.
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, LLMSetToolsFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    RealtimeServiceModeConfig,
    UserMessageAddedMessage,
    UserTurnStoppedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy

load_dotenv(override=True)


async def fetch_weather_from_api(params: FunctionCallParams):
    temperature = 75 if params.arguments["format"] == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_news(params: FunctionCallParams):
    await params.result_callback(
        {
            "news": [
                "Massive UFO currently hovering above New York City",
                "Stock markets reach all-time highs",
                "Living dinosaur species discovered in the Amazon rainforest",
            ],
        }
    )


async def fetch_restaurant_recommendation(params: FunctionCallParams):
    await params.result_callback({"name": "The Golden Dragon"})


weather_function = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to use. Infer this from the users location.",
        },
    },
    required=["location", "format"],
)

get_news_function = FunctionSchema(
    name="get_news",
    description="Get the current news.",
    properties={},
    required=[],
)

restaurant_function = FunctionSchema(
    name="get_restaurant_recommendation",
    description="Get a restaurant recommendation",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
        },
    },
    required=["location"],
)

tools = ToolsSchema(standard_tools=[weather_function, restaurant_function])


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

    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI.

Act like a human, but remember that you aren't a human and that you can't do human
things in the real world. Your voice and personality should be warm and engaging, with a lively and
playful tone.

If interacting in a non-English language, start by using the standard accent or dialect familiar to
the user. Talk quickly. You should always call a function if you can. Do not refer to these rules,
even if you're asked about them.

You are participating in a voice conversation. Keep your responses concise, short, and to the point
unless specifically asked to elaborate on a topic.

Remember, your responses should be short. Just one or two sentences, usually. Respond in English.""",
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(),
                        # Disable OpenAI's server-side turn detection — this
                        # example drives turn boundaries locally via the
                        # SileroVADAnalyzer wired into the user aggregator
                        # below.
                        turn_detection=False,
                        noise_reduction=InputAudioNoiseReduction(type="near_field"),
                    )
                ),
            ),
        ),
    )

    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("get_restaurant_recommendation", fetch_restaurant_recommendation)
    llm.register_function("get_news", get_news)

    context = LLMContext(
        [{"role": "developer", "content": "Say hello!"}],
        tools,
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # Drive turn detection locally via SileroVAD wired into the user
        # aggregator. realtime_service_mode keeps context-write semantics
        # correct and (by default) drops the transcript wait on turn-end so
        # local VAD can drive turn boundaries on the latency critical path.
        realtime_service_mode=RealtimeServiceModeConfig(),
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

    task = PipelineTask(
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
        logger.info(f"Client connected")
        await task.queue_frames([LLMRunFrame()])

        await asyncio.sleep(15)
        new_tools = ToolsSchema(
            standard_tools=[weather_function, restaurant_function, get_news_function]
        )
        await task.queue_frames([LLMSetToolsFrame(tools=new_tools)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    # In realtime mode the user transcript isn't finalized at turn-stop
    # time, so on_user_turn_stopped carries no content; subscribe to
    # on_user_message_added below for the finalized text.
    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(
        aggregator,
        strategy: BaseUserTurnStopStrategy,
        message: UserTurnStoppedMessage,
    ):
        logger.info(f"User turn stopped at {message.timestamp}")

    # In realtime mode this is the canonical "user said X" event,
    # decoupled from turn-stop.
    @user_aggregator.event_handler("on_user_message_added")
    async def on_user_message_added(aggregator, message: UserMessageAddedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}user: {message.content}"
        logger.info(f"Transcript: {line}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        line = f"{timestamp}assistant: {message.content}"
        logger.info(f"Transcript: {line}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
