#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld Realtime with locally-driven turn detection.

By default Inworld Realtime drives the conversation with its own
server-side semantic VAD (see `realtime-inworld.py`). This variant
disables server-side turn detection (``turn_detection=None``, the
"manual" mode in Inworld's session properties) and instead drives turn
boundaries locally with ``SileroVADAnalyzer`` wired into the user
aggregator. Use this variant if you want a turn analyzer like
``LocalSmartTurnV3`` to decide when the user is done speaking, or if you
need ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` to fire
from the same source as ``InterruptionFrame``.

Caveat: locally-generated turn boundaries are a heuristic and may not
match the provider's actual server-side turn decisions. Prefer
server-emitted turn frames (i.e. the base `realtime-inworld.py` example)
unless you have a specific reason to drive turn detection locally.
"""

import os
import random
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import (
    TranscriptionLogObserver,
)
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
from pipecat.services.inworld.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputTranscription,
    PCMAudioFormat,
    SessionProperties,
)
from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import BaseUserTurnStopStrategy

load_dotenv(override=True)


async def fetch_weather_from_api(params: FunctionCallParams):
    temperature = (
        random.randint(60, 85)
        if params.arguments["format"] == "fahrenheit"
        else random.randint(15, 30)
    )
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


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
            "description": "The temperature unit to use.",
        },
    },
    required=["location", "format"],
)

tools = ToolsSchema(standard_tools=[weather_function])


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
    logger.info("Starting Inworld Realtime bot (local VAD)")

    model = "openai/gpt-4.1-mini"
    voice = "Sarah"
    tts_model = "inworld-tts-2"
    stt_model = "assemblyai/u3-rt-pro"

    # Setting session_properties here replaces Inworld's defaults wholesale,
    # so we provide a complete SessionProperties — with turn_detection=None
    # (manual mode) so local VAD drives turn boundaries instead.
    session_properties = SessionProperties(
        model=model,
        output_modalities=["audio", "text"],
        audio=AudioConfiguration(
            input=AudioInput(
                format=PCMAudioFormat(rate=24000),
                transcription=InputTranscription(model=stt_model),
                turn_detection=None,
            ),
            output=AudioOutput(
                format=PCMAudioFormat(rate=24000),
                model=tts_model,
                voice=voice,
            ),
        ),
    )

    llm = InworldRealtimeLLMService(
        api_key=os.environ["INWORLD_API_KEY"],
        settings=InworldRealtimeLLMService.Settings(
            system_instruction="""You are a helpful and friendly AI assistant powered by Inworld.

Your voice and personality should be warm and engaging. Keep your responses
concise and conversational since this is a voice interaction.

Always be helpful and proactive in offering assistance.""",
            session_properties=session_properties,
        ),
    )

    # Note: function calling requires a paid Inworld account and a
    # function-calling-capable model
    llm.register_function("get_current_weather", fetch_weather_from_api)

    context = LLMContext(
        [{"role": "developer", "content": "Say hello and introduce yourself!"}],
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
        logger.info("Client connected")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
        logger.info(f"Transcript: {timestamp}user: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}assistant: {message.content}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
