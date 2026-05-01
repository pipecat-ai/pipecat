#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example demonstrating ``PipelineTask(app_resources=...)``.

``app_resources`` is an application-defined bag of anything your
application code may want to share across a session: database handles,
HTTP clients, feature flags, per-user state, observability clients,
in-memory caches — whatever fits your app. Pipecat passes it through
untouched and exposes it as ``task.app_resources``, so any code with a
handle on the task can read or mutate it.

Two of the convenience aliases exercised below:

- Tool handlers read it from ``FunctionCallParams.app_resources``.
- Custom ``FrameProcessor`` subclasses read it from
  ``self.pipeline_task.app_resources``.

This example uses two small loggers as stand-ins for that "shared thing":
``ToolCallLogger`` (written from tool handlers) and
``TranscriptionLogger`` (written from a custom ``FrameProcessor`` that
sits in the pipeline). A real app might just as easily pass a Postgres
pool, a Redis client, a Stripe SDK instance, or any combination thereof.
The mechanics shown here — construct once, hand to the task, read it
from each site, inspect it after the session — are the same regardless
of what you put in.

We bundle resources in a typed ``AppResources`` dataclass and cast back
to it at each read site. Pipecat doesn't care what type you pass (a
plain dict works too), but a typed container gives you autocomplete and
refactor safety instead of dict-by-string-key lookups.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMRunFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


class ToolCallLogger:
    """Stand-in shared resource — swap for whatever your app actually needs."""

    def __init__(self):
        """Initialize the logger with an empty list of recorded calls."""
        self._calls: list[dict[str, Any]] = []

    def log_tool_call(self, function_name: str, arguments: Mapping[str, Any]) -> None:
        """Record a tool call invocation.

        Args:
            function_name: The name of the tool being invoked.
            arguments: The arguments passed to the tool.
        """
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "function_name": function_name,
            "arguments": dict(arguments),
        }
        self._calls.append(entry)
        logger.info(f"[ToolCallLogger] {function_name} called with {dict(arguments)}")

    def dump(self) -> str:
        """Return all recorded tool calls as a JSON string."""
        return json.dumps(self._calls, indent=2)


class TranscriptionLogger:
    """Records final user transcriptions — written from a custom FrameProcessor."""

    def __init__(self):
        """Initialize the logger with an empty list of recorded transcriptions."""
        self._entries: list[dict[str, Any]] = []

    def log_transcription(self, text: str) -> None:
        """Record a transcription.

        Args:
            text: The transcribed user utterance.
        """
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "text": text,
        }
        self._entries.append(entry)
        logger.info(f"[TranscriptionLogger] {text!r}")

    def dump(self) -> str:
        """Return all recorded transcriptions as a JSON string."""
        return json.dumps(self._entries, indent=2)


@dataclass
class AppResources:
    """Typed container for everything the app shares across this session.

    Add fields here as the app grows (e.g. ``db: AsyncConnection``,
    ``http: httpx.AsyncClient``). Read sites ``cast()`` to this type to
    get autocomplete and refactor safety:

    - In tools: ``cast(AppResources, params.app_resources)``.
    - In custom processors: ``cast(AppResources, self.pipeline_task.app_resources)``.
    """

    tool_call_logger: ToolCallLogger
    transcription_logger: TranscriptionLogger


async def fetch_weather_from_api(params: FunctionCallParams):
    resources = cast(AppResources, params.app_resources)
    resources.tool_call_logger.log_tool_call(params.function_name, params.arguments)
    await params.result_callback({"conditions": "nice", "temperature": "75"})


async def fetch_restaurant_recommendation(params: FunctionCallParams):
    resources = cast(AppResources, params.app_resources)
    resources.tool_call_logger.log_tool_call(params.function_name, params.arguments)
    await params.result_callback({"name": "The Golden Dragon"})


class TranscriptionLoggingProcessor(FrameProcessor):
    """Logs each final user transcription into the shared app resources.

    Demonstrates the second read site for ``app_resources``: any custom
    ``FrameProcessor`` can reach the same bag every tool handler sees by
    going through ``self.pipeline_task.app_resources``. ``pipeline_task``
    is ``None`` until the task sets the processor up, so we guard against
    that case.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Forward all frames; log final user transcriptions on the way through."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and self.pipeline_task is not None:
            resources = cast(AppResources, self.pipeline_task.app_resources)
            resources.transcription_logger.log_transcription(frame.text)

        await self.push_frame(frame, direction)


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

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    # You can also register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("get_restaurant_recommendation", fetch_restaurant_recommendation)

    @llm.event_handler("on_connection_error")
    async def on_connection_error(service, error):
        logger.error(f"LLM connection error: {error}")

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        # Avoid appending this filler message to the LLM context — it would
        # alter the conversation history and prevent
        # OpenAIResponsesLLMService's previous_response_id optimization from
        # matching, forcing a full context resend.
        await tts.queue_frame(TTSSpeakFrame("Let me check on that.", append_to_context=False))

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
                "description": "The temperature unit to use. Infer this from the user's location.",
            },
        },
        required=["location", "format"],
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

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptionLoggingProcessor(),
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # Keep local handles so we can read collected state after the session
    # ends; Pipecat never copies or clears the object.
    tool_call_logger = ToolCallLogger()
    transcription_logger = TranscriptionLogger()
    resources = AppResources(
        tool_call_logger=tool_call_logger,
        transcription_logger=transcription_logger,
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        app_resources=resources,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)

    # The session has ended; read whatever state the handlers built up.
    logger.info(f"Tool calls logged during session:\n{tool_call_logger.dump()}")
    logger.info(f"Transcriptions logged during session:\n{transcription_logger.dump()}")


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
