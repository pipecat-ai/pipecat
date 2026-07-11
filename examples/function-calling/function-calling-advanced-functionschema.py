#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Advanced: defining a tool with an explicit ``FunctionSchema``.

Direct functions (see ``function-calling-direct.py``) are the preferred way to
define a tool: one async function is both the handler and the schema, which
Pipecat derives from its signature and docstring. Reach for an explicit
``FunctionSchema`` only when you need control the direct-function generator
can't give you — for example a strict ``enum`` constraint (used below), some
other JSON-schema detail it doesn't emit, or a handler that isn't shaped like a
direct function.

A ``FunctionSchema`` spells out the tool's name, description, and parameters by
hand. Bundle the ``handler`` that runs when the LLM calls it on the schema, then
list the schema in ``LLMContext(tools=[...])`` exactly as you would a direct
function: Pipecat auto-registers the handler wherever the schema is advertised
(an ``LLMContext`` or an ``LLMSetToolsFrame``).

Uses the OpenAI LLM service with defaults. Swap to another provider to validate
this behavior elsewhere.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


# A handler bundled on a FunctionSchema is a classic function-call handler: it
# takes the FunctionCallParams and reads the LLM-supplied arguments from
# params.arguments (the schema, not the signature, defines the tool's shape).
# Decorate it with @tool_options to override the default call options
# (cancel_on_interruption, timeout_secs), just as you would a direct function.
async def fetch_current_weather(params: FunctionCallParams):
    location = params.arguments["location"]
    logger.info(f"Fetching weather for {location}")
    await params.result_callback({"conditions": "nice", "temperature": "75"})


# Writing the schema out explicitly gives precise control over the parameters
# (here, a `format` enum) that a direct function would have to infer.
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
    handler=fetch_current_weather,
)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
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
    logger.info("Starting bot")

    stt = OpenAIRealtimeSTTService(api_key=os.environ["OPENAI_API_KEY"])

    tts = OpenAITTSService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAITTSService.Settings(
            instructions="Please speak clearly and at a moderate pace.",
            voice="ballad",
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses "
                "will be spoken aloud, so avoid emojis, bullet points, or other "
                "formatting that can't be spoken. Always use the get_current_weather "
                "function to answer questions about the current weather."
            ),
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    context = LLMContext(tools=[weather_function])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Please introduce yourself briefly to the user, then invite "
                    "them to ask about the weather."
                ),
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

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
