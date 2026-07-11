#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Manual demonstration of the missing-handler (developer-error) recovery path.

When a tool is advertised to the LLM via ``tools``/``LLMContext`` but has no
handler — neither set on its ``FunctionSchema`` nor registered via
``llm.register_function(...)`` — the LLM happily emits a tool call and then...
nothing happens on the Pipecat side, leaving the conversation stuck.

Pipecat's recovery path (``LLMService._missing_function_call_handler``)
catches this case:

- Logs a ``logger.error`` distinguishing **developer error** (tool advertised
  but no handler wired up) from a hallucination (tool not advertised),
  pointing at the missing handler.
- Returns a neutral terminal tool result
  (``LLMService.MISSING_FUNCTION_CALL_MESSAGE_TEMPLATE``: "The function
  `X` is not currently available.") so the call still terminates with a
  normal tool result instead of leaving the conversation stuck.

This example is **deliberately broken**: the weather schema is in ``tools``
but its handler is wired up neither on the ``FunctionSchema`` nor via
``register_function``. Ask the bot about the weather and observe:

1. The LLM emits a tool call for ``get_current_weather``.
2. ``logger.error`` fires with "advertised to the LLM but has no handler —
   set FunctionSchema.handler (recommended) or call register_function()".
3. The terminal tool result is fed back to the LLM.
4. The LLM responds in voice based on that result (typically something
   like "the weather function isn't available right now").

Uses the OpenAI LLM service with defaults. Swap to another provider to
validate this behavior elsewhere.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


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
weather_tools = ToolsSchema(standard_tools=[weather_function])


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
    logger.info("Starting missing-handler demo bot (no handler is registered on purpose)")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses "
                "will be spoken aloud, so avoid emojis, bullet points, or other "
                "formatting that can't be spoken. Respond briefly and naturally. "
                "Always use the get_current_weather function to answer questions "
                "about the current weather."
            ),
        ),
    )

    # *** DELIBERATELY OMITTED ***
    # The whole point of this example is to demonstrate the missing-handler
    # recovery path. To wire the tool up correctly, either pass the handler to
    # the schema above (recommended) —
    #
    #     weather_function = FunctionSchema(..., handler=fetch_weather_from_api)
    #
    # — or register it here:
    #
    # llm.register_function("get_current_weather", fetch_weather_from_api)

    context = LLMContext(tools=weather_tools)
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
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        logger.info(
            "=== Ask for the weather. Watch for a logger.error about the missing "
            "handler, and listen for the LLM's response based on the recovery "
            "message. ==="
        )
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
