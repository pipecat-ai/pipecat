#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Manual validation harness for the ``add_tool_change_messages`` feature.

When tools change mid-conversation, LLMs can produce a few different
flavors of tool-call-related hallucination:

- **Forward hallucination** — calling a tool that has been removed.
- **Negative hallucination** — refusing to call a tool that has been
  re-added (because recent context is full of "I can't" responses).
- **Hallucinated output when tools are unavailable** — making up an
  answer rather than declining gracefully, or producing JSON that
  *looks* like a tool call but is actually just an assistant text
  response.

The ``add_tool_change_messages`` feature mitigates these by appending a
developer-role message to the conversation whenever ``LLMSetToolsFrame``
changes the set of advertised tools, so the LLM stays in sync with what's
actually available.

This harness exercises all of those flavors by flipping the advertised
tool set on a turn counter:

    Phase 0 (turns 1–4):   weather tool ACTIVE — confirm baseline.
    Phase 1 (turns 5–8):   tool REMOVED — keep asking for weather.
    Phase 2 (turn 9+):     tool RE-ADDED — does the LLM call it again?

Set ``ADD_TOOL_CHANGE_MESSAGES=0`` to disable the mitigation and see the
unmitigated behavior. The default is ON so a fresh run shows the feature
working.

Defaults to Llama 3.1 8B Instruct via a locally-running Ollama —
anecdotally one of the more hallucination-prone of the easily accessible
models. Pull the model once with ``ollama pull llama3.1:8b`` and make
sure ``ollama serve`` is running. Swap the LLM service to validate other
providers.

Run with::

    uv run examples/features/features-add-tool-change-messages.py
    ADD_TOOL_CHANGE_MESSAGES=0 uv run examples/features/features-add-tool-change-messages.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, LLMSetToolsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# Default ON so a fresh run shows the feature working. Set to "0" to A/B
# against the unmitigated behavior.
ADD_TOOL_CHANGE_MESSAGES = os.environ.get("ADD_TOOL_CHANGE_MESSAGES", "1") == "1"


async def fetch_weather_from_api(params: FunctionCallParams):
    await params.result_callback({"conditions": "nice", "temperature": "75"})


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
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(
        f"Starting add_tool_change_messages demo bot "
        f"(ADD_TOOL_CHANGE_MESSAGES={ADD_TOOL_CHANGE_MESSAGES})"
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            # Llama 3.1 8B Instruct is anecdotally one of the more
            # hallucination-prone of the easily accessible models — exactly
            # what we want for this validation harness. Pull it with
            # ``ollama pull llama3.1:8b`` and make sure ``ollama serve``
            # is running.
            model="llama3.1:8b",
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses "
                "will be spoken aloud, so avoid emojis, bullet points, or other "
                "formatting that can't be spoken. Respond briefly and naturally. "
                "If the user asks for the current weather, use the `get_current_weather` "
                "function if it's available. IMPORTANT: if you do not have access to the function, "
                "say something along the lines of 'Sorry, I can't check the weather right now.'."
            ),
        ),
    )
    llm.register_function("get_current_weather", fetch_weather_from_api)

    context = LLMContext(tools=weather_tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        add_tool_change_messages=ADD_TOOL_CHANGE_MESSAGES,
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Phase controller: roughly 4 turns per phase.
    user_turn_count = 0
    REMOVE_AT_TURN = 5  # tool gone for turn N onward
    READD_AT_TURN = 9  # tool back for turn N onward

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        nonlocal user_turn_count
        user_turn_count += 1
        logger.info(f"=== User turn {user_turn_count} complete ===")

        if user_turn_count == REMOVE_AT_TURN - 1:
            logger.info(
                "=== Phase 1: weather tool REMOVED. Keep asking about the weather "
                "to exercise hallucination scenarios. ==="
            )
            await task.queue_frame(LLMSetToolsFrame(tools=NOT_GIVEN))
        elif user_turn_count == READD_AT_TURN - 1:
            logger.info(
                "=== Phase 2: weather tool RE-ADDED. Ask for the weather again — "
                "does the LLM call it, or keep refusing? (THIS IS THE TEST.) ==="
            )
            await task.queue_frame(LLMSetToolsFrame(tools=weather_tools))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        logger.info(
            "=== Phase 0: weather tool ACTIVE. Ask for the weather a few times "
            "to confirm it's working. ==="
        )
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Please introduce yourself briefly to the user, then invite them "
                    "to ask about the weather."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
