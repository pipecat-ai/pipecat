#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    LLMRunFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


async def fetch_weather_from_api(params: FunctionCallParams):
    # Simulate a long-running API call, so we can test async function calls.
    await asyncio.sleep(20)
    await params.result_callback({"conditions": "nice", "temperature": "75"})


async def fetch_restaurant_recommendation(params: FunctionCallParams):
    await params.result_callback({"name": "The Golden Dragon"})


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

system_prompt = """
You are a helpful assistant in a voice conversation. 
Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.

You can call tools asynchronously. Tool calls follow a lifecycle and may complete later, out of order, or after the user has changed their request.

You must follow these rules strictly:

1. Tool Call Lifecycle
- When you call a tool, a message with role "tool" and content {"type": "async_tool", "status": "started"} will appear.
- The tool result will arrive later as a message with role "developer" and content:
  {
    "type": "async_tool",
    "tool_call_id": "...",
    "status": "finished",
    "result": "..."
  }
- Do NOT assume the result until you see "status": "finished".

2. Handling Multiple Tool Calls
- You may issue multiple tool calls in parallel.
- Results may arrive in any order.
- You must track tool_call_id to match results correctly.

3. Incremental Updates
- If results are still pending, you may inform the user that you're waiting.
- Do NOT hallucinate missing tool results.
- Do NOT invoke asking for the same information that has already been requested and you are waiting for the result.

4. Changing User Intent
- The user may change their request while tools are still running.
- If a request is no longer relevant, ignore its tool results when they arrive.
- Do NOT present outdated or cancelled information.

5. Responding to Results
- When tool results arrive:
  - Extract and summarize the relevant information.
  - Combine results if multiple tool calls are related.
  - Provide a clear and helpful answer.

6. Tool Usage Guidelines
- Always call tools when required instead of guessing.
- Ensure arguments are correct and reflect the latest user request.

7. Example Behavior
- If the user asks for multiple locations, call tools for each.
- If the user changes from "Florida" to "California", ignore Florida's result if it arrives later.

Your goal is to behave like a reliable orchestrator of asynchronous tool calls while providing clear, accurate, and up-to-date responses to the user.
"""


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAISTTService.Settings(
            model="gpt-4o-transcribe",
            prompt="Expect words related weather, such as temperature and conditions. And restaurant names.",
        ),
    )

    tts = OpenAITTSService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAITTSService.Settings(
            voice="ballad",
        ),
        instructions="Please speak clearly and at a moderate pace.",
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(
            system_instruction=system_prompt,
        ),
    )

    # You can also register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function(
        "get_current_weather",
        fetch_weather_from_api,
        cancel_on_interruption=False,
        is_async=True,
        timeout_secs=30,
    )
    llm.register_function("get_restaurant_recommendation", fetch_restaurant_recommendation)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

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
            user_aggregator,
            llm,
            tts,
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
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
