#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMRunFrame,
    TTSSpeakFrame,
    UserImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAIAssistantContextAggregator, OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


async def get_current_time(params: FunctionCallParams):
    """Returns the current date and time."""
    print("Getting current time")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await params.result_callback({"current_time": current_time})


async def calculate(params: FunctionCallParams):
    """Performs a simple calculation."""
    expression = params.arguments.get("expression", "")
    try:
        print(f"Calculating {expression}")
        result = eval(expression, {"__builtins__": {}}, {})
        await params.result_callback({"result": str(result), "expression": expression})
    except Exception as e:
        await params.result_callback({"error": f"Could not calculate: {str(e)}"})


async def get_weather(params: FunctionCallParams):
    """Mock function to get weather information."""
    print("Getting weather - starting background task")
    location = params.arguments.get("location", "unknown location")

    async def delayed_response():
        """Background task that simulates a slow API call."""
        print(f"⏳ Weather API call started for {location}...")
        await asyncio.sleep(10)
        print(f"✅ Weather API call completed for {location}")
        await params.result_callback(
            {
                "location": location,
                "temperature": "72°F",
                "conditions": "Partly cloudy",
                "humidity": "65%",
            }
        )

    asyncio.create_task(delayed_response())
    print("Getting weather - background task scheduled, returning immediately")


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-2-andromeda-en")

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),  # HuggingFace may not need this
        base_url="https://v1vyrdc4v2gs2f3m.us-east-1.aws.endpoints.huggingface.cloud/v1",
        model="qforge/Qwen3-14B-AT",
        params=OpenAILLMService.InputParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=8192,
            presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
        ),
    )

    # Register function handlers
    llm.register_function("get_current_time", get_current_time)
    llm.register_function("calculate", calculate)
    llm.register_function("get_weather", get_weather)

    # Optional: Add a handler for when function calls start
    # @llm.event_handler("on_function_calls_started")
    # async def on_function_calls_started(service, function_calls):
    #     await tts.queue_frame(TTSSpeakFrame("Let me check on that for you."))

    # Define function schemas
    time_function = FunctionSchema(
        name="get_current_time",
        description="Get the current date and time",
        properties={},
        required=[],
    )

    calculate_function = FunctionSchema(
        name="calculate",
        description="Perform a mathematical calculation",
        properties={
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
            },
        },
        required=["expression"],
    )

    weather_function = FunctionSchema(
        name="get_weather",
        description="Get the current weather for a location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        required=["location"],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[time_function, calculate_function, weather_function])

    messages = [
        {
            "role": "system",
            "content": """You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way. You have access to tools to get the current time, perform calculations, and get weather information.

After receiving the tool ack be sure to give the user information about what you are doing. Do not only think about it. Say it when you have new information.

You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function call in tools call sections.

## Async Tool Call Protocol:
- Tools may execute asynchronously. When you call a tool, you may receive an ACK (acknowledgment) before the actual result.
- ACK format: A tool message with content '<tool_ack id="tN"/>' (where tN is the tool call id).
- RESPONSE format: A tool message with JSON content '{"id":"tN","ok":true|false,"data":...}' or '{"id":"tN","ok":false,"error":...}'.
- After making tool calls, wait for each tool's ACK or RESPONSE before proceeding with new user-facing messages.
- While waiting for results, provide result-independent content (e.g., 'I'm fetching that information now').
- Only present result-dependent information after receiving the actual RESPONSE, not just the ACK.
""",
        },
    ]

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
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
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
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
