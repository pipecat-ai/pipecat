#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesLLMService,
    OpenAIResponsesReasoningConfig,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


@tool_options(cancel_on_interruption=False, timeout_secs=30)
async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    # Simulate a long-running API call, so we can test async function calls.
    await asyncio.sleep(15)
    logger.debug("Returning get_current_weather result.")
    await params.result_callback({"conditions": "nice", "temperature": "75"})


# A lookup that hangs: it sleeps far past the deadline it was registered with,
# so the call is always cancelled before it can report anything. The result it
# would eventually have returned is distinctive on purpose — if the bot ever
# quotes a share price, a cancelled handler's result reached the conversation.
@tool_options(cancel_on_interruption=False, timeout_secs=5)
async def get_stock_price(params: FunctionCallParams, symbol: str):
    """Get the current share price for a stock.

    Args:
        symbol: The ticker symbol, e.g. "NVDA".
    """
    await asyncio.sleep(20)
    logger.debug("Returning get_stock_price result.")
    await params.result_callback({"price": "184.20", "currency": "USD"})


@tool_options(cancel_on_interruption=False, cancellable_by_llm=True, timeout_secs=120)
async def write_report(params: FunctionCallParams, topic: str):
    """Write a long research report on a topic.

    Args:
        topic: What the report should cover.
    """
    # Long enough to still be running when the LLM asks to stop it: listing what
    # is running and then calling the cancel tool, with a spoken reply often in
    # between, outlasts shorter work.
    await asyncio.sleep(25)
    logger.debug("Returning write_report result.")
    await params.result_callback({"report": f"A 5000-word report on {topic}."})


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"name": "The Golden Dragon"})


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
            model="gpt-5.4",
            reasoning=OpenAIResponsesReasoningConfig(effort="low"),
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

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

    @llm.event_handler("on_function_calls_cancelled")
    async def on_function_calls_cancelled(service, function_calls):
        for item in function_calls:
            logger.info(f"Function call cancelled: {item.function_name} [{item.tool_call_id}]")

    # cancel_on_interruption=False (set via @tool_options) makes this an async
    # function call.
    context = LLMContext(
        tools=[
            get_current_weather,
            write_report,
            get_stock_price,
            get_restaurant_recommendation,
        ]
    )
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation.
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
