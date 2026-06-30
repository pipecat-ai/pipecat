#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: async function call with the Grok Realtime LLM service.

The ``get_current_weather`` tool is registered with
``cancel_on_interruption=False`` and simulates a slow API call (10s sleep).
While the call is in flight the conversation continues; the result arrives
later via the async-tool mechanism and is forwarded to Grok Realtime as a
``function_call_output`` so the model can integrate it naturally into its
next turn.
"""

import asyncio
import os
import random
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.xai.realtime.events import SessionProperties
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


@tool_options(cancel_on_interruption=False)
async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    # Simulate a long-running API call so we can demonstrate that the
    # conversation continues while the tool is in flight.
    await asyncio.sleep(10)
    temperature = random.randint(60, 85) if format == "fahrenheit" else random.randint(15, 30)
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "location": location,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


system_instruction = (
    "You are a friendly assistant. The user and you will engage in a spoken "
    "dialog exchanging the transcripts of a natural real-time conversation. "
    "Keep your responses short, generally two or three sentences for chatty "
    "scenarios. When the user asks for the weather, call get_current_weather. "
    "While you wait for the result, keep chatting with the user. When the "
    "result arrives, share it with the user naturally."
)


# Note: Grok has built-in server-side VAD, so we don't need local VAD.
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
    logger.info(f"Starting bot")

    llm = GrokRealtimeLLMService(
        api_key=os.environ["XAI_API_KEY"],
        settings=GrokRealtimeLLMService.Settings(
            system_instruction=system_instruction,
            session_properties=SessionProperties(
                voice="Ara",
            ),
        ),
    )

    context = LLMContext(tools=[get_current_weather])
    # It appears that Grok Realtime can sometimes be slow to detect the start
    # of a user's turn; uncomment the below imports and user_params to
    # enable "supplemental" interruptions.
    # from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.turns.user_turn_strategies import UserTurnStrategies
    # from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        realtime_service_mode=True,
        # user_params=LLMUserAggregatorParams(
        #     vad_analyzer=SileroVADAnalyzer(),
        #     user_turn_strategies=UserTurnStrategies(start=[VADUserTurnStartStrategy(
        #         enable_interruptions=True,
        #         enable_user_speaking_frames=False,  # Grok already emits turn frames
        #     )], stop=[]) # Grok already emits turn frames
        # ),
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
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
