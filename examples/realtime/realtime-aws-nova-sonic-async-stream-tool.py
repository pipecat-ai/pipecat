#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: streaming async function call with the AWS Nova Sonic LLM service.

The ``track_current_location`` tool simulates a GPS tracker reporting the
device's position during a road trip from San Francisco to San Diego. It
sends two intermediate updates (via ``params.result_callback`` with
``is_final=False``) as the vehicle passes through cities along the way, then
delivers the final destination.

The placeholder is sent as a formal Nova Sonic ``toolResult``; each
intermediate result is forwarded as a cross-modal user-role text input event
so the model can fold each update into its next turn.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import FunctionCallResultProperties, LLMRunFrame
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
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


async def track_current_location(params: FunctionCallParams):
    """Simulate a GPS tracker reporting position during a road trip."""
    gps = {"lat": 37.7310, "lng": -122.4527}
    await params.result_callback(
        {"gps": gps, "city": "San Francisco"},
        properties=FunctionCallResultProperties(is_final=False),
    )

    await asyncio.sleep(10)
    gps = {"lat": 33.96003, "lng": -118.40639}
    await params.result_callback(
        {"gps": gps, "city": "Los Angeles"},
        properties=FunctionCallResultProperties(is_final=False),
    )

    await asyncio.sleep(10)
    gps = {"lat": 32.743569, "lng": -117.20466}
    await params.result_callback({"gps": gps, "city": "San Diego"})


location_function = FunctionSchema(
    name="track_current_location",
    description=(
        "Start tracking the user's current GPS location, reporting position "
        "updates until the user reaches their destination. "
        "Once this tracker is started, it doesn't need to be started again for subsequent updates; "
        "just call this function once to kick it off and the updates will come in automatically."
    ),
    properties={},
    required=[],
)

tools = ToolsSchema(standard_tools=[location_function])


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

    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken "
        "dialog exchanging the transcripts of a natural real-time conversation. "
        "Keep your responses short, generally two or three sentences for chatty "
        "scenarios. You have access to a function that starts tracking the user's "
        "location and provides regular updates on it. Narrate each position "
        "update to the user as it arrives (city only, no coordinates). "
        "When you receive the final location, tell the user the destination has "
        "been reached."
    )

    llm = AWSNovaSonicLLMService(
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        region=os.environ["AWS_REGION"],
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        settings=AWSNovaSonicLLMService.Settings(
            voice="tiffany",
            system_instruction=system_instruction,
        ),
    )

    llm.register_function(
        "track_current_location",
        track_current_location,
        cancel_on_interruption=False,
    )

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
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
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
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


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
