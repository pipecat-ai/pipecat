#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Example: async function call with intermediate updates.

The ``track_current_location`` tool simulates a GPS tracker reporting the
device's position during a road trip from San Francisco to San Diego.  It
sends two intermediate updates (via ``params.result_callback`` with
``is_final=False``) as the vehicle passes through cities along the way, then
delivers the final destination (via ``params.result_callback``).  Each update
returns the same structure with a different city:

  Update 1 – {gps, city: "San Francisco"}   ← trip start
  Update 2 – {gps, city: "Los Angeles"}     ← passing through
  Final     – {gps, city: "San Diego"}      ← destination reached

Because the function is registered with ``cancel_on_interruption=False``, the
LLM can keep talking while the trip is in progress; each position update
arrives as a developer message so the LLM can narrate the journey to the user.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    FunctionCallResultProperties,
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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


async def track_current_location(params: FunctionCallParams):
    """Simulate a GPS tracker reporting position during a road trip.

    Step 1 – San Francisco (trip start)     (update)
    Step 2 – Los Angeles   (passing through) (update)
    Step 3 – San Diego     (destination)     (final result)
    """

    # First update: initial city estimate.
    gps = {"lat": 37.7310, "lng": -122.4527}
    await params.result_callback(
        {"gps": gps, "city": "San Francisco"},
        properties=FunctionCallResultProperties(is_final=False),
    )

    # Second update: revised city estimate.
    await asyncio.sleep(10)
    gps = {"lat": 33.96003, "lng": -118.40639}
    await params.result_callback(
        {"gps": gps, "city": "Los Angeles"},
        properties=FunctionCallResultProperties(is_final=False),
    )

    # Final result: confirmed city.
    await asyncio.sleep(10)
    gps = {"lat": 32.743569, "lng": -117.20466}
    await params.result_callback({"gps": gps, "city": "San Diego"})


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

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        enable_async_tool_cancellation=True,
        settings=GoogleLLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Your responses will be spoken aloud, so avoid emojis, bullet points, or other "
                "formatting that can't be spoken. "
                "You have access to a function that starts tracking the user's location and "
                "provides regular updates on it. When you receive the final location, tell the user "
                "the destination has been reached."
            ),
        ),
    )

    # cancel_on_interruption=False makes this an async function call: the LLM
    # continues the conversation immediately and receives updates/result later.
    llm.register_function(
        "track_current_location",
        track_current_location,
        cancel_on_interruption=False,
        timeout_secs=30,
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Sure, tracking your location now."))

    @llm.event_handler("on_function_calls_cancelled")
    async def on_function_calls_cancelled(service, cancelled):
        for item in cancelled:
            logger.info(f"Function call cancelled: {item.function_name} [{item.tool_call_id}]")

    location_function = FunctionSchema(
        name="track_current_location",
        description="Start tracking the user's current GPS location, reporting position updates until the user reaches their destination.",
        properties={},
        required=[],
    )
    tools = ToolsSchema(standard_tools=[location_function])

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
