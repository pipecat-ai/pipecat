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
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame, UserImageRequestFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import (
    create_transport,
    get_transport_client_id,
    maybe_capture_participant_camera,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


@tool_options(cancel_on_interruption=False, timeout_secs=30)
async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    # Simulate a long-running API call, so we can test async function calls (cancel_on_interruption=False).
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


async def get_image(params: FunctionCallParams, user_id: str, question: str):
    """Fetch the user image and push it to the LLM.

    When called, this function pushes a UserImageRequestFrame upstream to the
    transport. As a result, the transport will request the user image and push a
    UserImageRawFrame downstream which will be added to the context by the LLM
    assistant aggregator. The result_callback will be invoked once the image is
    retrieved and processed.

    Args:
        user_id: The ID of the user to grab the image from.
        question: The question that the user is asking about the image.
    """
    logger.debug(f"Requesting image with user_id={user_id}, question={question}")

    # Request a user image frame and indicate that it should be added to the
    # context. Also associate it to the function call. Pass the result_callback
    # so it can be invoked when the image is actually retrieved.
    await params.llm.push_frame(
        UserImageRequestFrame(
            user_id=user_id,
            text=question,
            append_to_context=True,
            function_name=params.function_name,
            tool_call_id=params.tool_call_id,
            result_callback=params.result_callback,
        ),
        FrameDirection.UPSTREAM,
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
        video_in_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
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

    system_prompt = """\
You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.

Your response will be turned into speech so use only simple words and punctuation.

You have access to three tools: get_current_weather, get_restaurant_recommendation, and get_image.

You can respond to questions about the weather using the get_current_weather tool.

You can answer questions about the user's video stream using the get_image tool. Some examples of phrases that \
indicate you should use the get_image tool are:
- What do you see?
- What's in the video?
- Can you describe the video?
- Tell me about what you see.
- Tell me something interesting about what you see.
- What's happening in the video?
"""

    llm = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            system_instruction=system_prompt,
        ),
    )

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

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
            get_image,
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
        logger.info(f"Client connected: {client}")

        await maybe_capture_participant_camera(transport, client)

        client_id = get_transport_client_id(transport, client)

        # Kick off the conversation.
        context.add_message(
            {
                "role": "developer",
                "content": f"Please introduce yourself to the user. Use '{client_id}' as the user ID during function calls.",
            }
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
