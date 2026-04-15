#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame, UserImageRequestFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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

load_dotenv(override=True)


async def get_weather(params: FunctionCallParams):
    location = params.arguments["location"]
    await params.result_callback(f"The weather in {location} is currently 72 degrees and sunny.")


async def fetch_restaurant_recommendation(params: FunctionCallParams):
    await params.result_callback({"name": "The Golden Dragon"})


async def get_image(params: FunctionCallParams):
    """Fetch the user image and push it to the LLM.

    When called, this function pushes a UserImageRequestFrame upstream to the
    transport. As a result, the transport will request the user image and push a
    UserImageRawFrame downstream which will be added to the context by the LLM
    assistant aggregator. The result_callback will be invoked once the image is
    retrieved and processed.
    """
    user_id = params.arguments["user_id"]
    question = params.arguments["question"]
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
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
    llm.register_function("get_weather", get_weather)
    llm.register_function("get_image", get_image)
    llm.register_function("get_restaurant_recommendation", fetch_restaurant_recommendation)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    weather_function = FunctionSchema(
        name="get_weather",
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
    get_image_function = FunctionSchema(
        name="get_image",
        description="Called when the user requests a description of their camera feed",
        properties={
            "user_id": {
                "type": "string",
                "description": "The ID of the user to grab the image from",
            },
            "question": {
                "type": "string",
                "description": "The question that the user is asking about the image",
            },
        },
        required=["user_id", "question"],
    )
    tools = ToolsSchema(standard_tools=[weather_function, get_image_function, restaurant_function])

    system_prompt = """\
You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.

Your response will be turned into speech so use only simple words and punctuation.

You have access to three tools: get_weather, get_restaurant_recommendation, and get_image.

You can respond to questions about the weather using the get_weather tool.

You can answer questions about the user's video stream using the get_image tool. Some examples of phrases that \
indicate you should use the get_image tool are:
- What do you see?
- What's in the video?
- Can you describe the video?
- Tell me about what you see.
- Tell me something interesting about what you see.
- What's happening in the video?
"""
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    context = LLMContext(messages, tools)
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
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")

        await maybe_capture_participant_camera(transport, client)

        client_id = get_transport_client_id(transport, client)

        # Kick off the conversation.
        messages.append(
            {
                "role": "system",
                "content": f"Please introduce yourself to the user. Use '{client_id}' as the user ID during function calls.",
            }
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
