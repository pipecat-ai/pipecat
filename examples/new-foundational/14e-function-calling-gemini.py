#
# Copyright (c) 2024–2025, Daily
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
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


# Global variable to store the peer connection ID
webrtc_peer_id = None


async def get_weather(function_name, tool_call_id, arguments, llm, context, result_callback):
    await llm.push_frame(TTSSpeakFrame("Let me check on that."))
    location = arguments["location"]
    await result_callback(f"The weather in {location} is currently 72 degrees and sunny.")


async def get_image(function_name, tool_call_id, arguments, llm, context, result_callback):
    question = arguments["question"]
    logger.debug(f"Requesting image with user_id={webrtc_peer_id}, question={question}")

    # Request the image frame
    await llm.request_image_frame(
        user_id=webrtc_peer_id,
        function_name=function_name,
        tool_call_id=tool_call_id,
        text_content=question,
    )

    # Wait a short time for the frame to be processed
    await asyncio.sleep(0.5)

    # Return a result to complete the function call
    await result_callback(
        f"I've captured an image from your camera and I'm analyzing what you asked about: {question}"
    )


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    global webrtc_peer_id
    webrtc_peer_id = webrtc_connection.pc_id

    logger.info(f"Starting bot with peer_id: {webrtc_peer_id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_in_enabled=True,  # Make sure camera input is enabled
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-001")
    llm.register_function("get_weather", get_weather)
    llm.register_function("get_image", get_image)

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
    get_image_function = FunctionSchema(
        name="get_image",
        description="Get an image from the video stream.",
        properties={
            "question": {
                "type": "string",
                "description": "The question that the user is asking about the image.",
            }
        },
        required=["question"],
    )
    tools = ToolsSchema(standard_tools=[weather_function, get_image_function])

    system_prompt = """\
You are a helpful assistant who converses with a user and answers questions. Respond concisely to general questions.

Your response will be turned into speech so use only simple words and punctuation.

You have access to two tools: get_weather and get_image.

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
        {"role": "user", "content": "Say hello."},
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")

    @transport.event_handler("on_client_closed")
    async def on_client_closed(transport, client):
        logger.info(f"Client closed connection")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
