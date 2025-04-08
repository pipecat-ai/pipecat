#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Bot Implementation.

This module implements a chatbot using OpenAI's GPT-4 model for natural language
processing. It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Text-to-speech using ElevenLabs
- Support for both English and Spanish

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow.
"""

import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pipecatcloud.agent import DailySessionArguments, SessionArguments, WebSocketSessionArguments

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.gladia.stt import GladiaSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

# Check if we're in local development mode
LOCAL_RUN = os.getenv("LOCAL_RUN")
if LOCAL_RUN:
    import asyncio
    import webbrowser

    try:
        from local_runner import configure
    except ImportError:
        logger.error("Could not import local_runner module. Local development mode may not work.")

# Logger for local dev
# logger.add(sys.stderr, level="DEBUG")


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    """Fetch weather data dummy function.

    This function simulates fetching weather data from an external API.
    It demonstrates how to call an external service from the language model.
    """
    await llm.push_frame(TTSSpeakFrame("Let me check on that."))
    await result_callback({"conditions": "nice", "temperature": "75"})


async def main(transport: BaseTransport):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Animation processing
    - RTVI event handling

    Uses the transport defined by the calling function.
    See below for various ways to start the bot with different transports.
    """
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="c45bc5ec-dc68-4feb-8829-6e6b2748095d",  # Movieman
    )

    stt = GladiaSTTService(api_key=os.getenv("GLADIA_API_KEY"))

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # Register your function call providing the function name and callback
    llm.register_function("get_current_weather", fetch_weather_from_api)

    # Define your function call using the FunctionSchema
    # Learn more about function calling in Pipecat:
    # https://docs.pipecat.ai/guides/features/function-calling
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

    # Set up the tools schema with your weather function call
    tools = ToolsSchema(standard_tools=[weather_function])

    # Set up initial messages for the bot
    messages = [
        {
            "role": "system",
            "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
        },
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    # Pass your initial messages and tools to the context to initialize the context
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Add your processors to the pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Create a PipelineTask to manage the pipeline
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # Notify the client that the bot is ready
        await rtvi.set_bot_ready()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation by pushing a context frame to the pipeline
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        # Cancel the PipelineTask to stop processing
        await task.cancel()

    runner = PipelineRunner()

    await runner.run(task)


shared_params = {
    "audio_in_enabled": True,
    "audio_out_enabled": True,
    "video_in_enabled": False,
    "video_out_enabled": False,
    "vad_enabled": True,
    "vad_analyzer": SileroVADAnalyzer(),
    "vad_audio_passthrough": True,
}


async def bot(args: SessionArguments):
    """Bot entry point compatible with Pipecat Cloud. SessionArguments
    will be a different subclass depending on how the session is started.

    args: either DailySessionArguments or WebsocketSessionArguments
    DailySessionArguments:
        room_url: The Daily room URL
        token: The Daily room token
        body: The configuration object from the request body
        session_id: The session ID for logging

    WebsocketSessionArguments:
        websocket: The websocket for connecting to Twilio
    """
    logger.info(f"Starting PCC bot. args: {args}")

    if isinstance(args, WebSocketSessionArguments):
        logger.debug("Starting WebSocket bot")

        start_data = args.websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        stream_sid = call_data["start"]["streamSid"]
        transport = FastAPIWebsocketTransport(
            websocket=args.websocket,
            params=FastAPIWebsocketParams(
                **shared_params,
                serializer=TwilioFrameSerializer(stream_sid),
            ),
        )
    elif isinstance(args, DailySessionArguments):
        logger.debug("Starting Daily bot")
        transport = DailyTransport(
            args.room_url,
            args.token,
            "Respond bot",
            DailyParams(**shared_params, transcription_enabled=False),
        )
    try:
        await main(transport)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise


# Local development
async def local_daily():
    """This is an entrypoint for running your bot locally but using Daily
    for the transport. To use this, you'll need to have DAILY_API_KEY set in your .env file.
    """
    try:
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)
            logger.warning(f"Talk to your voice agent here: {room_url}")
            webbrowser.open(room_url)
            transport = DailyTransport(
                room_url=room_url,
                token=token,
                bot_name="Bot",
                params=DailyParams(**shared_params, transcription_enabled=False),
            )
            await main(transport)
    except Exception as e:
        logger.exception(f"Error in local development mode: {e}")


async def local_webrtc(webrtc_connection):
    """An entrypoint for using the SmallWebRTCTransport, which doesn't require a Daily
    account or API key. You'll need to run the web client and small API server included
    with this example to use this transport. Run `python server.py` to use it.
    """
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection, params=TransportParams(**shared_params)
    )
    await main(transport)


# Local development entry point
if LOCAL_RUN and __name__ == "__main__":
    try:
        # Change this line to run whichever entrypoint you want to use for your bot.
        asyncio.run(local_daily())
    except Exception as e:
        logger.exception(f"Failed to run in local mode: {e}")
