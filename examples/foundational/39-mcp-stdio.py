#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import os
import re
import shutil
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from mcp import StdioServerParameters
from PIL import Image

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    URLImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.mcp_service import MCPClient
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


class UrlToImageProcessor(FrameProcessor):
    def __init__(self, aiohttp_session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self._aiohttp_session = aiohttp_session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallResultFrame):
            await self.push_frame(frame, direction)
            image_url = self.extract_url(frame.result)
            if image_url:
                await self.run_image_process(image_url)
                # sometimes we get multiple image urls- process 1 at a time
                await asyncio.sleep(1)
        else:
            await self.push_frame(frame, direction)

    def extract_url(self, text: str):
        pattern = r"!\[[^\]]*\]\((https?://[^)]+\.(png|jpg|jpeg|PNG|JPG|JPEG))\)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    async def run_image_process(self, image_url: str):
        try:
            logger.debug(f"handling image from url: '{image_url}'")
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                image = image.convert("RGB")
                frame = URLImageRawFrame(
                    url=image_url, image=image.tobytes(), size=image.size, format="RGB"
                )
                await self.push_frame(frame)
        except Exception as e:
            error_msg = f"Error handling image url {image_url}: {str(e)}"
            logger.error(error_msg)


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_width=1024,
            video_out_height=1024,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Create an HTTP session for API calls
    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-7-sonnet-latest"
        )

        try:
            mcp = MCPClient(
                server_params=StdioServerParameters(
                    command=shutil.which("npx"),
                    args=["-y", "@programcomputer/nasa-mcp-server@latest"],
                    # https://api.nasa.gov
                    env={"NASA_API_KEY": os.getenv("NASA_API_KEY")},
                )
            )
        except Exception as e:
            logger.error(f"error setting up mcp")
            logger.exception("error trace:")

        mcp_image = UrlToImageProcessor(aiohttp_session=session)

        tools = await mcp.register_tools(llm)

        system = f"""
        You are a helpful LLM in a WebRTC call. 
        Your goal is to demonstrate your capabilities in a succinct way. 
        You have access to a number of tools provided by NASA MCP. Use any and all tools to help users.
        When asked for the astronomy picture of the day, PASS in NO date to the API.
        This ensures we get the latest picture available. If as specific date is asked for, you
        can pass in that date to the API.
        Your output will be converted to audio so don't include special characters in your answers. 
        Respond to what the user said in a creative and helpful way. 
        Don't overexplain what you are doing. 
        Just respond with short sentences when you are carrying out tool calls.
        """

        messages = [{"role": "system", "content": system}]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,
                context_aggregator.user(),  # User spoken responses
                llm,  # LLM
                tts,  # TTS
                mcp_image,  # URL image -> output
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses and tool context
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
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
