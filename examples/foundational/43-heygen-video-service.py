#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.heygen.api import HeyGenApi, HeyGenSession, NewSessionRequest
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=True,
        video_out_is_live=True,
        video_out_width=1280,
        video_out_height=720,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")
    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a167e0f3-df7e-4d52-a9c3-f949145efdab",
        )

        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

        # TODO: need to add the HeyGenVideoService
        # For now just using it to test that the API it is working
        # We should move it to inside a HeyGenClient, like we did with Tavus
        heygen_api = HeyGenApi(api_key=os.getenv("HEYGEN_API_KEY"), session=session)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                # TODO: Add heygen video service
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=16000,
                audio_out_sample_rate=24000,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        _heygen_session: Optional[HeyGenSession] = None

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            nonlocal _heygen_session  # Add nonlocal declaration
            logger.info(f"Client connected")

            # TODO: only for testing
            _heygen_session = await heygen_api.new_session(
                NewSessionRequest(
                    avatarName="Shawn_Therapist_public",
                    version="v2",
                )
            )
            logger.info(f"heygen_session: {_heygen_session}")

            # Kick off the conversation.
            messages.append(
                {
                    "role": "system",
                    "content": "Start by greeting the user and ask how you can help.",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            nonlocal _heygen_session  # Add nonlocal declaration
            logger.info(f"Client disconnected")
            if _heygen_session:  # Add check for None
                logger.info(f"closing heygen_session: {_heygen_session.session_id}")
                await heygen_api.close_session(_heygen_session.session_id)
            await task.cancel()

        runner = PipelineRunner(handle_sigint=handle_sigint)

        await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
