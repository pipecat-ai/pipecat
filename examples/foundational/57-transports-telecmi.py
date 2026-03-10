#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
TeleCMI Transport Example

This example demonstrates how to build a voice agent that connects to phone calls
using the TeleCMI (PSTN) transport.

For instructions on how to install the `pipecat-ai[telecmi]` dependencies, buy a number,
and configure this example, please see the TeleCMI Transport README:
https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/transports/telecmi
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaSTTSettings
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.openai.llm import OpenAILLMService, OpenAILLMSettings
from pipecat.transports.telecmi import TelecmiAPI, TelecmiParams, TelecmiTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def create_session(url: str, token: str, room_name: str, metadata: dict, **kwargs):
    """
    Fired dynamically by the TelecmiAPI when a new call connects.
    """
    logger.info(f"Incoming call. Room: {room_name}")

    # Optional metadata passed via TeleCMI
    metadata = metadata or {}
    customer_name = metadata.get("customer_name", "there")

    transport = TelecmiTransport(
        url=url,
        token=token,
        room_name=room_name,
        params=TelecmiParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY"))

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMSettings(
            system_instruction="You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSSettings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            user_aggregator,  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            assistant_aggregator,  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        await asyncio.sleep(1)
        await task.queue_frame(
            TTSSpeakFrame(
                "Hello there! How are you doing today? Would you like to talk about the weather?"
            )
        )

    runner = PipelineRunner(handle_sigint=False)

    logger.info(f"Starting Pipecat Pipeline runner over TeleCMI for room {room_name}...")

    await runner.run(task)


async def main():
    agent_id = os.getenv("AGENT_ID")
    agent_token = os.getenv("AGENT_TOKEN")

    if not agent_id or not agent_token:
        logger.error("Missing AGENT_ID or AGENT_TOKEN inside .env")
        sys.exit(1)

    # The TelecmiAPI handles the Socket.IO connection to receive inbound calls
    # and spawns 'create_session' continuously as calls arrive.
    agent = TelecmiAPI(
        agent_id=agent_id,
        agent_token=agent_token,
        create_session=create_session,
        debug=True,
    )

    logger.info(f"🚀 TeleCMI API connecting for ID: {agent_id}")
    await agent.connect()


if __name__ == "__main__":
    asyncio.run(main())
