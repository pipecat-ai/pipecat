#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    body_data = json.loads(body)

    dialout_settings = body_data["dialout_settings"]

    # ------------ TRANSPORT SETUP ------------

    transport_params = DailyParams(
        api_url=daily_api_url,
        api_key=daily_api_key,
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        vad_analyzer=SileroVADAnalyzer(),
        transcription_enabled=True,
    )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-out Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ LLM AND CONTEXT SETUP ------------

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Create system message and initialize messages list
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly phone assistant. Your responses will be read aloud, "
                "so keep them concise and conversational. Avoid special characters or "
                "formatting. Begin by greeting the caller and asking how you can help them today."
            ),
        },
    ]
    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        # Start dialout if needed
        logger.debug("Dialout settings detected; starting dialout")
        await transport.start_dialout(transport, dialout_settings)

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.debug(f"Dial-out connected: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        logger.debug(f"Dial-out answered: {data}")
        # Automatically start capturing transcription for the participant
        await transport.capture_participant_transcription(data["sessionId"])
        # The bot will wait to hear the user before the bot speaks

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-out Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
