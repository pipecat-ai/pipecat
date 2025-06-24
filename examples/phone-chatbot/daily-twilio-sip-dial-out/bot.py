#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""simple_dialout.py.

Simple Dial-out Bot.
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

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


async def run_bot(
    room_url: str,
    token: str,
    body: dict,
) -> None:
    """Run the voice bot with the given parameters.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        body: Body passed to the bot from the webhook

    """
    # ------------ CONFIGURATION AND SETUP ------------
    logger.info(f"Starting bot with room: {room_url}")
    logger.info(f"Token: {token}")
    logger.info(f"Body: {body}")
    # Parse the body to get the dial-in settings
    body_data = json.loads(body)

    # Check if the body contains dial-in settings
    logger.debug(f"Body data: {body_data}")

    if not body_data.get("dialout_settings"):
        logger.error("Dial-out settings not found in the body data")
        return

    dialout_settings = body_data["dialout_settings"]

    if not dialout_settings.get("sip_uri"):
        logger.error("Dial-out sip_uri not found in the dial-out settings")
        return

    # Extract sip_uri
    sip_uri = dialout_settings["sip_uri"]

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
        "Phone Bot",
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
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ------------ RETRY LOGIC VARIABLES ------------
    max_retries = 5
    retry_count = 0
    dialout_successful = False

    # Build dialout parameters conditionally
    dialout_params = {"sipUri": sip_uri}

    logger.debug(f"Dialout parameters: {dialout_params}")

    async def attempt_dialout():
        """Attempt to start dialout with retry logic."""
        nonlocal retry_count, dialout_successful

        if retry_count < max_retries and not dialout_successful:
            retry_count += 1
            logger.info(f"Attempting dialout (attempt {retry_count}/{max_retries}) to: {sip_uri}")
            await transport.start_dialout(dialout_params)
        else:
            logger.error(f"Maximum retry attempts ({max_retries}) reached. Giving up on dialout.")

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_joined")
    async def on_joined(transport, data):
        # Start initial dialout attempt
        logger.debug(f"Dialout settings detected; starting dialout to number: {sip_uri}")
        await attempt_dialout()

    @transport.event_handler("on_dialout_connected")
    async def on_dialout_connected(transport, data):
        logger.debug(f"Dial-out connected: {data}")

    @transport.event_handler("on_dialout_answered")
    async def on_dialout_answered(transport, data):
        nonlocal dialout_successful
        logger.debug(f"Dial-out answered: {data}")
        dialout_successful = True  # Mark as successful to stop retries
        # Automatically start capturing transcription for the participant
        await transport.capture_participant_transcription(data["sessionId"])
        # The bot will wait to hear the user before the bot speaks

    @transport.event_handler("on_dialout_error")
    async def on_dialout_error(transport, data: Any):
        logger.error(f"Dial-out error (attempt {retry_count}/{max_retries}): {data}")

        if retry_count < max_retries:
            logger.info(f"Retrying dialout")
            await attempt_dialout()
        else:
            logger.error(f"All {max_retries} dialout attempts failed. Stopping bot.")
            await task.cancel()

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


async def main():
    """Parse command line arguments and run the bot."""
    parser = argparse.ArgumentParser(description="Simple Dial-out Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    logger.debug(f"url: {args.url}")
    logger.debug(f"token: {args.token}")
    logger.debug(f"body: {args.body}")
    if not all([args.url, args.token, args.body]):
        logger.error("All arguments (-u, -t, -b) are required")
        parser.print_help()
        sys.exit(1)

    await run_bot(args.url, args.token, args.body)


if __name__ == "__main__":
    asyncio.run(main())
