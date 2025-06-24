#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""simple_dialin.py.

Daily PSTN Dial-in Bot.
"""

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
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

# Setup logging
load_dotenv()
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

    if not all([body_data.get("callId"), body_data.get("callDomain")]):
        logger.error("Call ID and Call Domain are required in the body.")
        return None

    call_id = body_data.get("callId")
    call_domain = body_data.get("callDomain")
    logger.debug(f"Call ID: {call_id}")
    logger.debug(f"Call Domain: {call_domain}")

    if not call_id or not call_domain:
        logger.error("Call ID and Call Domain are required for dial-in.")
        sys.exit(1)

    daily_dialin_settings = DailyDialinSettings(call_id=call_id, call_domain=call_domain)
    logger.debug(f"Dial-in settings: {daily_dialin_settings}")
    transport_params = DailyParams(
        api_url=daily_api_url,
        api_key=daily_api_key,
        dialin_settings=daily_dialin_settings,
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,
        vad_analyzer=SileroVADAnalyzer(),
        transcription_enabled=True,
    )
    logger.debug("setup transport params")

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-in Bot",
        transport_params,
    )
    logger.debug("setup transport")

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )
    logger.debug("setup tts")

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    logger.debug("setup llm")

    # Initialize LLM context with system prompt
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

    # Setup the conversational context
    context = OpenAILLMContext(messages)
    logger.debug("setup context")
    context_aggregator = llm.create_context_aggregator(context)
    logger.debug("setup context aggregator")

    # ------------ PIPELINE SETUP ------------

    # Build the pipeline
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
    logger.debug("setup pipeline")

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    logger.debug("setup task")

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()

    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(transport, cdata):
        logger.debug(f"Dial-in ready: {cdata}")

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
        logger.debug(f"Dial-in connected: {data}")

    @transport.event_handler("on_dialin_stopped")
    async def on_dialin_stopped(transport, data):
        logger.debug(f"Dial-in stopped: {data}")

    @transport.event_handler("on_dialin_error")
    async def on_dialin_error(transport, data):
        logger.error(f"Dial-in error: {data}")
        # If there is an error, the bot should leave the call
        # This may be also handled in on_participant_left with
        # await task.cancel()

    @transport.event_handler("on_dialin_warning")
    async def on_dialin_warning(transport, data):
        logger.warning(f"Dial-in warning: {data}")

    # Run the pipeline
    runner = PipelineRunner()
    await runner.run(task)


async def main():
    """Parse command line arguments and run the bot."""
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Daily room URL")
    parser.add_argument("-t", "--token", type=str, help="Daily room token")
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
