#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from livekit import api
from loguru import logger

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def generate_token(room_name: str, participant_name: str, api_key: str, api_secret: str) -> str:
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
        )
    )

    return token.to_jwt()


async def configure_livekit():
    parser = argparse.ArgumentParser(description="LiveKit AI SDK Bot Sample")
    parser.add_argument(
        "-r", "--room", type=str, required=False, help="Name of the LiveKit room to join"
    )
    parser.add_argument("-u", "--url", type=str, required=False, help="URL of the LiveKit server")

    args, unknown = parser.parse_known_args()

    room_name = args.room or os.getenv("LIVEKIT_ROOM_NAME")
    url = args.url or os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not room_name:
        raise Exception(
            "No LiveKit room specified. Use the -r/--room option from the command line, or set LIVEKIT_ROOM_NAME in your environment."
        )

    if not url:
        raise Exception(
            "No LiveKit server URL specified. Use the -u/--url option from the command line, or set LIVEKIT_URL in your environment."
        )

    if not api_key or not api_secret:
        raise Exception(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment variables."
        )

    token = generate_token(room_name, "Say One Thing", api_key, api_secret)

    user_token = generate_token(room_name, "User", api_key, api_secret)
    logger.info(f"User token: {user_token}")

    return (url, token, room_name)


async def main():
    async with aiohttp.ClientSession() as session:
        (url, token, room_name) = await configure_livekit()

        transport = LiveKitTransport(
            url=url,
            token=token,
            room_name=room_name,
            params=LiveKitParams(audio_out_enabled=True),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        runner = PipelineRunner()

        task = PipelineTask(Pipeline([tts, transport.output()]))

        # Register an event handler so we can play the audio when the
        # participant joins.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant_id):
            await asyncio.sleep(1)
            await task.queue_frame(
                TextFrame(
                    "Hello there! How are you doing today? Would you like to talk about the weather?"
                )
            )

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
