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

import aiohttp
from deepgram import LiveOptions
from dotenv import load_dotenv
from livekit import api
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
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


def generate_token_with_agent(
    room_name: str, participant_name: str, api_key: str, api_secret: str
) -> str:
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
            agent=True,  # This is the only difference, this makes livekit client know agent has joined
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

    token = generate_token_with_agent(room_name, "Say One Thing", api_key, api_secret)

    user_token = generate_token(room_name, "User", api_key, api_secret)
    logger.info(f"User token: {user_token}")

    return url, token, room_name


async def main():
    async with aiohttp.ClientSession() as session:
        (url, token, room_name) = await configure_livekit()

        transport = LiveKitTransport(
            url=url,
            token=token,
            room_name=room_name,
            params=LiveKitParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_enabled=True,
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                vad_events=True,
            ),
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. "
                "Your goal is to demonstrate your capabilities in a succinct way. "
                "Your output will be converted to audio so don't include special characters in your answers. "
                "Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        runner = PipelineRunner()

        task = PipelineTask(
            Pipeline(
                [
                    transport.input(),
                    stt,
                    context_aggregator.user(),
                    llm,
                    tts,
                    transport.output(),
                    context_aggregator.assistant(),
                ],
            ),
            params=PipelineParams(
                allow_interruptions=True, enable_metrics=True, enable_usage_metrics=True
            ),
        )

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

        # Register an event handler to receive data from the participant via text chat
        # in the LiveKit room. This will be used to as transcription frames and
        # interrupt the bot and pass it to llm for processing and
        # then pass back to the participant as audio output.
        @transport.event_handler("on_data_received")
        async def on_data_received(transport, data, participant_id):
            logger.info(f"Received data from participant {participant_id}: {data}")
            # convert data from bytes to string
            json_data = json.loads(data)

            await task.queue_frames(
                [
                    BotInterruptionFrame(),
                    UserStartedSpeakingFrame(),
                    TranscriptionFrame(
                        user_id=participant_id,
                        timestamp=json_data["timestamp"],
                        text=json_data["message"],
                    ),
                    UserStoppedSpeakingFrame(),
                ],
            )

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
