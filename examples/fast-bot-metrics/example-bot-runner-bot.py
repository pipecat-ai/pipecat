#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import aiohttp
import sys

from pydantic import BaseModel, ValidationError
from typing import Optional

from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.deepgram import DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport, DailyTransportMessageFrame
from pipecat.vad.silero import SileroVADAnalyzer

from loguru import logger

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class BotSettings(BaseModel):
    room_url: str
    room_token: str
    bot_name: str = "Pipecat"
    prompt: Optional[str] = "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Respond to what the user said in a creative and helpful way in a few short sentences."
    deepgram_api_key: Optional[str] = None
    deepgram_voice: Optional[str] = "aura-asteria-en"
    deepgram_base_url: Optional[str] = "https://api.deepgram.com/v1/speak"
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = "gpt-4o"
    openai_base_url: Optional[str] = None


async def main(settings: BotSettings):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            settings.room_url,
            settings.room_token,
            settings.bot_name,
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        tts = DeepgramTTSService(
            aiohttp_session=session,
            api_key=settings.deepgram_api_key,
            voice=settings.deepgram_voice,
            base_url=settings.deepgram_base_url
        )

        llm = OpenAILLMService(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url
        )

        messages = [
            {
                "role": "system",
                "content": settings.prompt,
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),   # Transport user input
            tma_in,              # User responses
            llm,                 # LLM
            tts,                 # TTS
            transport.output(),  # Transport bot output
            tma_out,             # Assistant spoken responses
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))

        # When the first participant joins, the bot should introduce itself.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Kick off the conversation.
            messages.append(
                {"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frame(LLMMessagesFrame(messages))

        # When a participant joins, start transcription for that participant so the
        # bot can "hear" and respond to them.
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])

        # When the participant leaves, we exit the bot.
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        # If the call is ended make sure we quit as well.
        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        # Handle "latency-ping" messages. The client will send app messages that look like
        # this:
        #   { "latency-ping": { ts: <client-side timestamp> }}
        #
        # We want to send an immediate pong back to the client from this handler function.
        # Also, we will push a frame into the top of the pipeline and send it after the
        #
        @transport.event_handler("on_app_message")
        async def on_app_message(transport, message, sender):
            try:
                if "latency-ping" in message:
                    logger.debug(f"Received latency ping app message: {message}")
                    ts = message["latency-ping"]["ts"]
                    # Send immediately
                    transport.output().send_message(DailyTransportMessageFrame(
                        message={"latency-pong-msg-handler": {"ts": ts}},
                        participant_id=sender))
                    # And push to the pipeline for the Daily transport.output to send
                    await tma_in.push_frame(
                        DailyTransportMessageFrame(
                            message={"latency-pong-pipeline-delivery": {"ts": ts}},
                            participant_id=sender))
            except Exception as e:
                logger.debug(f"message handling error: {e} - {message}")

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot")
    parser.add_argument("-s", "--settings", type=str, required=True, help="Pipecat bot settings")

    args, unknown = parser.parse_known_args()

    try:
        settings = BotSettings.model_validate_json(args.settings)
        asyncio.run(main(settings))
    except ValidationError as e:
        print(e)
