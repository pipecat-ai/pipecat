#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
    DailyTransportMessageFrame,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = DeepgramTTSService(
            aiohttp_session=session,
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            voice="aura-asteria-en",
            base_url="http://0.0.0.0:8080/v1/speak",
        )

        llm = OpenAILLMService(
            # To use OpenAI
            # api_key=os.getenv("OPENAI_API_KEY"),
            # model="gpt-4o"
            # Or, to use a local vLLM (or similar) api server
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            base_url="http://0.0.0.0:8000/v1",
        )

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
                context_aggregator.user(),
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))

        # When a participant joins, start transcription for that participant so the
        # bot can "hear" and respond to them.
        @transport.event_handler("on_participant_joined")
        async def on_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])

        # When the first participant joins, the bot should introduce itself.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

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
                    transport.output().send_message(
                        DailyTransportMessageFrame(
                            message={"latency-pong-msg-handler": {"ts": ts}}, participant_id=sender
                        )
                    )
                    # And push to the pipeline for the Daily transport.output to send
                    await task.queue_frame(
                        DailyTransportMessageFrame(
                            message={"latency-pong-pipeline-delivery": {"ts": ts}},
                            participant_id=sender,
                        )
                    )
            except Exception as e:
                logger.debug(f"message handling error: {e} - {message}")

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
