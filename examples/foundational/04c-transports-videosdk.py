#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMRunFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InterruptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.videosdk.transport import  VideoSDKTransport,VideoSDKParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    if not os.getenv("VIDEOSDK_AUTH_TOKEN") or not os.getenv("VIDEOSDK_MEETING_ID"):
        logger.error("Please set VIDEOSDK_AUTH_TOKEN and VIDEOSDK_MEETING_ID in your .env file")
        return

    transport = VideoSDKTransport(
        params=VideoSDKParams(
            token=os.getenv("VIDEOSDK_AUTH_TOKEN"),
            meeting_id=os.getenv("VIDEOSDK_MEETING_ID"),
            name=os.getenv("NAME", "Pipecat-AI Bot"),
            audio_out_enabled=True,
            audio_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
            pubsub_topic="CHAT",  # Subscribe to pubsub topic for text messages (default: "CHAT").
        )
    )
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a VideoSDK meeting. Your goal is to demonstrate your capabilities in a succinct way. Your output will be spoken aloud, so avoid special characters that can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speech-to-Text (transcribe user audio)
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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
    async def on_first_participant_joined(transport, participant):
        # Kick off the conversation when first participant joins
        logger.info(f"First participant joined: {participant.id}")
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant):
        # When a participant leaves, cancel the task to stop the bot
        logger.info(f"Participant left: {participant.id}")
        await task.cancel()

    # Register an event handler to receive text messages from participants via PubSub
    # This allows users to send text messages that the bot can respond to with voice
    @transport.event_handler("on_pubsub_message_received")
    async def on_pubsub_message_received(transport, message, sender_id):
        logger.info(f"Received text message from {sender_id}: {message}")
        
        message_text = message.get("message", "")
        timestamp = message.get("timestamp", "")
        
        if not message_text:
            logger.warning("Received empty message")
            return
        
        logger.info(f"Processing text message: {message_text}")
        
        await task.queue_frames(
            [
                InterruptionFrame(),
                UserStartedSpeakingFrame(),
                TranscriptionFrame(
                    user_id=sender_id,
                    timestamp=timestamp,
                    text=message_text,
                ),
                UserStoppedSpeakingFrame(),
            ],
        )

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())