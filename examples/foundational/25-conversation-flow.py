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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.conversation_flow import ConversationFlowProcessor
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Define our conversation flow
flow_config = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "message": {
                "role": "assistant",
                "content": "You are starting a conversation. Ask the user if they'd like to hear a joke or get weather information.",
            },
            "functions": [
                {
                    "name": "tell_joke",
                    "description": "User wants to hear a joke",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "get_weather",
                    "description": "User wants weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ],
        },
        "tell_joke": {
            "message": {
                "role": "assistant",
                "content": "Tell a funny, clean joke and then ask if they'd like to hear another joke or get weather information.",
            },
            "functions": [
                {
                    "name": "tell_joke",
                    "description": "User wants another joke",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "get_weather",
                    "description": "User wants weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ],
            "actions": [{"type": "tts.say", "text": "Let me think of a good one..."}],
        },
        "get_weather": {
            "message": {
                "role": "assistant",
                "content": "Provide the weather information and ask if they'd like to hear a joke or check another location's weather.",
            },
            "functions": [
                {
                    "name": "tell_joke",
                    "description": "User wants to hear a joke",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "get_weather",
                    "description": "User wants weather for another location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            ],
            "actions": [
                {"type": "tts.say", "text": "Let me check that weather information for you..."}
            ],
        },
    },
}


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

        # Initialize conversation flow processor
        flow_processor = ConversationFlowProcessor(flow_config)

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant in a WebRTC call. Your responses will be converted to audio so avoid special characters. Always use the available functions to progress the conversation.",
            }
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                flow_processor,  # Conversation flow management
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Initialize the flow processor
            await flow_processor.initialize(messages)
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
