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
from pipecat_flows import FlowManager
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Movie Booking
#
# This configuration defines a movie ticket booking system with the following states:
#
# 1. start
#    - Initial state where user chooses between today or tomorrow's showings
#    - Functions: check_today, check_tomorrow
#    - Pre-action: Welcome message
#    - Transitions to: check_today or check_tomorrow
#
# 2. check_today
#    - Handles movie selection for today's showings
#    - Functions:
#      * select_movie (terminal function with today's movies)
#      * select_showtime (terminal function with available times)
#      * end (transitions to end node after confirmation)
#    - Pre-action: Today's movie listing message
#
# 3. check_tomorrow
#    - Handles movie selection for tomorrow's showings
#    - Functions:
#      * select_movie (terminal function with tomorrow's movies)
#      * select_showtime (terminal function with available times)
#      * end (transitions to end node after confirmation)
#    - Pre-action: Tomorrow's movie listing message
#
# 4. end
#    - Final state that closes the conversation
#    - No functions available
#    - Pre-action: Ticket confirmation message
#    - Post-action: Ends conversation
#
# Note: Both check_today and check_tomorrow allow multiple selections
# until the user confirms their final choice

flow_config = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "messages": [
                {
                    "role": "system",
                    "content": "For this step, ask if they want to see what's playing today or tomorrow, and wait for them to choose. Start with a warm greeting and be helpful and enthusiastic; you're helping them plan their entertainment.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "check_today",
                        "description": "User wants to see today's movies",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "check_tomorrow",
                        "description": "User wants to see tomorrow's movies",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {
                    "type": "tts_say",
                    "text": "Welcome to MoviePlex! Let me help you book some tickets.",
                }
            ],
        },
        "check_today": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are handling today's movie selection. Use the available functions:\n - Use select_movie when the user chooses a movie (can be used multiple times if they change their mind)\n - Use select_showtime after they've chosen a movie to pick their preferred time\n - Use the end function ONLY when the user confirms their final selection\n\nAfter each selection, confirm their choice and ask about the next step. Remember to be enthusiastic and helpful.\n\nStart by telling them today's available movies: 'Jurassic Park' at 3:00 PM and 7:00 PM, or 'The Matrix' at 4:00 PM and 8:00 PM.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_movie",
                        "description": "Record the selected movie",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "movie": {
                                    "type": "string",
                                    "enum": ["Jurassic Park", "The Matrix"],
                                    "description": "Selected movie",
                                }
                            },
                            "required": ["movie"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "select_showtime",
                        "description": "Record the selected showtime",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time": {
                                    "type": "string",
                                    "enum": ["3:00 PM", "4:00 PM", "7:00 PM", "8:00 PM"],
                                    "description": "Selected showtime",
                                }
                            },
                            "required": ["time"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "Complete the booking (use only after user confirms)",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [{"type": "tts_say", "text": "Let me show you what's playing today..."}],
        },
        "check_tomorrow": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are handling tomorrow's movie selection. Use the available functions:\n - Use select_movie when the user chooses a movie (can be used multiple times if they change their mind)\n - Use select_showtime after they've chosen a movie to pick their preferred time\n - Use the end function ONLY when the user confirms their final selection\n\nAfter each selection, confirm their choice and ask about the next step. Remember to be enthusiastic and helpful.\n\nStart by telling them tomorrow's available movies: 'The Lion King' at 2:00 PM and 6:00 PM, or 'Inception' at 3:00 PM and 7:00 PM.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_movie",
                        "description": "Record the selected movie",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "movie": {
                                    "type": "string",
                                    "enum": ["The Lion King", "Inception"],
                                    "description": "Selected movie",
                                }
                            },
                            "required": ["movie"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "select_showtime",
                        "description": "Record the selected showtime",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time": {
                                    "type": "string",
                                    "enum": ["2:00 PM", "3:00 PM", "6:00 PM", "7:00 PM"],
                                    "description": "Selected showtime",
                                }
                            },
                            "required": ["time"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "Complete the booking (use only after user confirms)",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Let me show you what's playing tomorrow..."}
            ],
        },
        "end": {
            "messages": [
                {
                    "role": "system",
                    "content": "The booking is complete. Thank the user enthusiastically and end the conversation.",
                }
            ],
            "functions": [],
            "pre_actions": [
                {"type": "tts_say", "text": "Your tickets are confirmed! Enjoy the show!"}
            ],
            "post_actions": [{"type": "end_conversation"}],
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
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Get initial tools from the first node
        initial_tools = flow_config["nodes"]["start"]["functions"]

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are a movie ticket booking assistant. You must ALWAYS use one of the available functions to progress the conversation. This is a phone conversations and your responses will be converted to audio. Avoid outputting special characters and emojis.",
            }
        ]

        context = OpenAILLMContext(messages, initial_tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(flow_config, task, tts)

        # Register functions with LLM service
        await flow_manager.register_functions(llm)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Initialize the flow processor
            await flow_manager.initialize(messages)
            # Kick off the conversation using the context aggregator
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
