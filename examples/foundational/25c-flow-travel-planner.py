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

# Flow Configuration - Travel Planner
#
# This configuration defines a vacation planning system with the following states:
#
# 1. start
#    - Initial state where user chooses between beach or mountain vacation
#    - Functions: choose_beach, choose_mountain
#    - Pre-action: Welcome message
#    - Transitions to: choose_beach or choose_mountain
#
# 2. choose_beach/choose_mountain
#    - Handles destination selection for chosen vacation type
#    - Functions:
#      * select_destination (terminal function with location-specific options)
#      * get_dates (transitions to date selection)
#    - Pre-action: Destination-specific welcome message
#
# 3. get_dates
#    - Handles travel date selection
#    - Functions:
#      * record_dates (terminal function, can be modified)
#      * get_activities (transitions to activity selection)
#
# 4. get_activities
#    - Handles activity preference selection
#    - Functions:
#      * record_activities (terminal function, array-based selection)
#      * verify_itinerary (transitions to verification)
#
# 5. verify_itinerary
#    - Reviews complete vacation plan
#    - Functions:
#      * revise_plan (loops back to get_dates)
#      * confirm_booking (transitions to confirmation)
#
# 6. confirm_booking
#    - Handles final confirmation and tips
#    - Functions: end
#    - Pre-action: Confirmation message
#
# 7. end
#    - Final state that closes the conversation
#    - No functions available
#    - Post-action: Ends conversation

flow_config = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "messages": [
                {
                    "role": "system",
                    "content": "For this step, ask if they're interested in planning a beach vacation or a mountain retreat, and wait for them to choose. Start with an enthusiastic greeting and be conversational; you're helping them plan their dream vacation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_beach",
                        "description": "User wants to plan a beach vacation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "choose_mountain",
                        "description": "User wants to plan a mountain retreat",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {
                    "type": "tts_say",
                    "text": "Welcome to Dream Vacations! I'll help you plan your perfect getaway.",
                }
            ],
        },
        "choose_beach": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are handling beach vacation planning. Use the available functions:\n - Use select_destination when the user chooses their preferred beach location\n - Use get_dates once they've selected a destination\n\nAvailable beach destinations are: 'Maui', 'Cancun', or 'Maldives'. After they choose, confirm their selection and proceed to dates. Be enthusiastic and paint a picture of each destination.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_destination",
                        "description": "Record the selected beach destination",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "destination": {
                                    "type": "string",
                                    "enum": ["Maui", "Cancun", "Maldives"],
                                    "description": "Selected beach destination",
                                }
                            },
                            "required": ["destination"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_dates",
                        "description": "Proceed to date selection",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Let's find your perfect beach paradise..."}
            ],
        },
        "choose_mountain": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are handling mountain retreat planning. Use the available functions:\n - Use select_destination when the user chooses their preferred mountain location\n - Use get_dates once they've selected a destination\n\nAvailable mountain destinations are: 'Swiss Alps', 'Rocky Mountains', or 'Himalayas'. After they choose, confirm their selection and proceed to dates. Be enthusiastic and paint a picture of each destination.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_destination",
                        "description": "Record the selected mountain destination",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "destination": {
                                    "type": "string",
                                    "enum": ["Swiss Alps", "Rocky Mountains", "Himalayas"],
                                    "description": "Selected mountain destination",
                                }
                            },
                            "required": ["destination"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_dates",
                        "description": "Proceed to date selection",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Let's find your perfect mountain getaway..."}
            ],
        },
        "get_dates": {
            "messages": [
                {
                    "role": "system",
                    "content": "Handle travel date selection. Use the available functions:\n - Use record_dates when the user specifies their travel dates (can be used multiple times if they change their mind)\n - Use get_activities once dates are confirmed\n\nAsk for their preferred travel dates within the next 6 months. After recording dates, confirm the selection and proceed to activities.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_dates",
                        "description": "Record the selected travel dates",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "check_in": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "Check-in date (YYYY-MM-DD)",
                                },
                                "check_out": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "Check-out date (YYYY-MM-DD)",
                                },
                            },
                            "required": ["check_in", "check_out"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_activities",
                        "description": "Proceed to activity selection",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "get_activities": {
            "messages": [
                {
                    "role": "system",
                    "content": "Handle activity preferences. Use the available functions:\n - Use record_activities to save their activity preferences\n - Use verify_itinerary once activities are selected\n\nFor beach destinations, suggest: snorkeling, surfing, sunset cruise\nFor mountain destinations, suggest: hiking, skiing, mountain biking\n\nAfter they choose, confirm their selections and proceed to verification.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_activities",
                        "description": "Record selected activities",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "activities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": 3,
                                    "description": "Selected activities",
                                }
                            },
                            "required": ["activities"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "verify_itinerary",
                        "description": "Proceed to itinerary verification",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "verify_itinerary": {
            "messages": [
                {
                    "role": "system",
                    "content": "Review the complete itinerary with the user. Summarize their destination, dates, and chosen activities. Use the available functions:\n - Use get_dates if they want to make changes\n - Use confirm_booking if they're happy with everything\n\nBe thorough in reviewing all details and ask for their confirmation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_dates",
                        "description": "Return to date selection to revise the plan",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "confirm_booking",
                        "description": "Confirm the booking and proceed to end",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "confirm_booking": {
            "messages": [
                {
                    "role": "system",
                    "content": "The booking is confirmed. Share some relevant tips about their chosen destination, thank them warmly, and use end to complete the conversation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "End the conversation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Fantastic! Your dream vacation is confirmed!"}
            ],
        },
        "end": {
            "messages": [
                {
                    "role": "system",
                    "content": "Wish them a wonderful trip and end the conversation.",
                }
            ],
            "functions": [],
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
                "content": "You are a travel planning assistant. You must ALWAYS use one of the available functions to progress the conversation. This is a phone conversations and your responses will be converted to audio. Avoid outputting special characters and emojis.",
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
