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
                "role": "system",
                "content": "You are an order-taking assistant. You must ALWAYS use one of the available functions to progress the conversation. For this step, ask the user if they want pizza or sushi, and wait for them to use a function to choose.",
            },
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_pizza",
                        "description": "User wants to order pizza",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "choose_sushi",
                        "description": "User wants to order sushi",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "choose_pizza": {
            "message": {
                "role": "system",
                "content": "The user has chosen pizza. You must now ask them to select a size using the select_pizza_size function. Do not proceed until they use this function. Do not assume any selections have been made.",
            },
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_pizza_size",
                        "description": "Select pizza size",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {
                                    "type": "string",
                                    "enum": ["small", "medium", "large"],
                                    "description": "Size of the pizza",
                                }
                            },
                            "required": ["size"],
                        },
                    },
                }
            ],
            "actions": [{"type": "tts.say", "text": "Let me help you order a pizza..."}],
        },
        "choose_sushi": {
            "message": {
                "role": "system",
                "content": "The user has chosen sushi. Immediately ask them: 'How many sushi rolls would you like to order?' If they answer provide to the question of how many rolls, use the select_roll_count function.",
            },
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_roll_count",
                        "description": "Select number of sushi rolls",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Number of rolls to order",
                                }
                            },
                            "required": ["count"],
                        },
                    },
                }
            ],
            "actions": [{"type": "tts.say", "text": "Ok, one moment..."}],
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

        # Get initial tools from the first node
        initial_tools = flow_config["nodes"]["start"]["functions"]

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. Never assume an order is complete without the proper function calls. Your responses will be converted to audio so avoid special characters.",
            }
        ]

        # Register function handlers
        async def handle_function_call(
            function_name, tool_call_id, arguments, llm, context, result_callback
        ):
            logger.info(f"Function called: {function_name} with arguments: {arguments}")
            # Handle the state transition
            await flow_processor.handle_transition(function_name)
            # Send the acknowledgment
            await result_callback("Acknowledged")
            logger.info(f"Function call result sent: {function_name}")

        # Register functions from all nodes
        for node in flow_config["nodes"].values():
            for function in node["functions"]:
                function_name = function["function"]["name"]
                llm.register_function(function_name, handle_function_call)

        context = OpenAILLMContext(messages, initial_tools)
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
            # Kick off the conversation using the context aggregator
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
