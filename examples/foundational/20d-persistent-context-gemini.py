#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import glob
import json
import os
import sys
from datetime import datetime

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.google import GoogleLLMService

from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

video_participant_id = None


BASE_FILENAME = "/tmp/pipecat_conversation_"
tts = None


async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    temperature = 75 if args["format"] == "fahrenheit" else 24
    await result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": args["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_image(function_name, tool_call_id, arguments, llm, context, result_callback):
    question = arguments["question"]
    await llm.request_image_frame(user_id=video_participant_id, text_content=question)


async def get_saved_conversation_filenames(
    function_name, tool_call_id, args, llm, context, result_callback
):
    # Construct the full pattern including the BASE_FILENAME
    full_pattern = f"{BASE_FILENAME}*.json"

    # Use glob to find all matching files
    matching_files = glob.glob(full_pattern)
    logger.debug(f"matching files: {matching_files}")

    await result_callback({"filenames": matching_files})


async def save_conversation(function_name, tool_call_id, args, llm, context, result_callback):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"{BASE_FILENAME}{timestamp}.json"
    logger.debug(
        f"writing conversation to {filename}\n{json.dumps(context.get_messages_for_logging(), indent=4)}"
    )
    try:
        with open(filename, "w") as file:
            # todo: extract 'system' into the first message in the list
            messages = context.get_messages_for_persistent_storage()
            # remove the last message (the instruction to save the context)
            messages.pop()
            json.dump(messages, file, indent=2)
        await result_callback({"success": True})
    except Exception as e:
        logger.debug(f"error saving conversation: {e}")
        await result_callback({"success": False, "error": str(e)})


async def load_conversation(function_name, tool_call_id, args, llm, context, result_callback):
    global tts
    filename = args["filename"]
    logger.debug(f"loading conversation from {filename}")
    try:
        with open(filename, "r") as file:
            context.set_messages(json.load(file))
        await result_callback(
            {
                "success": True,
                "message": "The most recent conversation has been loaded. Awaiting further instructions.",
            }
        )
    except Exception as e:
        await result_callback({"success": False, "error": str(e)})


# Test message munging ...
messages = [
    {
        "role": "system",
        "content": """You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your
capabilities in a succinct way. Your output will be converted to audio so don't include special
characters in your answers. Respond to what the user said in a creative and helpful way.

You have several tools you can use to help you.

You can respond to questions about the weather using the get_weather tool.

You can save the current conversation using the save_conversation tool. This tool allows you to save
the current conversation to external storage. If the user asks you to save the conversation, use this
save_conversation too.

You can load a saved conversation using the load_conversation tool. This tool allows you to load a
conversation from external storage. You can get a list of conversations that have been saved using the
get_saved_conversation_filenames tool.

You can answer questions about the user's video stream using the get_image tool. Some examples of phrases that \
indicate you should use the get_image tool are:
  - What do you see?
  - What's in the video?
  - Can you describe the video?
  - Tell me about what you see.
  - Tell me something interesting about what you see.
  - What's happening in the video?
        """,
    },
    # {"role": "user", "content": ""},
    # {"role": "assistant", "content": []},
    # {"role": "user", "content": "Tell me"},
    # {"role": "user", "content": "a joke"},
]
tools = [
    {
        "function_declarations": [
            {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            },
            {
                "name": "save_conversation",
                "description": "Save the current conversation. Use this function to persist the current conversation to external storage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_request_text": {
                            "type": "string",
                            "description": "The text of the user's request to save the conversation.",
                        }
                    },
                    "required": ["user_request_text"],
                },
            },
            {
                "name": "get_saved_conversation_filenames",
                "description": "Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session.",
                "parameters": None,
            },
            {
                "name": "load_conversation",
                "description": "Load a conversation history. Use this function to load a conversation history into the current session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename of the conversation history to load.",
                        }
                    },
                    "required": ["filename"],
                },
            },
            {
                "name": "get_image",
                "description": "Get and image from the camera or video stream.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to to use when running inference on the acquired image.",
                        },
                    },
                    "required": ["question"],
                },
            },
        ]
    },
]


async def main():
    global tts
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
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        llm = GoogleLLMService(model="gemini-1.5-flash-latest", api_key=os.getenv("GOOGLE_API_KEY"))

        # you can either register a single function for all function calls, or specific functions
        # llm.register_function(None, fetch_weather_from_api)
        llm.register_function("get_current_weather", fetch_weather_from_api)
        llm.register_function("save_conversation", save_conversation)
        llm.register_function("get_saved_conversation_filenames", get_saved_conversation_filenames)
        llm.register_function("load_conversation", load_conversation)
        llm.register_function("get_image", get_image)

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                context_aggregator.user(),
                llm,  # LLM
                tts,
                context_aggregator.assistant(),
                transport.output(),  # Transport bot output
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                # report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            global video_participant_id
            video_participant_id = participant["id"]
            await transport.capture_participant_transcription(participant["id"])
            await transport.capture_participant_video(video_participant_id, framerate=0)
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
