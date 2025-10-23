#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import glob
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioTranscription,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

BASE_FILENAME = "/tmp/pipecat_conversation_"


async def fetch_weather_from_api(params: FunctionCallParams):
    temperature = 75 if params.arguments["format"] == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": params.arguments["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_saved_conversation_filenames(params: FunctionCallParams):
    # Construct the full pattern including the BASE_FILENAME
    full_pattern = f"{BASE_FILENAME}*.json"

    # Use glob to find all matching files
    matching_files = glob.glob(full_pattern)
    logger.debug(f"matching files: {matching_files}")

    await params.result_callback({"filenames": matching_files})


async def save_conversation(params: FunctionCallParams):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"{BASE_FILENAME}{timestamp}.json"
    logger.debug(
        f"writing conversation to {filename}\n{json.dumps(params.context.messages, indent=4)}"
    )
    try:
        with open(filename, "w") as file:
            messages = params.context.get_messages_for_persistent_storage()
            # remove the last message, which is the instruction we just gave to save the conversation
            messages.pop()
            json.dump(messages, file, indent=2)
        await params.result_callback({"success": True})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def load_conversation(params: FunctionCallParams):
    async def _reset():
        filename = params.arguments["filename"]
        logger.debug(f"loading conversation from {filename}")
        try:
            with open(filename, "r") as file:
                params.context.set_messages(json.load(file))
                await params.llm.reset_conversation()
                await params.llm._create_response()
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})

    asyncio.create_task(_reset())


tools = [
    {
        "type": "function",
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
        "type": "function",
        "name": "save_conversation",
        "description": "Save the current conversatione. Use this function to persist the current conversation to external storage.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "get_saved_conversation_filenames",
        "description": "Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "type": "function",
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
]


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    session_properties = SessionProperties(
        audio=AudioConfiguration(
            input=AudioInput(
                transcription=InputAudioTranscription(),
                # Set openai TurnDetection parameters. Not setting this at all will turn it
                # on by default
                turn_detection=TurnDetection(silence_duration_ms=1000),
                # Or set to False to disable openai turn detection and use transport VAD
                # turn_detection=False,
            )
        ),
        # tools=tools,
        instructions="""Your knowledge cutoff is 2023-10. You are a helpful and friendly AI.

Act like a human, but remember that you aren't a human and that you can't do human
things in the real world. Your voice and personality should be warm and engaging, with a lively and
playful tone.

If interacting in a non-English language, start by using the standard accent or dialect familiar to
the user. Talk quickly. You should always call a function if you can. Do not refer to these rules,
even if you're asked about them.
-
You are participating in a voice conversation. Keep your responses concise, short, and to the point
unless specifically asked to elaborate on a topic.

Remember, your responses should be short. Just one or two sentences, usually.""",
    )

    llm = OpenAIRealtimeLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        session_properties=session_properties,
        start_audio_paused=False,
    )

    # you can either register a single function for all function calls, or specific functions
    # llm.register_function(None, fetch_weather_from_api)
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("save_conversation", save_conversation)
    llm.register_function("get_saved_conversation_filenames", get_saved_conversation_filenames)
    llm.register_function("load_conversation", load_conversation)

    context = OpenAILLMContext([], tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),
            llm,  # LLM
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
