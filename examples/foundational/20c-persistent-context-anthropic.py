#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import glob
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


BASE_FILENAME = "/tmp/pipecat_conversation_"
tts = None


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
        f"writing conversation to {filename}\n{json.dumps(params.context.get_messages(), indent=4)}"
    )
    try:
        with open(filename, "w") as file:
            # todo: extract 'system' into the first message in the list
            messages = params.context.get_messages()
            # remove the last message, which is the instruction we just gave to save the conversation
            messages.pop()
            json.dump(messages, file, indent=2)
        await params.result_callback({"success": True})
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


async def load_conversation(params: FunctionCallParams):
    global tts
    filename = params.arguments["filename"]
    logger.debug(f"loading conversation from {filename}")
    try:
        with open(filename, "r") as file:
            params.context.set_messages(json.load(file))
            logger.debug(
                f"loaded conversation from {filename}\n{json.dumps(params.context.get_messages(), indent=4)}"
            )
        await params.llm.queue_frame(TTSSpeakFrame("Ok, I've loaded that conversation."))
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


# Test message munging ...
messages = [
    {
        "role": "system",
        "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a succinct, creative and helpful way. Prefer responses that are one sentence long unless you are asked for a longer or more detailed response.",
    },
    {"role": "user", "content": "Start the call by saying the word 'hello'. Say only that word."},
    # {"role": "user", "content": ""},
    # {"role": "assistant", "content": []},
    # {"role": "user", "content": "Tell me"},
    # {"role": "user", "content": "a joke"},
]

weather_function = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather",
    properties={
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
    required=["location", "format"],
)

save_conversation_function = FunctionSchema(
    name="save_conversation",
    description="Save the current conversation. Use this function to persist the current conversation to external storage.",
    properties={},
    required=[],
)

get_filenames_function = FunctionSchema(
    name="get_saved_conversation_filenames",
    description="Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session.",
    properties={},
    required=[],
)

load_conversation_function = FunctionSchema(
    name="load_conversation",
    description="Load a conversation history. Use this function to load a conversation history into the current session.",
    properties={
        "filename": {
            "type": "string",
            "description": "The filename of the conversation history to load.",
        }
    },
    required=["filename"],
)

tools = ToolsSchema(
    standard_tools=[
        weather_function,
        save_conversation_function,
        get_filenames_function,
        load_conversation_function,
    ]
)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    global tts

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-latest"
    )

    # you can either register a single function for all function calls, or specific functions
    # llm.register_function(None, fetch_weather_from_api)
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("save_conversation", save_conversation)
    llm.register_function("get_saved_conversation_filenames", get_saved_conversation_filenames)
    llm.register_function("load_conversation", load_conversation)

    context = LLMContext(messages, tools)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),
            llm,  # LLM
            tts,
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
