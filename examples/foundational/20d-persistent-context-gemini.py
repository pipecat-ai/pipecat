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
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import (
    create_transport,
    get_transport_client_id,
    maybe_capture_participant_camera,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

load_dotenv(override=True)


BASE_FILENAME = "/tmp/pipecat_conversation_"

# Global variable to store the client ID
client_id = ""


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


async def get_image(params: FunctionCallParams):
    question = params.arguments["question"]
    logger.debug(f"Requesting image with user_id={client_id}, question={question}")

    # Request the image frame
    await params.llm.request_image_frame(
        user_id=client_id,
        function_name=params.function_name,
        tool_call_id=params.tool_call_id,
        text_content=question,
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
            # remove the last message (the instruction to save the context)
            messages.pop()
            json.dump(messages, file, indent=2)
        await params.result_callback({"success": True})
    except Exception as e:
        logger.debug(f"error saving conversation: {e}")
        await params.result_callback({"success": False, "error": str(e)})


async def load_conversation(params: FunctionCallParams):
    filename = params.arguments["filename"]
    logger.debug(f"loading conversation from {filename}")
    try:
        with open(filename, "r") as file:
            params.context.set_messages(json.load(file))
        await params.result_callback(
            {
                "success": True,
                "message": "The most recent conversation has been loaded. Awaiting further instructions.",
            }
        )
    except Exception as e:
        await params.result_callback({"success": False, "error": str(e)})


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
    properties={
        "user_request_text": {
            "type": "string",
            "description": "The text of the user's request to save the conversation.",
        }
    },
    required=["user_request_text"],
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

get_image_function = FunctionSchema(
    name="get_image",
    description="Get and image from the camera or video stream.",
    properties={
        "question": {
            "type": "string",
            "description": "The question to to use when running inference on the acquired image.",
        },
    },
    required=["question"],
)

tools = ToolsSchema(
    standard_tools=[
        weather_function,
        save_conversation_function,
        get_filenames_function,
        load_conversation_function,
        get_image_function,
    ]
)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = GoogleLLMService(model="gemini-2.0-flash-001", api_key=os.getenv("GOOGLE_API_KEY"))

    # you can either register a single function for all function calls, or specific functions
    # llm.register_function(None, fetch_weather_from_api)
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("save_conversation", save_conversation)
    llm.register_function("get_saved_conversation_filenames", get_saved_conversation_filenames)
    llm.register_function("load_conversation", load_conversation)
    llm.register_function("get_image", get_image)

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

        await maybe_capture_participant_camera(transport, client)

        global client_id
        client_id = get_transport_client_id(transport, client)

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
