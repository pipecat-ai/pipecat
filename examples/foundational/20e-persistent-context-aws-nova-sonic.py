#
# Copyright (c) 2025, Daily
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

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
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


# async def get_saved_conversation_filenames(
#     function_name, tool_call_id, args, llm, context, result_callback
# ):
#     pattern = re.compile(re.escape(BASE_FILENAME) + "\\d{8}_\\d{6}\\.json$")
#     matching_files = []

#     for filename in os.listdir("."):
#         if pattern.match(filename):
#             matching_files.append(filename)

#     await result_callback({"filenames": matching_files})


async def save_conversation(params: FunctionCallParams):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"{BASE_FILENAME}{timestamp}.json"
    try:
        with open(filename, "w") as file:
            messages = params.context.get_messages_for_persistent_storage()
            # remove the last few messages. in reverse order, they are:
            # - the in progress save tool call
            # - the invocation of the save tool call
            # - the user ask to save (which may encompass one or more messages)
            # the simplest thing to do is to pop messages until the last one is an assistant
            # response
            while messages and not (
                messages[-1].get("role") == "assistant" and "content" in messages[-1]
            ):
                messages.pop()
            if messages:  # we never expect this to be empty
                logger.debug(
                    f"writing conversation to {filename}\n{json.dumps(messages, indent=4)}"
                )
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
                messages = json.load(file)
                messages.append(
                    {
                        "role": "user",
                        "content": f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}",
                    }
                )
                params.context.set_messages(messages)
                await params.llm.reset_conversation()
                await params.llm.trigger_assistant_response()
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})

    asyncio.create_task(_reset())


get_current_weather_tool = FunctionSchema(
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
            "description": "The temperature unit to use. Infer this from the user's location.",
        },
    },
    required=["location", "format"],
)

save_conversation_tool = FunctionSchema(
    name="save_conversation",
    description="Save the current conversation. Use this function to persist the current conversation to external storage.",
    properties={},
    required=[],
)

get_saved_conversation_filenames_tool = FunctionSchema(
    name="get_saved_conversation_filenames",
    description="Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session.",
    properties={},
    required=[],
)

load_conversation_tool = FunctionSchema(
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
        get_current_weather_tool,
        save_conversation_tool,
        get_saved_conversation_filenames_tool,
        load_conversation_tool,
    ]
)


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

    # Specify initial system instruction.
    # HACK: note that, for now, we need to inject a special bit of text into this instruction to
    # allow the first assistant response to be programmatically triggered (which happens in the
    # on_client_connected handler, below)
    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog exchanging "
        "the transcripts of a natural real-time conversation. Keep your responses short, generally "
        "two or three sentences for chatty scenarios. "
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )

    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        region=os.getenv("AWS_REGION"),  # as of 2025-05-06, us-east-1 is the only supported region
        voice_id="tiffany",  # matthew, tiffany, amy
        # you could choose to pass instruction here rather than via context
        # system_instruction=system_instruction,
        # you could choose to pass tools here rather than via context
        # tools=tools
    )

    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("save_conversation", save_conversation)
    llm.register_function("get_saved_conversation_filenames", get_saved_conversation_filenames)
    llm.register_function("load_conversation", load_conversation)

    context = OpenAILLMContext(
        messages=[
            {"role": "system", "content": f"{system_instruction}"},
        ],
        tools=tools,
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
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
        # HACK: for now, we need this special way of triggering the first assistant response in AWS
        # Nova Sonic. Note that this trigger requires a special corresponding bit of text in the
        # system instruction. In the future, simply queueing the context frame should be sufficient.
        await llm.trigger_assistant_response()

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
