#
# Copyright (c) 2025-2026, Daily
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

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

BASE_FILENAME = "/tmp/pipecat_conversation_"


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_saved_conversation_filenames(params: FunctionCallParams):
    """Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session."""
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
    """Save the current conversation. Use this function to persist the current conversation to external storage."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    filename = f"{BASE_FILENAME}{timestamp}.json"
    try:
        with open(filename, "w") as file:
            messages = params.context.get_messages()
            # remove the last few messages. in reverse order, they are:
            # - the in progress save tool call
            # - the invocation of the save tool call
            # - the user ask to save (which may encompass one or more messages)
            # the simplest thing to do is to pop messages until the last one is an assistant
            # response
            while messages and not (
                isinstance(messages[-1], dict)
                and messages[-1].get("role") == "assistant"
                and "content" in messages[-1]
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


async def load_conversation(params: FunctionCallParams, filename: str):
    """Load a conversation history. Use this function to load a conversation history into the current session.

    Args:
        filename: The filename of the conversation history to load.
    """

    async def _reset():
        logger.debug(f"loading conversation from {filename}")
        try:
            with open(filename) as file:
                messages = json.load(file)
                # HACK: if using the older Nova Sonic (pre-2) model, you need a special way of
                # triggering the first assistant response. The call to trigger_assistant_response(),
                # commented out below, is part of this.
                # messages.append(
                #     {
                #         "role": "developer",
                #         "content": f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}",
                #     }
                # )
                # If the last message isn't from the user, add a message asking for a recap
                if messages and messages[-1].get("role") != "user":
                    messages.append(
                        {
                            "role": "user",
                            "content": "Can you catch me up on what we were talking about?",
                        }
                    )
                params.context.set_messages(messages)
                assert isinstance(params.llm, AWSNovaSonicLLMService)
                await params.llm.reset_conversation()
                # await params.llm.trigger_assistant_response()
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})

    asyncio.create_task(_reset())


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # Specify initial system instruction.
    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog exchanging "
        "the transcripts of a natural real-time conversation. Keep your responses short, generally "
        "two or three sentences for chatty scenarios. "
        # HACK: if using the older Nova Sonic (pre-2) model, note that you need to inject a special
        # bit of text into this instruction to allow the first assistant response to be
        # programmatically triggered (which happens in the on_client_connected handler)
        # f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )

    llm = AWSNovaSonicLLMService(
        secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        region=os.environ["AWS_REGION"],  # as of 2025-05-06, us-east-1 is the only supported region
        settings=AWSNovaSonicLLMService.Settings(
            voice="tiffany",  # matthew, tiffany, amy
            system_instruction=system_instruction,
        ),
        # you could choose to pass tools here rather than via context
        # tools=tools
    )

    context = LLMContext(
        tools=[
            get_current_weather,
            save_conversation,
            get_saved_conversation_filenames,
            load_conversation,
        ]
    )
    # Nova Sonic doesn't emit user-turn frames. To get them (for RTVI
    # speech events, turn observers, etc.) uncomment the local-VAD
    # imports + `user_params=` below. See realtime-aws-nova-sonic.py for
    # the full discussion.
    #
    # from pipecat.audio.vad.silero import SileroVADAnalyzer
    # from pipecat.processors.aggregators.llm_response_universal import (
    #     LLMUserAggregatorParams,
    # )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        # user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            user_aggregator,
            llm,  # LLM
            transport.output(),  # Transport bot output
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
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
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])
        # HACK: if using the older Nova Sonic (pre-2) model, you need this special way of
        # triggering the first assistant response. Note that this trigger requires a special
        # corresponding bit of text in the system instruction.
        # await llm.trigger_assistant_response()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
