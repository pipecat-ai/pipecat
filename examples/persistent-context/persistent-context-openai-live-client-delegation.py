#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live (gpt-live-1) with client delegation and a persisted context.

The conversation recorded in the live model's ``LLMContext`` can be saved to a
file and loaded back with tools the Anthropic backend calls. Loading restarts
the Live session with the restored history as its prior conversation (the
Live API only takes history at session start) and seeds the backend's own
context with the same turns.
"""

import asyncio
import glob
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.llm import BackendLLMWorker
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

BASE_FILENAME = "/tmp/pipecat_conversation_"

FRONTEND_INSTRUCTIONS = """## Role and speaking style
You are a friendly, concise voice assistant. Speak naturally, in one or two
sentences at a time, and let the user finish before responding.

## Delegation
Answer simple conversational questions directly. Delegate when the user asks
for current information such as the weather, or asks you to save the
conversation, list saved conversations, or load one. When delegating, include
the user's goal and the exact details they gave, so the request is
self-contained. Relay the result once it arrives.

## Interruptions
Stop speaking when the user interrupts and listen to the new request."""

BACKEND_INSTRUCTIONS = """You are the backend of a voice assistant. Each message you receive
contains the recent voice conversation between the user and the assistant,
as a transcript, followed by a task the assistant delegated to you. The
transcript may contain transcription errors; use the most likely intent.

Use the available tools to answer questions about the weather and to save,
list and load conversations. Reply with the verified result in concise,
conversational plain text that the assistant can say to the user — no
Markdown, no raw JSON."""


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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    # The live model's context: the conversation as recorded from the
    # session's transcripts. This is what gets saved and restored.
    context = LLMContext(
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
    )

    backend_context = LLMContext(
        [{"role": "system", "content": BACKEND_INSTRUCTIONS}],
    )

    # The persistence tools run in the backend but act on the live model's
    # context, so they close over it.

    async def get_saved_conversation_filenames(params: FunctionCallParams):
        """Get a list of saved conversation histories. Returns a list of filenames. Each filename includes a date and timestamp. Each file is conversation history that can be loaded into this session."""
        matching_files = glob.glob(f"{BASE_FILENAME}*.json")
        logger.debug(f"matching files: {matching_files}")
        await params.result_callback({"filenames": matching_files})

    async def save_conversation(params: FunctionCallParams):
        """Save the current conversation. Use this function to persist the current conversation to external storage."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        filename = f"{BASE_FILENAME}{timestamp}.json"
        messages = context.get_messages()
        logger.debug(f"writing conversation to {filename}\n{json.dumps(messages, indent=4)}")
        try:
            with open(filename, "w") as file:
                json.dump(messages, file, indent=2)
            await params.result_callback({"success": True, "filename": filename})
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})

    async def load_conversation(params: FunctionCallParams, filename: str):
        """Load a conversation history. Use this function to load a conversation history into the current session.

        Args:
            filename: The filename of the conversation history to load.
        """
        try:
            with open(filename) as file:
                messages = json.load(file)
        except Exception as e:
            await params.result_callback({"success": False, "error": str(e)})
            return

        context.set_messages(messages)
        context.add_message(
            {
                "role": "developer",
                "content": "The saved conversation above has just been restored. Briefly "
                "tell the user it's loaded and that you're ready to continue.",
            }
        )
        # Give the backend the restored turns too, as its own prior history.
        backend_context.set_messages(
            [{"role": "system", "content": BACKEND_INSTRUCTIONS}]
            + [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
        )
        await params.result_callback({"success": True})

        async def _reset():
            # Restart the Live session from the restored context once this
            # delegation has been answered.
            await asyncio.sleep(1)
            await llm.reset_conversation()

        asyncio.create_task(_reset())

    backend = BackendLLMWorker(
        llm=AnthropicLLMService(api_key=os.environ["ANTHROPIC_API_KEY"]),
        context=backend_context,
    )
    backend_context.set_tools(
        [
            get_current_weather,
            save_conversation,
            get_saved_conversation_filenames,
            load_conversation,
        ]
    )

    llm = OpenAILiveLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILiveLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        delegation=OpenAILiveLLMService.ClientDelegation(backend=backend),
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

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
        observers=[TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Start the Live session from the context.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


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


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
