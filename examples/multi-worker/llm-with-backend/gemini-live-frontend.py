#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A voice agent with an LLM and a backend: a speech-to-speech frontend delegating to a backend LLM.

The frontend is Gemini Live, holding the spoken conversation with no
tools of its own. Anything that needs tools or careful reasoning it hands to
a backend running Claude, and relays what comes back. ``LLMWithBackend`` wires
the two together: it installs the ``delegate`` tool on the frontend and runs
the backend as a worker of its own.

With a speech-to-speech frontend the defaults have the model word the
request itself, since its context can lag the audio and the backend cannot
read the conversation. The backend's progress is relayed as it comes, as it
is for a text frontend: a Gemini Live model takes it on the tool-response
channel itself, which ``gemini-3.8-live`` and the 2.5 native-audio models
allow. ``openai-realtime-frontend.py`` puts OpenAI Realtime in the frontend's
place and ``openai-responses-frontend.py`` a cascade pipeline, against the same backend
and the same prompts.

Architecture::

    Main worker (transport + LLMWithBackend)
      ├── frontend: realtime model, ``delegate`` tool
      └── backend: BackendLLMWorker (Claude + tools), delegated to over a job

Requirements:

- GOOGLE_API_KEY
- ANTHROPIC_API_KEY
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.llm_with_backend import LLMWithBackend
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import (
    RTVIFunctionCallReportLevel,
    RTVIObserverParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.llm import BackendLLMWorker
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

FRONTEND_INSTRUCTIONS = """You are a friendly, concise voice assistant. Your responses are spoken
aloud, so keep them to one or two natural sentences without any formatting.
The backend answers questions about the weather and restaurants."""

BACKEND_INSTRUCTIONS = """Use the available tools to answer questions about the weather and
restaurants."""


transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def get_current_weather(params: FunctionCallParams, location: str, format: str):
    """Get the current weather.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
        format: The temperature unit to use. Must be either "celsius" or "fahrenheit". Infer this from the user's location.
    """
    # Uncomment to exercise longer-running backend work.
    # import asyncio
    # await asyncio.sleep(6)
    temperature = 75 if format == "fahrenheit" else 24
    await params.result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": format,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


async def get_restaurant_recommendation(params: FunctionCallParams, location: str):
    """Get a restaurant recommendation.

    Args:
        location: The city and state, e.g. "San Francisco, CA".
    """
    await params.result_callback({"name": "The Golden Dragon"})


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    # Thinking summaries stream back to the frontend as "thought" outputs.
    backend = BackendLLMWorker(
        name="backend",
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                thinking=AnthropicLLMService.ThinkingConfig(type="adaptive", display="summarized"),
            ),
        ),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
    )

    # A backend that takes a while can say so the moment work is handed to
    # it: the frontend says the line while the backend works, instead of
    # waiting in silence. Even a quick lookup here is a few model round trips,
    # so the line is worth it; drop it for a backend that answers at once.
    @backend.event_handler("on_delegation_started")
    async def on_delegation_started(backend, request):
        await backend.say("Let me look into that, this takes a moment.")

    llm = LLMWithBackend(
        frontend=GeminiLiveLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GeminiLiveLLMService.Settings(
                model="models/gemini-3.8-live",
                system_instruction=FRONTEND_INSTRUCTIONS,
            ),
        ),
        backend=backend,
    )

    # The frontend's only tool, ``delegate``, is installed by the service;
    # the real tools live in the backend.
    context = LLMContext(
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name="frontend",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # The handoff is hidden, so the backend's calls show at top level, as
        # they do for OpenAI Live's client delegation. Remove the "delegate"
        # entry to see the handoff itself as a call, with the backend's calls
        # nested under it.
        rtvi_observer_params=RTVIObserverParams(
            function_call_report_level={"delegate": RTVIFunctionCallReportLevel.DISABLED},
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
