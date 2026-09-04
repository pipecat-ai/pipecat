#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A two-tier voice agent: a speech-to-speech frontend delegating to a backend LLM.

The frontend is OpenAI Realtime, holding the spoken conversation with no tools
of its own. Anything that needs tools or careful reasoning it hands to a
``BackendLLMWorker`` running Claude, through the ``delegate`` tool, and relays
its answer.

The split does not depend on what the frontend is: ``backend-llm-cascade-frontend.py``
puts a cascade pipeline in this role, against the same backend and the same job
contract, and ``OpenAILiveLLMService`` builds client delegation on it too.

Architecture::

    Main worker (transport + realtime model, ``delegate`` tool)
      └── job → BackendLLMWorker (Claude + tools)

Requirements:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.llm import BackendLLMWorker, delegate_to_backend
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

BACKEND_NAME = "backend"

FRONTEND_INSTRUCTIONS = """You are a friendly, concise voice assistant. Your responses are spoken
aloud, so keep them to one or two natural sentences without any formatting.

Answer simple conversational questions yourself. Whenever the user asks for
current information, such as the weather or a restaurant recommendation, or
asks you to look something up, call the delegate tool with a self-contained
request: the user's goal, the exact details they gave (places, dates, names)
and their latest correction. While it runs, keep the conversation going;
when the result comes back, relay it in your own words."""

BACKEND_INSTRUCTIONS = """You are the backend of a voice assistant. Each message you receive
is a request the assistant has handed you from a live voice conversation. It
may contain transcription errors; use the most likely intent.

Use the available tools to answer questions about the weather and
restaurants. Reply with the verified result in concise, conversational plain
text that the assistant can say to the user — no Markdown, no raw JSON — and
never claim an action completed without a tool result confirming it."""

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

    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            system_instruction=FRONTEND_INSTRUCTIONS,
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription(),
                        turn_detection=SemanticTurnDetection(),
                    )
                ),
            ),
        ),
    )

    # When using a speech-to-speech model for the frontend, we can't rely on the
    # context to be up-to-date at delegation time (see realtime_service_mode
    # for background on that). So instead of asking the backend to extract the
    # user's intent from the conversation, like we would with a cascade
    # frontend, we have the frontend pass a specific request to the backend.
    @tool_options(cancel_on_interruption=False)
    async def delegate(params: FunctionCallParams, task: str):
        """Hand work to the backend, for anything needing tools, current information or careful reasoning.

        Args:
            task: What the backend should do, self-contained: the user's goal,
                the details they gave and their latest correction.
        """
        logger.info(f"Delegating to the backend: {task!r}")

        # No on_update here: realtime models don't accept intermediate tool
        # results — they take one result, when the call completes. So the
        # backend's progress, which a cascade frontend can use, is ignored.
        text = await delegate_to_backend(
            params.pipeline_worker,
            BACKEND_NAME,
            request=task,
            timeout_secs=120,
        )
        logger.info(f"Backend result: {text!r}")
        await params.result_callback(text)

    # The frontend's only tool is the handoff; the real tools live in the backend.
    context = LLMContext(
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
        tools=ToolsSchema(standard_tools=[delegate]),
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
        # Whether a client sees the handoff as a tool call is the app's call:
        # it is a real tool call, though it reports the handoff rather than the
        # work behind it. This example hides it. (The backend's own calls run in
        # its worker's pipeline, which this observer doesn't watch.)
        rtvi_observer_params=RTVIObserverParams(
            function_call_report_level={"delegate": RTVIFunctionCallReportLevel.DISABLED},
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    backend = BackendLLMWorker(
        name=BACKEND_NAME,
        llm=AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                system_instruction=BACKEND_INSTRUCTIONS,
                thinking=AnthropicLLMService.ThinkingConfig(type="adaptive", display="summarized"),
            ),
        ),
        context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker, backend)

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
