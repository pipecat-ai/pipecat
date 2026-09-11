#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A two-layer voice agent: a cascade frontend delegating to a backend LLM.

The frontend keeps the conversation moving with a small, fast model and no
tools of its own. Anything that needs tools or careful reasoning it hands to
a backend running Claude, and relays the answer. ``TwoLayerLLMService`` wires
the two together: it installs the ``delegate`` tool on the frontend and runs
the backend as a worker of its own.

With a text frontend the defaults hand the backend the conversation itself
(the frontend words nothing) and relay the backend's progress as it comes.
``realtime-frontend.py`` puts a speech-to-speech model in the frontend's
place, against the same backend.

Architecture::

    Main worker (transport + STT + TwoLayerLLMService + TTS)
      ├── frontend: fast LLM, ``delegate`` tool
      └── backend: BackendLLMWorker (Claude + tools), delegated to over a job

Requirements:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.two_layer_llm_service import BackendConnector, TwoLayerLLMService
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import (
    RTVIFunctionCallReportLevel,
    RTVIObserverParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.llm import BackendLLMWorker
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

FRONTEND_INSTRUCTIONS = """You are a friendly, concise voice assistant. Your responses are spoken
aloud, so keep them to one or two natural sentences without any formatting."""

BACKEND_INSTRUCTIONS = """You are the backend of a voice assistant. Use the available tools to
answer questions about the weather and restaurants."""

BACKEND_DESCRIPTION = "current information such as the weather or a restaurant recommendation"

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

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )

    llm = TwoLayerLLMService(
        frontend=OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(
                model="gpt-5.4-mini", system_instruction=FRONTEND_INSTRUCTIONS
            ),
        ),
        # Thinking summaries stream back to the frontend as "thought" outputs.
        backend=BackendLLMWorker(
            name="backend",
            llm=AnthropicLLMService(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                settings=AnthropicLLMService.Settings(
                    system_instruction=BACKEND_INSTRUCTIONS,
                    thinking=AnthropicLLMService.ThinkingConfig(
                        type="adaptive", display="summarized"
                    ),
                ),
            ),
            context=LLMContext(tools=[get_current_weather, get_restaurant_recommendation]),
        ),
        connector=BackendConnector(backend_description=BACKEND_DESCRIPTION),
    )

    # The frontend's only tool, ``delegate``, is installed by the service;
    # the real tools live in the backend.
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
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

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Greet the user and ask how you can help."}
        )
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
