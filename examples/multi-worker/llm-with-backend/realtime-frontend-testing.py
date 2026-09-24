#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A voice agent with an LLM and a backend: a speech-to-speech frontend delegating to a backend LLM.

The frontend is OpenAI Realtime, holding the spoken conversation with no
tools of its own. Anything that needs tools or careful reasoning it hands to
a backend running Claude, and relays what comes back. ``LLMWithBackend`` wires
the two together: it installs the ``delegate`` tool on the frontend and runs
the backend as a worker of its own.

With a speech-to-speech frontend the defaults have the model word the
request itself, since its context can lag the audio and the backend cannot
read the conversation. The backend's progress is relayed as it comes, as it
is for a text frontend. ``cascade-frontend.py`` puts a cascade pipeline in
the frontend's place, against the same backend and the same prompts.

Architecture::

    Main worker (transport + LLMWithBackend)
      ├── frontend: realtime model, ``delegate`` tool
      └── backend: BackendLLMWorker (Claude + tools), delegated to over a job

Requirements:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
"""

import os
from datetime import datetime
from typing import Any

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
from pipecat.services.llm_service import FunctionCallParams, LLMService
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
    import asyncio

    await asyncio.sleep(6)
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


# TEMP (manual testing): pick the frontend with --frontend.
FRONTENDS = ("openai", "azure", "gemini", "grok", "nova", "inworld", "ultravox")
_FRONTEND = "openai"
_MODEL: str | None = None  # --model, for the gemini frontend


def _build_frontend(name: str) -> LLMService[Any]:
    """Build the speech-to-speech frontend service named on the command line."""
    if name in ("openai", "azure"):
        from pipecat.services.openai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            InputAudioTranscription,
            SemanticTurnDetection,
            SessionProperties,
        )

        session_properties = SessionProperties(
            audio=AudioConfiguration(
                input=AudioInput(
                    transcription=InputAudioTranscription(),
                    turn_detection=SemanticTurnDetection(),
                )
            ),
        )
        if name == "openai":
            from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

            return OpenAIRealtimeLLMService(
                api_key=os.environ["OPENAI_API_KEY"],
                settings=OpenAIRealtimeLLMService.Settings(
                    system_instruction=FRONTEND_INSTRUCTIONS,
                    session_properties=session_properties,
                ),
            )
        from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService

        return AzureRealtimeLLMService(
            api_key=os.environ["AZURE_REALTIME_API_KEY"],
            base_url=os.environ["AZURE_REALTIME_BASE_URL"],
            settings=AzureRealtimeLLMService.Settings(
                system_instruction=FRONTEND_INSTRUCTIONS,
                session_properties=session_properties,
            ),
        )
    if name == "gemini":
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

        settings = GeminiLiveLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS)
        if _MODEL:
            settings.model = _MODEL
        return GeminiLiveLLMService(api_key=os.environ["GOOGLE_API_KEY"], settings=settings)
    if name == "grok":
        from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService

        return GrokRealtimeLLMService(
            api_key=os.environ["XAI_API_KEY"],
            settings=GrokRealtimeLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    if name == "nova":
        from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

        return AWSNovaSonicLLMService(
            secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            region=os.environ.get("AWS_REGION", "us-east-1"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
            settings=AWSNovaSonicLLMService.Settings(
                voice="tiffany", system_instruction=FRONTEND_INSTRUCTIONS
            ),
        )
    if name == "inworld":
        from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService

        # Takes no intermediate results, so it gets every backend output at once.
        return InworldRealtimeLLMService(
            api_key=os.environ["INWORLD_API_KEY"],
            llm_model="google-ai-studio/gemini-3.1-flash-lite",
            voice="Sarah",
            settings=InworldRealtimeLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    if name == "ultravox":
        from pipecat.services.ultravox.llm import OneShotInputParams, UltravoxRealtimeLLMService

        logger.warning(
            "Ultravox as a frontend: built-in tools don't reach its session yet, so its model "
            "never sees `delegate` and will answer without the backend."
        )
        return UltravoxRealtimeLLMService(
            params=OneShotInputParams(
                api_key=os.environ["ULTRAVOX_API_KEY"], system_prompt=FRONTEND_INSTRUCTIONS
            ),
        )
    raise ValueError(f"unknown frontend {name!r}; pick one of {', '.join(FRONTENDS)}")


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
        # pass

    print(  # TEMP
        f"\n\033[1;35m━━ frontend: {_FRONTEND}{f' ({_MODEL})' if _MODEL else ''} ━━\033[0m\n",
        flush=True,
    )
    llm = LLMWithBackend(frontend=_build_frontend(_FRONTEND), backend=backend)

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
    import argparse

    from pipecat.runner.run import main

    # TEMP (manual testing): --frontend picks the speech-to-speech service. It
    # is read here rather than from runner_args.cli_args, which not every
    # transport path fills in.
    parser = argparse.ArgumentParser(description="LLMWithBackend, realtime frontend")
    parser.add_argument("--frontend", choices=FRONTENDS, default="openai")
    parser.add_argument("--model", help="model for the gemini frontend, e.g. gemini-3.8-live")
    _known = parser.parse_known_args()[0]
    _FRONTEND, _MODEL = _known.frontend, _known.model
    main(parser)
