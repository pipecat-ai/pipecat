#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An engineering assistant with a backend: a cascade frontend delegating to a backend LLM.

The frontend keeps the conversation moving with a fast model on OpenAI's
Responses API and no tools of its own. Anything that needs tools or careful
reasoning it hands to a backend running Claude, and relays what comes back
as it comes. ``LLMWithBackend`` wires the two together: it installs the
``delegate`` and ``cancel_delegated_work`` tools on the frontend and runs the
backend as a worker of its own.

The backend, its tools and both prompts are in ``backend.py``, shared with
``openai-realtime-frontend.py`` and ``gemini-live-frontend.py``, which put a
speech-to-speech model in the frontend's place. Try: "fix the flaky retry test
in the HTTP client", then ask for something else while it works.

Architecture::

    Main worker (transport + STT + LLMWithBackend + TTS)
      ├── frontend: fast LLM, ``delegate`` and ``cancel_delegated_work`` tools
      └── backend: BackendLLMWorker (Claude + tools), attached for the session

Requirements:

- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
"""

import os

from backend import FRONTEND_INSTRUCTIONS, build_backend
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.llm_with_backend import LLMWithBackend
from pipecat.pipeline.pipeline import Pipeline
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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )

    llm = LLMWithBackend(
        frontend=OpenAIResponsesLLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAIResponsesLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        ),
        backend=build_backend(),
    )

    # The frontend's tools are installed by the service; the real tools live
    # in the backend.
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
        # The handoff is hidden, so the backend's calls show at top level, as
        # they do for OpenAI Live's client delegation. Remove the "delegate"
        # entry to see the handoff itself as a call.
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
