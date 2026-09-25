#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An engineering assistant with a backend: a speech-to-speech frontend delegating to a backend LLM.

The frontend is Gemini Live, holding the spoken conversation with no tools
of its own. Anything that needs tools or careful reasoning it hands to a
backend running Claude, and relays what comes back as it comes.
``LLMWithBackend`` wires the two together: it installs the ``delegate`` and
``cancel_delegated_work`` tools on the frontend and runs the backend as a
worker of its own. With a speech-to-speech frontend the model words the
request itself, since its context can lag the audio.

The backend, its tools and both prompts are in ``backend.py``, shared with
``openai-responses-frontend.py``, which puts a cascade pipeline in the
frontend's place. Try: "fix the flaky retry test in the HTTP client", then
ask for something else while it works.

Architecture::

    Main worker (transport + LLMWithBackend)
      ├── frontend: speech-to-speech model, ``delegate`` and ``cancel_delegated_work`` tools
      └── backend: BackendLLMWorker (Claude + tools), attached for the session

Requirements:

- GOOGLE_API_KEY
- ANTHROPIC_API_KEY
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
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
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

    llm = LLMWithBackend(
        frontend=GeminiLiveLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GeminiLiveLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        ),
        backend=build_backend(),
    )

    # The frontend's tools are installed by the service; the real tools live
    # in the backend.
    context = LLMContext(
        [{"role": "developer", "content": "Greet the user and ask how you can help."}],
    )
    # Gemini Live drives the conversation server-side and emits no turn frames
    # (UserStartedSpeakingFrame, UserStoppedSpeakingFrame). The local VAD adds
    # supplemental turn frames for processors that expect them, such as RTVI,
    # whose clients start a new user transcript entry on them. They are
    # approximate: they may not always align with the server's turns.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

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
        # The backend's calls are reported to the client with their names,
        # arguments and results, so a UI can show what the backend is doing.
        # The handoff itself is hidden, so those calls show at top level, as
        # they do for OpenAI Live's client delegation; remove the "delegate"
        # entry to see it as a call.
        rtvi_observer_params=RTVIObserverParams(
            function_call_report_level={
                "*": RTVIFunctionCallReportLevel.FULL,
                "delegate": RTVIFunctionCallReportLevel.DISABLED,
            },
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
