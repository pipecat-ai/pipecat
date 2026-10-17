#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A cascade frontend whose connector records none of the backend's progress.

The example bots record the backend's tool calls and thoughts as silent notes
in the frontend's context, which is what the frontend answers from when the
user asks how the work is going. This bot's connector keeps only the
backend's spoken messages, so a question about the work finds nothing to
answer it from.

The frontend service comes from the runner body, as the manifest gives it:
``{"frontend": "openai"}``, ``"google"`` or ``"anthropic"``, the same services
and instructions as the example of that name. The backend, its tools and the
frontend instruction are the examples' own, from ``backend.py``.
"""

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "examples/multi-worker/llm-with-backend")
)

from backend import FRONTEND_INSTRUCTIONS, build_backend  # noqa: E402

from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.evals.transport import EvalTransportParams  # noqa: E402
from pipecat.frames.frames import LLMRunFrame  # noqa: E402
from pipecat.pipeline.llm_with_backend import (  # noqa: E402
    BackendConnector,
    BackendOutput,
    BackendToolCall,
    LLMWithBackend,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.worker import (  # noqa: E402
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments  # noqa: E402
from pipecat.runner.utils import create_transport  # noqa: E402
from pipecat.services.cartesia.tts import CartesiaTTSService  # noqa: E402
from pipecat.services.deepgram.stt import DeepgramSTTService  # noqa: E402
from pipecat.services.llm_service import LLMService  # noqa: E402
from pipecat.transports.base_transport import BaseTransport  # noqa: E402
from pipecat.workers.runner import WorkerRunner  # noqa: E402

load_dotenv(override=True)

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


class NoProgressConnector(BackendConnector):
    """A connector that records only the backend's spoken messages."""

    def render_output(self, output: BackendOutput) -> LLMContextMessage | None:
        return super().render_output(output) if output.prefers_spoken else None

    def render_tool_call(self, call: BackendToolCall) -> LLMContextMessage | None:
        return None


def build_frontend(name: str) -> LLMService[Any]:
    """The frontend service of the example of that name."""
    if name == "openai":
        from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService

        return OpenAIResponsesLLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAIResponsesLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    if name == "google":
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    if name == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(system_instruction=FRONTEND_INSTRUCTIONS),
        )
    raise ValueError(f"unknown frontend {name!r}: expected openai, google or anthropic")


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    frontend = (runner_args.body or {}).get("frontend", "openai")
    logger.info(f"Starting bot with the {frontend} frontend")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
    )
    llm = LLMWithBackend(
        frontend=build_frontend(frontend),
        backend=build_backend(),
        connector=NoProgressConnector(),
    )

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
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.add_message(
            {"role": "developer", "content": "Greet the user and ask how you can help."}
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Bot entry point."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
