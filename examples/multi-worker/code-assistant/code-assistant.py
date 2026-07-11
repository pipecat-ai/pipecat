#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice code assistant powered by Claude Agent SDK.

Talk to your codebase hands-free. Ask questions like "what does the
auth middleware do?" or "find all TODO comments" and get spoken answers
based on actual file contents. The Claude Agent SDK worker navigates
the filesystem using Read, Bash, Glob, and Grep tools.

Architecture::

    Main worker (transport + LLM + ``ask_code`` tool)
      └── job → CodeWorker (Claude Agent SDK)

Requirements:

- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import os

from code_worker import CodeWorker
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

PROJECT_PATH = os.getenv("PROJECT_PATH", os.getcwd())

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


@tool_options(cancel_on_interruption=False)
async def ask_code(params: FunctionCallParams, question: str):
    """Ask a question about the codebase. A Claude Code worker will
    explore the project by reading files, searching code, and running
    commands. It remembers previous questions for follow-ups.

    Args:
        question (str): The question about code, files, structure,
            dependencies, or anything in the project.
    """
    logger.info(f"Asking code worker: '{question}'")
    async with params.pipeline_worker.job("code-worker", payload={"question": question}) as job:
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": "Give me a moment."}],
                run_llm=True,
            )
        )
        # The LLM keeps talking while the worker runs.
    await params.result_callback(job.response)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting code assistant")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a voice interface to a code assistant powered by Claude Code. "
                "Behind you is a worker that can read files, search code with grep and "
                "glob patterns, and run bash commands on the project. It maintains "
                "context across questions, so follow-up questions work naturally.\n\n"
                "When the user asks anything about code, project structure, files, "
                "dependencies, tests, or wants to explore the codebase, call the "
                "ask_code tool. When the worker result comes back, summarize it naturally "
                "for speaking. Keep responses concise and conversational.\n"
            ),
        ),
    )

    context = LLMContext(tools=[ask_code])
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
        name="code-assistant",
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": "Greet the user and tell them you're a code assistant.",
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(CodeWorker("code-worker", project_path=PROJECT_PATH), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
