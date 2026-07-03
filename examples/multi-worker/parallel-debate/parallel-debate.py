#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Parallel debate using job groups.

A voice bot receives a topic from the user and fans out to three
workers in parallel via ``worker.job_group(...)``. Each worker
runs its own LLM context, so it remembers previous topics across
debate rounds. The bot collects all three perspectives and the
main-worker LLM synthesizes a balanced answer.

Architecture::

    Main worker (transport + LLM + ``debate`` tool)
      └── job_group(advocate, critic, analyst)
            └── DebateWorker (LLMContextWorker, one per role)

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus import BusJobRequestMessage
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
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
from pipecat.workers.llm import LLMContextWorker
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

ROLE_PROMPTS = {
    "advocate": (
        "You argue IN FAVOR of the topic. Present the strongest case for why "
        "this is a good idea, with concrete benefits. Be persuasive but honest. "
        "Be concise, just 2-3 sentences."
    ),
    "critic": (
        "You argue AGAINST the topic. Present the strongest concerns, risks, "
        "and downsides. Be critical but fair. Be concise, just 2-3 sentences."
    ),
    "analyst": (
        "You provide a BALANCED, NEUTRAL analysis. Weigh both sides objectively "
        "and highlight the key trade-offs. Be concise, just 2-3 sentences."
    ),
}

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


class DebateWorker(LLMContextWorker):
    """Worker that generates a perspective using its own LLM context.

    Each worker keeps its own ``LLMContext`` so it remembers previous
    topics across multiple debate rounds. Job requests append the new
    topic and trigger the LLM; the assistant-aggregator captures the
    full reply and sends it back as the job response.
    """

    def __init__(self, role: str):
        """Initialize the DebateWorker.

        Args:
            role: One of ``"advocate"``, ``"critic"``, ``"analyst"`` —
                used as the worker name and selects the system prompt.
        """
        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(system_instruction=ROLE_PROMPTS[role]),
        )
        super().__init__(role, llm=llm)
        self._role = role
        self._current_job_id: str | None = None

        @self.assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
            text = message.content
            logger.info(f"Worker '{self.name}': completed ({len(text)} chars)")
            if self._current_job_id:
                job_id = self._current_job_id
                self._current_job_id = None
                await self.send_job_response(job_id, {"role": self._role, "text": text})

    async def on_job_request(self, message: BusJobRequestMessage) -> None:
        """Inject the topic and run the LLM."""
        await super().on_job_request(message)
        self._current_job_id = message.job_id
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "developer", "content": f"Topic: {message.payload['topic']}"}],
                run_llm=True,
            )
        )


@tool_options(cancel_on_interruption=False, timeout_secs=60)
async def debate(params: FunctionCallParams, topic: str):
    """Analyze a topic from multiple perspectives (advocate, critic, analyst).

    Args:
        topic (str): The topic or question to debate.
    """
    logger.info(f"Starting debate on '{topic}'")
    async with params.pipeline_worker.job_group(
        *ROLE_PROMPTS, payload={"topic": topic}, timeout=30
    ) as tg:
        pass
    result = "\n\n".join(f"{r['role'].upper()}: {r['text']}" for r in tg.responses.values())
    logger.info("Debate complete, synthesizing")
    await params.result_callback(result)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting parallel-debate bot")

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
                "You are a debate moderator in a voice conversation. When the user "
                "gives you a topic, call the debate tool to gather perspectives from "
                "three viewpoints (advocate, critic, analyst). Then synthesize the "
                "results into a clear, balanced summary for the user. Keep your "
                "responses concise and natural for speaking."
            ),
        ),
    )

    context = LLMContext(tools=[debate])
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
        name="parallel-debate",
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
                "content": (
                    "Greet the user and tell them you can moderate a debate on any "
                    "topic. Ask what they'd like to explore."
                ),
            }
        )
        await worker.queue_frame(LLMRunFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(
        DebateWorker("advocate"),
        DebateWorker("critic"),
        DebateWorker("analyst"),
        worker,
    )

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
