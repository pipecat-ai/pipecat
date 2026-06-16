#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Two LLM workers with a main worker bridging transport to the bus.

Demonstrates multi-worker coordination: a main worker handles transport I/O
(STT, TTS) and bridges frames to the bus. Two LLM workers — a greeter and
a support worker — each run their own LLM pipeline and hand off control
between each other.

The user talks to one worker at a time. Hand-offs are seamless — the LLM
decides when to transfer based on its tools.

Requirements:

- OPENAI_API_KEY
- DEEPGRAM_API_KEY
- CARTESIA_API_KEY
- DAILY_API_KEY (for Daily transport)
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus import BusBridgeProcessor
from pipecat.evals.transport import EvalTransportParams
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
from pipecat.workers.llm import LLMWorker, LLMWorkerActivationArgs, tool
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

MAIN_NAME = "acme"


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


class AcmeLLMTask(LLMWorker):
    """LLM-only child worker with transfer/end tools.

    Receives user context from the main worker via the bus, runs its LLM,
    and ships generated text frames back. The main worker's TTS turns the
    text into audio.

    Passing ``bridged=()`` tells :class:`PipelineWorker` to wrap the LLM
    pipeline with bus edge processors so frames flow between this worker
    and the main worker automatically.
    """

    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(self, params: FunctionCallParams, agent: str, reason: str):
        """Transfer the user to another agent.

        Args:
            agent (str): The agent to transfer to (e.g. 'greeter', 'support').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Task '{self.name}': transferring to '{agent}' ({reason})")
        await self.activate_worker(
            agent,
            args=LLMWorkerActivationArgs(
                messages=[{"role": "developer", "content": reason}],
            ),
            deactivate_self=True,
            result_callback=params.result_callback,
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"Task '{self.name}': ending conversation ({reason})")
        await self.end(
            reason=reason,
            messages=[{"role": "developer", "content": reason}],
            result_callback=params.result_callback,
        )


def build_greeter() -> AcmeLLMTask:
    """Greeter: routes the user to support when they pick a product."""
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a friendly greeter for Acme Corp. The available products "
                "are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme "
                "Tornado Kit. Ask which one they'd like to learn more about. "
                "When the user picks a product or asks a question about one, "
                "immediately call the transfer_to_agent tool with agent 'support'. "
                "Do not answer product questions yourself. If the user says goodbye, "
                "call the end_conversation tool. Do not mention transferring — just do it "
                "seamlessly. Keep responses brief — this is a voice conversation."
            ),
        ),
    )
    return AcmeLLMTask("greeter", llm=llm, bridged=())


def build_support() -> AcmeLLMTask:
    """Support: answers product questions, can hand back to the greeter."""
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a support agent for Acme Corp. You know about three "
                "products: Acme Rocket Boots (jet-powered boots, $299, run up "
                "to 60 mph), Acme Invisible Paint (makes anything invisible for "
                "24 hours, $49 per can), and Acme Tornado Kit (portable tornado "
                "generator, $199, batteries included). Answer the user's questions "
                "about these products. If the user wants to browse other products "
                "or start over, call the transfer_to_agent tool with agent "
                "'greeter'. If the user says goodbye, call the end_conversation "
                "tool. Do not mention transferring — just do it seamlessly. "
                "Keep responses brief — this is a voice conversation."
            ),
        ),
    )
    return AcmeLLMTask("support", llm=llm, bridged=())


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting two-agent bot")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # The main bridge sends user-side context downstream to the children
    # via the bus, and the children's generated text comes back here so
    # the TTS can speak it.
    bridge = BusBridgeProcessor(
        bus=runner.bus,
        worker_name=MAIN_NAME,
        name=f"{MAIN_NAME}::BusBridge",
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            bridge,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await worker.activate_worker(
            "greeter",
            args=LLMWorkerActivationArgs(
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "Welcome the user to Acme Corp, mention the available products "
                            "and ask how you can help."
                        ),
                    },
                ],
            ),
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(build_greeter(), build_support(), worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
