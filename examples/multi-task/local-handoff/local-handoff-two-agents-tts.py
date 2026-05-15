#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Two LLM tasks with per-task TTS voices.

Same shape as ``local-handoff-two-agents.py``, but each child task
runs its own TTS with a distinct voice. The main task has no TTS —
audio comes from the child tasks via the bus and is played by the
main task's transport. Tasks announce the transfer ("let me connect
you with...") before handing off.

Architecture::

    Main task (no TTS):
      transport.in → STT → user_agg → BusBridge → transport.out → assistant_agg

    Child task (with TTS):
      bridge_in → LLM → TTS → bridge_out

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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
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
from pipecat.tasks.llm import LLMTask, LLMTaskActivationArgs, tool
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

load_dotenv(override=True)

MAIN_NAME = "acme"


transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


class AcmeTTSTask(LLMTask):
    """Child task with its own LLM + TTS, bridged to the main task.

    Each child wraps the standard ``Pipeline([llm])`` with an extra
    TTS processor so audio is produced locally by each child and
    shipped to the main task over the bus.
    """

    def __init__(self, name: str, *, llm: OpenAILLMService, voice_id: str):
        """Initialize the child task.

        Args:
            name: Unique task name.
            llm: The LLM service for this child.
            voice_id: Cartesia voice id for this child's TTS.
        """
        tts = CartesiaTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            settings=CartesiaTTSService.Settings(voice=voice_id),
        )
        super().__init__(
            name,
            llm=llm,
            pipeline=Pipeline([llm, tts]),
            bridged=(),
        )

    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(self, params: FunctionCallParams, agent: str, reason: str):
        """Transfer the user to another agent.

        Args:
            agent (str): The agent to transfer to (e.g. 'greeter', 'support').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Task '{self.name}': transferring to '{agent}' ({reason})")
        await self.handoff_to(
            agent,
            messages=[
                {
                    "role": "developer",
                    "content": f"Tell the user about the transfer ({reason}).",
                }
            ],
            activation_args=LLMTaskActivationArgs(
                messages=[{"role": "developer", "content": reason}],
            ),
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


def build_greeter() -> AcmeTTSTask:
    """Greeter: routes the user to support when they pick a product."""
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a friendly greeter for Acme Corp. The available products "
                "are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme "
                "Tornado Kit. Ask which one they'd like to learn more about. "
                "When the user picks a product or asks a question about one, "
                "call the transfer_to_agent tool with agent 'support'. "
                "Do not answer product questions yourself. If the user says goodbye, "
                "call the end_conversation tool. Keep responses brief — this is a "
                "voice conversation."
            ),
        ),
    )
    return AcmeTTSTask(
        "greeter",
        llm=llm,
        voice_id="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
    )


def build_support() -> AcmeTTSTask:
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
                "tool. Keep responses brief — this is a voice conversation."
            ),
        ),
    )
    return AcmeTTSTask(
        "support",
        llm=llm,
        voice_id="a167e0f3-df7e-4d52-a9c3-f949145efdab",  # Blake
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting two-agents-with-tts bot")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # The main task has no TTS. Audio comes from the children over
    # the bus; the main bridge tees user context out and pushes
    # incoming audio/text frames back into the local pipeline.
    bridge = BusBridgeProcessor(
        bus=runner.bus,
        task_name=MAIN_NAME,
        name=f"{MAIN_NAME}::BusBridge",
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            bridge,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
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
        await task.activate_task(
            "greeter",
            args=LLMTaskActivationArgs(
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

    await runner.spawn(build_greeter())
    await runner.spawn(build_support())
    await runner.spawn(task)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
