#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""End a call with a scripted farewell and no extra LLM run.

A common telephony pattern: the user says goodbye, the LLM calls `end_call`
with a farewell message, and the bot speaks that farewell and hangs up. Done
naively, the farewell plays twice:

1. If `end_call` returns only a result (no next node), Flows runs the LLM
   again on the result, which generates its own goodbye.
2. Even with a proper transition, a node runs an LLM completion as soon as
   it's set (`respond_immediately=True` by default), so a `tts_say`
   pre-action plays one farewell and the completion generates another.

This example shows the zero-extra-LLM version:

- `end_call` returns `(None, create_end_node(...))`, so Flows transitions
  instead of re-running the LLM.
- The end node sets `respond_immediately=False`, so entering the node does
  not trigger a completion.
- The farewell is spoken by an `end_conversation` action with a `text`
  field, which needs no LLM: it speaks the text, then gracefully ends the
  pipeline. The LLM already wrote the farewell as the `end_call` function
  argument during its one and only completion, so the goodbye is still
  dynamic.
- The action goes in `pre_actions`. With `respond_immediately=False`,
  post-actions are deferred until after the node's first LLM response,
  which never comes on an end node, so a post-action would never run and
  the call would never end.

Requirements:
- CARTESIA_API_KEY
- GOOGLE_API_KEY

Run the example:
uv run end_call_scripted_farewell.py
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import FlowManager, NodeConfig
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    # Behavioral evals: run with `-t eval` to drive this bot via `pipecat eval`.
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# Flow nodes
def create_initial_node() -> NodeConfig:
    """Create the initial node of the flow.

    A simple phone-agent persona whose only special ability is ending the
    call when the user is done.
    """
    return NodeConfig(
        name="initial",
        role_message="You are a friendly phone agent for Cheerful Payments. Keep your responses short and conversational. You must use the end_call function when the user wants to end the call. Your responses will be converted to audio. Avoid outputting special characters and emojis.",
        task_messages=[
            {
                "role": "developer",
                "content": "Greet the caller and ask how you can help them today.",
            }
        ],
        functions=[end_call],
    )


async def end_call(flow_manager: FlowManager, farewell_message: str) -> tuple[None, NodeConfig]:
    """End the call once the user is done.

    Args:
        farewell_message: A short, friendly goodbye to speak to the user
            before hanging up, e.g. "Thanks for calling. Goodbye!"
    """
    # Returning a next node (instead of just a result) tells Flows to
    # transition rather than run the LLM again on the function result.
    return None, create_end_node(farewell_message)


def create_end_node(farewell_message: str) -> NodeConfig:
    """End the conversation by speaking the scripted farewell.

    No LLM completion runs in this node: `respond_immediately=False` skips
    the completion on node entry, and the `end_conversation` pre-action
    speaks `farewell_message` directly, then gracefully ends the pipeline.
    """
    return NodeConfig(
        name="end",
        # Required field, but it never reaches the LLM: respond_immediately
        # is False and the conversation ends before any user turn.
        task_messages=[{"role": "developer", "content": "Say goodbye."}],
        respond_immediately=False,
        # Must be a pre-action: with respond_immediately=False, post-actions
        # are deferred until after the node's first LLM response, which never
        # comes here, so a post-action would never run.
        pre_actions=[{"type": "end_conversation", "text": farewell_message}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="32b3f3c5-7171-46aa-abe7-b598964aa793",
        ),
    )
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY", ""))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=True,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
