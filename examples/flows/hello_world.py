#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""A 'Hello-World' introduction to Pipecat Flows.

Requirements:
- CARTESIA_API_KEY
- GOOGLE_API_KEY

Run the example:
uv run hello_world.py
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

    Define the bot's role and task for the node, plus the function it can call.
    The function records the result and transitions to the next node.
    """
    return NodeConfig(
        name="initial",
        role_message="You are an inquisitive child. Use very simple language. Ask simple questions. You must ALWAYS use one of the available functions to progress the conversation. Your responses will be converted to audio. Avoid outputting special characters and emojis.",
        task_messages=[
            {
                "role": "developer",
                "content": "Say 'Hello world' and ask what is the user's favorite color.",
            }
        ],
        functions=[record_favorite_color],
    )


async def record_favorite_color(flow_manager: FlowManager, color: str) -> tuple[str, NodeConfig]:
    """Record the color the user said is their favorite.

    Here "record" means print to the console, but any logic could go here:
    write to a database, make an API call, etc.

    Args:
        color: The user's favorite color.
    """
    print(f"Your favorite color is: {color}")
    return color, create_end_node()


def create_end_node() -> NodeConfig:
    """End the conversation.

    Flows transitions to this node when the user has answered the question.
    It thanks the user and ends the conversation using the `end_conversation`
    post-action.
    """
    return NodeConfig(
        name="create_end_node",
        task_messages=[
            {
                "role": "developer",
                "content": "Thank the user for answering and end the conversation",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
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
