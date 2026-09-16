#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The food ordering flow, configured from YAML and driven by TypeSafe.

The same conversation as python/food_ordering.py, split along the seam Pipecat
Flows offers for runtime configuration:

- flow.yaml holds the graph: the nodes, which tools each offers, and where
  each tool leads.
- handlers.py holds the tools: direct functions whose schema comes from
  their signature and docstring.
- This file holds every line the bot can say.

There is no text-generating LLM. TypeSafeFlowsLLMService stands in its place:
when the flow enters a node it speaks that node's written line, and at the end
of each caller turn it asks TypeSafe's Jev which of the node's tools the turn
calls for and what each tool argument is, then runs that tool so the flow
moves on exactly as it would with an LLM. A caller who gives the order one
detail at a time is asked for the missing detail; a turn that matches no tool
gets the node's reprompt line. Lines refer to session facts and to what
handlers have stored as {{ key }}, filled in from the manager's state.

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- TYPESAFE_API_KEY (for the judgments)
- DAILY_API_KEY (for the Daily transport)
"""

import os
from pathlib import Path

import handlers
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import Flow, FlowConfig, FlowManager
from pipecat.flows.typesafe_llm import NodeLines, ToolLines, TypeSafeFlowsLLMService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.typesafe import TypeSafeJudge
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

FLOW_CONFIG_PATH = Path(__file__).with_name("flow.yaml")

# What the bot says in each node of flow.yaml: on entry, and again when the
# caller's turn matched none of the node's tools. A reprompt never says the
# bot did not hear the caller; it repeats the question.
NODE_LINES = {
    "initial": NodeLines(
        say="Hi, welcome to {{ restaurant_name }}! Would you like pizza or sushi today?",
        reprompt="We have pizza and sushi tonight. Which would you like?",
    ),
    "choose_pizza": NodeLines(
        say=(
            "Great, pizza it is! What size would you like, and what kind? "
            "We have cheese, pepperoni, supreme, and vegetarian."
        ),
        reprompt="What size and what kind of pizza would you like?",
    ),
    "choose_sushi": NodeLines(
        say=(
            "Sushi, nice choice! How many rolls would you like, and which kind? "
            "We have California, spicy tuna, rainbow, and dragon."
        ),
        reprompt="How many rolls would you like, and which kind?",
    ),
    "confirm": NodeLines(
        say="So that's {{ order.summary }}, {{ order.total }} total. Does that sound right?",
        reprompt=(
            "Just to check: {{ order.summary }} for {{ order.total }}. "
            "Say yes to place the order, or tell me what to change."
        ),
    ),
    "restart": NodeLines(
        say="No problem, let's start over. Would you like pizza or sushi?",
        reprompt="We have pizza and sushi tonight. Which would you like?",
    ),
    "end": NodeLines(say="Thanks for your order! It'll be on its way soon. Goodbye!"),
}

# How each tool is judged, the values its arguments can take, and what the bot
# says around it. Tools not listed here (choose_pizza, choose_sushi,
# complete_order, revise_order) are judged by the descriptions in flow.yaml.
TOOL_LINES = {
    "select_pizza_order": ToolLines(
        description="The caller gives pizza order details: a size, a kind of pizza, or both",
        options={
            "size": ["small", "medium", "large"],
            "pizza_type": ["cheese", "pepperoni", "supreme", "vegetarian"],
        },
        ask={
            "size": "Sure. What size would you like: small, medium, or large?",
            "pizza_type": (
                "Got it. What kind of pizza would you like: "
                "cheese, pepperoni, supreme, or vegetarian?"
            ),
        },
    ),
    "select_sushi_order": ToolLines(
        description="The caller gives sushi order details: how many rolls, which roll, or both",
        options={
            "count": list(range(1, 11)),
            "roll_type": ["california", "spicy tuna", "rainbow", "dragon"],
        },
        ask={
            "count": "Sure. How many rolls would you like?",
            "roll_type": (
                "Got it. Which roll would you like: California, spicy tuna, rainbow, or dragon?"
            ),
        },
    ),
    "get_delivery_estimate": ToolLines(
        description="The caller asks how long delivery takes or when the food will arrive",
        result="Delivery takes about {{ result.minutes }} minutes.",
    ),
    "get_prices": ToolLines(
        description="The caller asks what something costs, or about prices",
        result=(
            "Pizzas are ten dollars for a small, fifteen for a medium, and twenty for a "
            "large. Sushi rolls are eight dollars each."
        ),
    ),
}

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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the food ordering bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        ),
    )
    # Reads TYPESAFE_API_KEY. The timeout is longer than the judge's default
    # because nothing here falls back to an LLM: a judgment that arrives late
    # still moves the order along, while a timeout only gets the caller a
    # repeat of the question.
    judge = TypeSafeJudge(timeout=3.0)
    llm = TypeSafeFlowsLLMService(
        judge=judge,
        nodes=NODE_LINES,
        tools=TOOL_LINES,
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    # Load the flow graph and join it to the handlers module. The config is
    # validated as it loads; constructing the Flow checks that every tool it
    # names exists and has a valid direct-function signature.
    config = FlowConfig.from_file(FLOW_CONFIG_PATH)
    flow = Flow(
        config,
        handlers=handlers,
    )

    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=flow.global_functions,
    )
    # The service reads the current node and the state its lines refer to.
    llm.flow_manager = flow_manager

    # Session facts the lines refer to as {{ key }}. The service fills them in
    # from the manager's state when it speaks.
    flow_manager.state.update(
        {"restaurant_name": os.getenv("RESTAURANT_NAME", "Pipecat Pizza and Sushi")}
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await flow_manager.initialize(flow.initial_node)

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
