#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The food ordering flow, configured from YAML, routed by TypeSafe, voiced by PhoneLLM.

The same conversation as python/food_ordering.py, split along the seam Pipecat
Flows offers for runtime configuration:

- flow.yaml holds the graph: the nodes, which tools each offers, where each
  tool leads, and the prompt the LLM speaks from in each node.
- handlers.py holds the tools: direct functions whose schema comes from
  their signature and docstring.
- This file holds the written lines the bot says when it is sure, and wires
  the pipeline.

TypeSafeFlowsRouter sits in front of the LLM. At the end of each caller turn
it asks TypeSafe's Jev which of the node's tools the turn calls for, what each
argument is, and whether the caller said anything more than what was asked.
When every answer is sure (0.9 and up), the router runs the tool and speaks
the next node's written line itself, in about a quarter of a second. When the
tool is clear but something is less sure, such as a caller who also asked a
question, the router still runs the tool and PhoneLLM speaks the next node
from its prompt. When nothing is clear, the whole turn goes to PhoneLLM,
tools included. Tool results (prices, delivery times) are always phrased by
PhoneLLM.

PhoneLLM is Pipecat's open-weights model for phone calls, served from a Modal
endpoint through the OpenAI-compatible API. Set LLM_SERVICE=openai to use
OpenAI instead.

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- TYPESAFE_API_KEY (for the judgments)
- MODAL_ENDPOINT_URL and MODAL_API_KEY (for PhoneLLM), plus PHONELLM_MODEL
  when the endpoint serves the model under a name other than
  pipecat-ai/phonellm-alpha-1; or OPENAI_API_KEY with LLM_SERVICE=openai
- DAILY_API_KEY (for the Daily transport)
"""

import os
from pathlib import Path
from typing import Any

import handlers
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import Flow, FlowConfig, FlowManager
from pipecat.flows.typesafe_flows_router import TypeSafeFlowsRouter
from pipecat.flows.typesafe_llm import NodeLines, ToolLines
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
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.typesafe import TypeSafeJudge
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

FLOW_CONFIG_PATH = Path(__file__).with_name("flow.yaml")

# What the bot says on entering each node of flow.yaml when TypeSafe was sure
# about the turn that led there. A node without a line is always spoken by
# the LLM from its prompt.
NODE_LINES = {
    "initial": NodeLines(
        say="Hi, welcome to {{ restaurant_name }}! Would you like pizza or sushi today?",
    ),
    "choose_pizza": NodeLines(
        say=(
            "Great, pizza it is! What size would you like, and what kind? "
            "We have cheese, pepperoni, supreme, and vegetarian."
        ),
    ),
    "choose_sushi": NodeLines(
        say=(
            "Sushi, nice choice! How many rolls would you like, and which kind? "
            "We have California, spicy tuna, rainbow, and dragon."
        ),
    ),
    "confirm": NodeLines(
        say="So that's {{ order.summary }}, {{ order.total }} total. Does that sound right?",
    ),
    "restart": NodeLines(say="No problem, let's start over. Would you like pizza or sushi?"),
    "end": NodeLines(say="Thanks for your order! It'll be on its way soon. Goodbye!"),
}

# How each tool is judged, the values its arguments can take, and the written
# question for an argument the caller plainly left out. Tools not listed here
# (choose_pizza, choose_sushi, complete_order, revise_order) are judged by
# the descriptions in flow.yaml.
TOOL_LINES = {
    "select_pizza_order": ToolLines(
        description="The caller gives pizza order details: a size, a kind of pizza, or both",
        examples=["A large pepperoni", "Medium please", "Cheese", "The veggie one"],
        options={
            "size": ["small", "medium", "large"],
            "pizza_type": ["cheese", "pepperoni", "supreme", "vegetarian"],
        },
        option_examples={
            "size": {"small": ["a small one", "the smallest"], "large": ["big", "the biggest"]},
            "pizza_type": {"vegetarian": ["veggie", "the vegetable one"]},
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
        examples=["Two California rolls", "Spicy tuna", "Three, please"],
        options={
            "count": list(range(1, 11)),
            "roll_type": ["california", "spicy tuna", "rainbow", "dragon"],
        },
        option_examples={
            "count": {"1": ["one", "a single roll", "just one"], "2": ["two", "a couple"]},
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
    ),
    "get_prices": ToolLines(
        description="The caller asks what something costs, or about prices",
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


def require_env(name: str) -> str:
    """Return a required environment variable or raise."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_llm() -> LLMService[Any]:
    """Build the conversation LLM selected by ``LLM_SERVICE`` (``phonellm`` or ``openai``).

    Flows sets the prompt per node, so neither service gets a system
    instruction here.
    """
    service = os.getenv("LLM_SERVICE", "phonellm").strip().lower()
    logger.info(f"LLM service: {service}")

    if service == "openai":
        return OpenAILLMService(
            api_key=require_env("OPENAI_API_KEY"),
            settings=OpenAILLMService.Settings(model=os.getenv("OPENAI_MODEL", "gpt-4.1")),
        )
    if service != "phonellm":
        raise RuntimeError(f"Unknown LLM_SERVICE: {service!r} (expected 'phonellm' or 'openai')")

    # PhoneLLM served by a Modal endpoint (OpenAI-compatible API).
    # MODAL_ENDPOINT_URL is the URL printed by `modal endpoint create`;
    # MODAL_API_KEY is a proxy token, combined as <token-id>.<token-secret>.
    return OpenAILLMService(
        api_key=require_env("MODAL_API_KEY"),
        base_url=require_env("MODAL_ENDPOINT_URL"),
        settings=OpenAILLMService.Settings(
            model=os.getenv("PHONELLM_MODEL", "pipecat-ai/phonellm-alpha-1"),
            # PhoneLLM is trained for temperature 0.
            temperature=0,
            # Disable reasoning.
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        ),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the food ordering bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        ),
    )
    llm = build_llm()

    # Reads TYPESAFE_API_KEY. A judgment that arrives late still saves the
    # LLM a turn, and a timeout costs only the fallback to the LLM, so the
    # timeout is longer than the judge's default.
    judge = TypeSafeJudge(timeout=3.0)
    router = TypeSafeFlowsRouter(
        judge=judge,
        llm=llm,
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
            router,
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
    # The router reads the current node and the state its lines refer to.
    router.flow_manager = flow_manager

    # Session facts the lines and prompts refer to as {{ key }}.
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
