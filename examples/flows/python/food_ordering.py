#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A food ordering flow example for Pipecat Flows.

This example demonstrates a food ordering system using flows where
conversation paths are determined at runtime. The flow handles:

1. Initial greeting and food type selection (pizza or sushi)
2. Order details collection based on food type
3. Order confirmation and revision
4. Order completion

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai_responses (default), openai, anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import os
from datetime import datetime, timedelta
from typing import TypedDict

from dotenv import load_dotenv
from loguru import logger
from utils import create_llm

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import FlowManager, NodeConfig
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


# Type definitions
class PizzaOrderResult(TypedDict):
    size: str
    type: str
    price: float


class SushiOrderResult(TypedDict):
    count: int
    type: str
    price: float


class DeliveryEstimateResult(TypedDict):
    time: str


# Pre-action handlers
async def check_kitchen_status(action: dict, flow_manager: FlowManager) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


# Functions for Initial Node
async def choose_pizza(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """
    User wants to order pizza. Let's get that order started.
    """
    return None, create_pizza_node()


async def choose_sushi(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """
    User wants to order sushi. Let's get that order started.
    """
    return None, create_sushi_node()


# Functions for Pizza Node
async def select_pizza_order(
    flow_manager: FlowManager, size: str, pizza_type: str
) -> tuple[PizzaOrderResult, NodeConfig]:
    """
    Record the pizza order details.

    Args:
        size (str): Size of the pizza. Must be one of "small", "medium", or "large".
        pizza_type (str): Type of pizza. Must be one of "pepperoni", "cheese", "supreme", or "vegetarian".
    """
    # Simple pricing
    base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
    price = base_price[size]

    result = PizzaOrderResult(size=size, type=pizza_type, price=price)

    # Store order details in flow state
    flow_manager.state["order"] = {
        "type": "pizza",
        "size": size,
        "pizza_type": pizza_type,
        "price": price,
    }

    return result, create_confirmation_node()


# Functions for Sushi Node
async def select_sushi_order(
    flow_manager: FlowManager, count: int, roll_type: str
) -> tuple[SushiOrderResult, NodeConfig]:
    """
    Record the sushi order details.

    Args:
        count (int): Number of sushi rolls to order. Must be between 1 and 10.
        roll_type (str): Type of sushi roll. Must be one of "california", "spicy tuna", "rainbow", or "dragon".
    """
    # Simple pricing: $8 per roll
    price = count * 8.00

    result = SushiOrderResult(count=count, type=roll_type, price=price)

    # Store order details in flow state
    flow_manager.state["order"] = {
        "type": "sushi",
        "count": count,
        "roll_type": roll_type,
        "price": price,
    }

    return result, create_confirmation_node()


# Functions for Confirmation Node
async def complete_order(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """
    User confirms the order is correct.
    """
    return None, create_end_node()


async def revise_order(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """
    User wants to make changes to their order.
    """
    return None, create_initial_node()


# Node creation functions
def create_initial_node() -> NodeConfig:
    """Create the initial node for food type selection."""
    return NodeConfig(
        name="initial",
        role_message="You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Keep every reply to one or two short sentences, the way a person on the phone talks, and don't repeat the order unless the caller asks. Only give a delivery time that came from get_delivery_estimate. Avoid outputting special characters and emojis.",
        task_messages=[
            {
                "role": "developer",
                "content": "Greet the caller briefly and ask whether they'd like pizza or sushi, then wait for them to use a function to choose. Be friendly and casual; you're taking an order for food over the phone.",
            }
        ],
        pre_actions=[
            {
                "type": "function",
                "handler": check_kitchen_status,
            },
        ],
        functions=[choose_pizza, choose_sushi],
    )


def create_pizza_node() -> NodeConfig:
    """Create the pizza ordering node."""
    return NodeConfig(
        name="choose_pizza",
        task_messages=[
            {
                "role": "developer",
                "content": """Your only job on this step is to get the pizza's size and type and call select_pizza_order with them. If the caller has already given both, call it immediately; if one is missing, ask for it in one short sentence. Don't quote a price or say the order is in: the function works out the price and the next step reads the order back. Pizza is the only thing on this menu; don't offer drinks or sides.

Pricing, for questions only:
- Small: $10
- Medium: $15
- Large: $20""",
            }
        ],
        functions=[select_pizza_order],
    )


def create_sushi_node() -> NodeConfig:
    """Create the sushi ordering node."""
    return NodeConfig(
        name="choose_sushi",
        task_messages=[
            {
                "role": "developer",
                "content": """Your only job on this step is to get the roll count and type and call select_sushi_order with them. If the caller has already given both, call it immediately; if one is missing, ask for it in one short sentence. Don't quote a price or say the order is in: the function works out the price and the next step reads the order back. Sushi is the only thing on this menu; don't offer drinks or sides.

Pricing, for questions only:
- $8 per roll""",
            }
        ],
        functions=[select_sushi_order],
    )


def create_confirmation_node() -> NodeConfig:
    """Create the order confirmation node."""
    return NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "developer",
                "content": """Say the order back once with the total and ask if that's right, for example "So that's one large pepperoni, twenty dollars. Sound good?" Nothing is ordered until the caller confirms, so don't say the order is placed or on its way before then. If they ask a question first, answer it in a sentence without repeating the order. Use the available functions:
- Use complete_order when the user confirms the order is correct
- Use revise_order if they want to change something""",
            }
        ],
        functions=[complete_order, revise_order],
    )


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "developer",
                "content": "Thank the caller for the order and say goodbye, in one or two sentences.",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
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
    # LLM service is created using the create_llm function from utils.py
    # Default is OpenAI; can be changed by setting LLM_PROVIDER environment variable
    llm = create_llm()

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

    # Define "global" functions available at every node
    async def get_delivery_estimate(
        flow_manager: FlowManager,
    ) -> tuple[DeliveryEstimateResult, None]:
        """Provide delivery estimate information."""
        delivery_time = datetime.now() + timedelta(minutes=30)
        return DeliveryEstimateResult(
            time=f"{delivery_time}",
        ), None

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=[get_delivery_estimate],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation with the initial node
        await flow_manager.initialize(create_initial_node())

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
