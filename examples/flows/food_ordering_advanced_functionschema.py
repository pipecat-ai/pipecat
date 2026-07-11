#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An "advanced" food ordering flow example using FlowsFunctionSchema.

This is the FlowsFunctionSchema counterpart to the standard food_ordering.py
(which uses direct functions). Direct functions are the recommended way to
define a node's functions: their schema is derived from the function signature
and docstring. Reach for a FlowsFunctionSchema when you need property control
the direct-function generator can't give you — for example a strict ``enum``
constraint or a numeric ``minimum``/``maximum`` (both used below, on the pizza
size and type and the sushi count and type), which a direct function can only
hint at in prose in its docstring.

The flow handles:

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
from pipecat.flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)
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


# Node creation functions
def create_initial_node() -> NodeConfig:
    """Create the initial node for food type selection."""

    async def choose_pizza(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to pizza order selection."""
        return None, create_pizza_node()

    async def choose_sushi(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to sushi order selection."""
        return None, create_sushi_node()

    choose_pizza_func = FlowsFunctionSchema(
        name="choose_pizza",
        handler=choose_pizza,
        description="User wants to order pizza. Let's get that order started.",
        properties={},
        required=[],
    )

    choose_sushi_func = FlowsFunctionSchema(
        name="choose_sushi",
        handler=choose_sushi,
        description="User wants to order sushi. Let's get that order started.",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="initial",
        role_message="You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.",
        task_messages=[
            {
                "role": "developer",
                "content": "For this step, ask the user if they want pizza or sushi, and wait for them to use a function to choose. Start off by greeting them. Be friendly and casual; you're taking an order for food over the phone.",
            }
        ],
        pre_actions=[
            {
                "type": "function",
                "handler": check_kitchen_status,
            },
        ],
        functions=[choose_pizza_func, choose_sushi_func],
    )


def create_pizza_node() -> NodeConfig:
    """Create the pizza ordering node."""

    async def select_pizza_order(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[PizzaOrderResult, NodeConfig]:
        """Handle pizza size and type selection."""
        size = args["size"]
        pizza_type = args["type"]

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

    # Spelling the schema out explicitly gives precise control over the
    # parameters — here, strict ``enum`` constraints on size and type (and, in
    # the sushi node, a numeric ``minimum``/``maximum`` on the roll count) —
    # that a direct function could only describe in prose.
    select_pizza_func = FlowsFunctionSchema(
        name="select_pizza_order",
        handler=select_pizza_order,
        description="Record the pizza order details",
        properties={
            "size": {
                "type": "string",
                "enum": ["small", "medium", "large"],
                "description": "Size of the pizza",
            },
            "type": {
                "type": "string",
                "enum": ["pepperoni", "cheese", "supreme", "vegetarian"],
                "description": "Type of pizza",
            },
        },
        required=["size", "type"],
    )

    return NodeConfig(
        name="choose_pizza",
        task_messages=[
            {
                "role": "developer",
                "content": """You are handling a pizza order.

As soon as the user has given both a size AND a type, immediately call
select_pizza_order to record it. Do not acknowledge the order conversationally,
ask whether they want anything else, or wait for further confirmation first — the
confirmation step handles all of that. If the size or the type is still missing,
ask only for the missing detail.

Pricing:
- Small: $10
- Medium: $15
- Large: $20

Remember to be friendly and casual.""",
            }
        ],
        functions=[select_pizza_func],
    )


def create_sushi_node() -> NodeConfig:
    """Create the sushi ordering node."""

    async def select_sushi_order(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[SushiOrderResult, NodeConfig]:
        """Handle sushi roll count and type selection."""
        count = args["count"]
        roll_type = args["type"]

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

    select_sushi_func = FlowsFunctionSchema(
        name="select_sushi_order",
        handler=select_sushi_order,
        description="Record the sushi order details",
        properties={
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of rolls to order",
            },
            "type": {
                "type": "string",
                "enum": ["california", "spicy tuna", "rainbow", "dragon"],
                "description": "Type of sushi roll",
            },
        },
        required=["count", "type"],
    )

    return NodeConfig(
        name="choose_sushi",
        task_messages=[
            {
                "role": "developer",
                "content": """You are handling a sushi order.

As soon as the user has given both a roll count AND a roll type, immediately call
select_sushi_order to record it. Do not acknowledge the order conversationally,
ask whether they want anything else, or wait for further confirmation first — the
confirmation step handles all of that. If the count or the type is still missing,
ask only for the missing detail.

Pricing:
- $8 per roll

Remember to be friendly and casual.""",
            }
        ],
        functions=[select_sushi_func],
    )


def create_confirmation_node() -> NodeConfig:
    """Create the order confirmation node."""

    async def complete_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to end state."""
        return None, create_end_node()

    async def revise_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to start for order revision."""
        return None, create_initial_node()

    complete_order_func = FlowsFunctionSchema(
        name="complete_order",
        handler=complete_order,
        description="User confirms the order is correct",
        properties={},
        required=[],
    )

    revise_order_func = FlowsFunctionSchema(
        name="revise_order",
        handler=revise_order,
        description="User wants to make changes to their order",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "developer",
                "content": """Read back the complete order details to the user and ask if they want anything else or if they want to make changes. Use the available functions:
- Use complete_order when the user confirms that the order is correct and no changes are needed
- Use revise_order if they want to change something

Be friendly and clear when reading back the order details.""",
            }
        ],
        functions=[complete_order_func, revise_order_func],
    )


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "developer",
                "content": "Thank the user for their order and end the conversation politely and concisely.",
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
    )

    # Define "global" functions available at every node
    async def get_delivery_estimate(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[DeliveryEstimateResult, None]:
        """Provide delivery estimate information."""
        delivery_time = datetime.now() + timedelta(minutes=30)
        return DeliveryEstimateResult(
            time=f"{delivery_time}",
        ), None

    get_delivery_estimate_func = FlowsFunctionSchema(
        name="get_delivery_estimate",
        handler=get_delivery_estimate,
        description="Get a delivery estimate for the current order",
        properties={},
        required=[],
    )

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=[get_delivery_estimate_func],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation with the initial node
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
