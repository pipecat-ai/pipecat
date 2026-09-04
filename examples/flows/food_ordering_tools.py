#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools for the food-ordering flow defined in food_ordering.yaml.

Each tool is a Flows direct function: its name, description, and parameters
come from the signature and docstring, and the body does the work. None of
them chooses the next node. They return ``(result, None)`` and the flow config
decides where each one leads.
"""

from datetime import datetime, timedelta
from typing import TypedDict

from loguru import logger

from pipecat.flows import FlowManager


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


# Pre-action handler


async def check_kitchen_status(action: dict, flow_manager: FlowManager) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


# Transitions with no work behind them


async def choose_pizza(flow_manager: FlowManager) -> tuple[None, None]:
    """
    User wants to order pizza. Let's get that order started.
    """
    return None, None


async def choose_sushi(flow_manager: FlowManager) -> tuple[None, None]:
    """
    User wants to order sushi. Let's get that order started.
    """
    return None, None


async def complete_order(flow_manager: FlowManager) -> tuple[None, None]:
    """
    User confirms the order is correct.
    """
    return None, None


async def revise_order(flow_manager: FlowManager) -> tuple[None, None]:
    """
    User wants to make changes to their order.
    """
    return None, None


# Tools that do work


async def select_pizza_order(
    flow_manager: FlowManager, size: str, pizza_type: str
) -> tuple[PizzaOrderResult, None]:
    """
    Record the pizza order details.

    Args:
        size (str): Size of the pizza. Must be one of "small", "medium", or "large".
        pizza_type (str): Type of pizza. Must be one of "pepperoni", "cheese", "supreme", or "vegetarian".
    """
    base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
    price = base_price[size]

    flow_manager.state["order"] = {
        "type": "pizza",
        "size": size,
        "pizza_type": pizza_type,
        "price": price,
    }

    return PizzaOrderResult(size=size, type=pizza_type, price=price), None


async def select_sushi_order(
    flow_manager: FlowManager, count: int, roll_type: str
) -> tuple[SushiOrderResult, None]:
    """
    Record the sushi order details.

    Args:
        count (int): Number of sushi rolls to order. Must be between 1 and 10.
        roll_type (str): Type of sushi roll. Must be one of "california", "spicy tuna", "rainbow", or "dragon".
    """
    price = count * 8.00

    flow_manager.state["order"] = {
        "type": "sushi",
        "count": count,
        "roll_type": roll_type,
        "price": price,
    }

    return SushiOrderResult(count=count, type=roll_type, price=price), None


async def get_delivery_estimate(flow_manager: FlowManager) -> tuple[DeliveryEstimateResult, None]:
    """Provide delivery estimate information."""
    delivery_time = datetime.now() + timedelta(minutes=30)
    return DeliveryEstimateResult(time=f"{delivery_time}"), None
