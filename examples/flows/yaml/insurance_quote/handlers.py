#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tools for the insurance quote flow defined in flow.yaml.

Each tool is a Flows direct function: its name, description, and parameters
come from the signature and docstring, and the body does the work. None of
them chooses the next node. They return ``(result, TRANSITION_IN_YAML)`` and
the flow config decides where each one leads.

The tools that compute a quote store it in the manager's state under
``quote``, with each figure already formatted for reading aloud. The config's
quote_results node reads those fields as ``{{ quote.monthly_premium }}`` and
so on, and is rendered again each time update_coverage re-enters it.
"""

from typing import Literal, TypedDict

from pipecat.flows import TRANSITION_IN_YAML, FlowManager

# Simulated rate table.
INSURANCE_RATES = {
    "young_single": {"base_rate": 150, "risk_multiplier": 1.5},
    "young_married": {"base_rate": 130, "risk_multiplier": 1.3},
    "adult_single": {"base_rate": 100, "risk_multiplier": 1.0},
    "adult_married": {"base_rate": 90, "risk_multiplier": 0.9},
}


class AgeResult(TypedDict):
    age: int


class MaritalStatusResult(TypedDict):
    marital_status: str


class QuoteResult(TypedDict):
    monthly_premium: float
    coverage_amount: int
    deductible: int


def _store_quote(flow_manager: FlowManager, quote: QuoteResult) -> None:
    """Keep the quote in state, formatted the way the prompt reads it."""
    flow_manager.state["quote"] = {
        "monthly_premium": f"{quote['monthly_premium']:.2f}",
        "coverage_amount": f"{quote['coverage_amount']:,}",
        "deductible": f"{quote['deductible']:,}",
    }


async def collect_age(flow_manager: FlowManager, age: int):
    """Record customer's age.

    Args:
        age (int): The customer's age.
    """
    flow_manager.state["age"] = age
    return AgeResult(age=age), TRANSITION_IN_YAML


async def collect_marital_status(
    flow_manager: FlowManager, marital_status: Literal["single", "married"]
):
    """Record marital status after customer provides it.

    Args:
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    flow_manager.state["marital_status"] = marital_status
    return MaritalStatusResult(marital_status=marital_status), TRANSITION_IN_YAML


async def calculate_quote(flow_manager: FlowManager, age: int, marital_status: str):
    """Calculate initial insurance quote.

    Args:
        age (int): The customer's age.
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    age_category = "young" if age < 25 else "adult"
    rates = INSURANCE_RATES.get(f"{age_category}_{marital_status}", INSURANCE_RATES["adult_single"])
    quote = QuoteResult(
        monthly_premium=rates["base_rate"] * rates["risk_multiplier"],
        coverage_amount=250000,
        deductible=1000,
    )
    _store_quote(flow_manager, quote)
    return quote, TRANSITION_IN_YAML


async def update_coverage(flow_manager: FlowManager, coverage_amount: int, deductible: int):
    """Recalculate quote with new coverage options.

    Args:
        coverage_amount (int): The desired coverage amount in dollars.
        deductible (int): The desired deductible amount in dollars.
    """
    monthly_premium = (coverage_amount / 250000) * 100
    if deductible > 1000:
        monthly_premium *= 0.9  # 10% discount for a higher deductible
    quote = QuoteResult(
        monthly_premium=monthly_premium,
        coverage_amount=coverage_amount,
        deductible=deductible,
    )
    _store_quote(flow_manager, quote)
    return quote, TRANSITION_IN_YAML
