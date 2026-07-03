#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Insurance Quote Example using Pipecat Flows.

This example demonstrates how to create a conversational insurance quote bot using:
- Flow management for flexible conversation paths
- Node configurations for different conversation states
- Pre/post actions for user feedback
- Transition logic based on user responses

The flow allows users to:
1. Provide their age
2. Specify marital status
3. Get an insurance quote
4. Adjust coverage options
5. Complete the quote process

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
from typing import Any, TypedDict

from dotenv import load_dotenv
from loguru import logger
from utils import create_llm

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
class InsuranceQuote(TypedDict):
    monthly_premium: float
    coverage_amount: int
    deductible: int


class AgeCollectionResult(TypedDict):
    age: int


class MaritalStatusResult(TypedDict):
    marital_status: str


class QuoteCalculationResult(InsuranceQuote):
    pass


class CoverageUpdateResult(InsuranceQuote):
    pass


# Simulated insurance data
INSURANCE_RATES = {
    "young_single": {"base_rate": 150, "risk_multiplier": 1.5},
    "young_married": {"base_rate": 130, "risk_multiplier": 1.3},
    "adult_single": {"base_rate": 100, "risk_multiplier": 1.0},
    "adult_married": {"base_rate": 90, "risk_multiplier": 0.9},
}


# Functions
async def collect_age(
    flow_manager: FlowManager, age: int
) -> tuple[AgeCollectionResult, NodeConfig]:
    """Record customer's age.

    Args:
        age (int): The customer's age.
    """
    logger.debug(f"collect_age handler executing with age: {age}")

    flow_manager.state["age"] = age
    result = AgeCollectionResult(age=age)

    next_node = create_marital_status_node()

    return result, next_node


async def collect_marital_status(
    flow_manager: FlowManager, marital_status: str
) -> tuple[MaritalStatusResult, NodeConfig]:
    """Record marital status after customer provides it.

    Args:
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    logger.debug(f"collect_marital_status handler executing with status: {marital_status}")

    result = MaritalStatusResult(marital_status=marital_status)

    next_node = create_quote_calculation_node(flow_manager.state["age"], marital_status)

    return result, next_node


async def calculate_quote(
    flow_manager: FlowManager, age: int, marital_status: str
) -> tuple[QuoteCalculationResult, NodeConfig]:
    """Calculate initial insurance quote.

    Args:
        age (int): The customer's age.
        marital_status (str): The customer's marital status. Must be one of "single", "married".
    """
    logger.debug(f"calculate_quote handler executing with age: {age}, status: {marital_status}")

    # Determine rate category
    age_category = "young" if age < 25 else "adult"
    rate_key = f"{age_category}_{marital_status}"
    rates = INSURANCE_RATES.get(rate_key, INSURANCE_RATES["adult_single"])

    # Calculate quote
    monthly_premium = rates["base_rate"] * rates["risk_multiplier"]

    result = QuoteCalculationResult(
        monthly_premium=monthly_premium,
        coverage_amount=250000,
        deductible=1000,
    )
    next_node = create_quote_results_node(result)
    return result, next_node


async def update_coverage(
    flow_manager: FlowManager, coverage_amount: int, deductible: int
) -> tuple[CoverageUpdateResult, NodeConfig]:
    """Recalculate quote with new coverage options.

    Args:
        coverage_amount (int): The desired coverage amount in dollars.
        deductible (int): The desired deductible amount in dollars.
    """
    logger.debug(
        f"update_coverage handler executing with amount: {coverage_amount}, deductible: {deductible}"
    )

    # Calculate adjusted quote
    monthly_premium = (coverage_amount / 250000) * 100
    if deductible > 1000:
        monthly_premium *= 0.9  # 10% discount for higher deductible

    result = CoverageUpdateResult(
        monthly_premium=monthly_premium,
        coverage_amount=coverage_amount,
        deductible=deductible,
    )
    next_node = create_quote_results_node(result)
    return result, next_node


async def end_quote(flow_manager: FlowManager) -> tuple[Any, NodeConfig]:
    """Complete the quote process when customer is satisfied."""
    logger.debug("end_quote handler executing")
    return {"status": "completed"}, create_end_node()


# Node configurations
def create_initial_node() -> NodeConfig:
    """Create the initial node asking for age."""
    return NodeConfig(
        name="initial",
        role_message="You are a friendly insurance agent. Your responses will be converted to audio, so avoid special characters. Always use the available functions to progress the conversation naturally.",
        task_messages=[
            {
                "role": "developer",
                "content": "Start by asking for the customer's age.",
            }
        ],
        functions=[collect_age],
    )


def create_marital_status_node() -> NodeConfig:
    """Create node for collecting marital status."""
    return NodeConfig(
        name="marital_status",
        task_messages=[
            {
                "role": "developer",
                "content": "Ask about the customer's marital status for premium calculation.",
            }
        ],
        functions=[collect_marital_status],
    )


def create_quote_calculation_node(age: int, marital_status: str) -> NodeConfig:
    """Create node for calculating initial quote."""
    return NodeConfig(
        name="quote_calculation",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Calculate a quote for {age} year old {marital_status} customer. "
                    "First, call calculate_quote with their information. "
                    "Then explain the quote details and ask if they'd like to adjust coverage."
                ),
            }
        ],
        functions=[calculate_quote],
    )


def create_quote_results_node(
    quote: QuoteCalculationResult | CoverageUpdateResult,
) -> NodeConfig:
    """Create node for showing quote and adjustment options."""
    return NodeConfig(
        name="quote_results",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Quote details:\n"
                    f"Monthly Premium: ${quote['monthly_premium']:.2f}\n"
                    f"Coverage Amount: ${quote['coverage_amount']:,}\n"
                    f"Deductible: ${quote['deductible']:,}\n\n"
                    "Explain these quote details to the customer. When they request changes, "
                    "use update_coverage to recalculate their quote. Explain how their "
                    "changes affected the premium and compare it to their previous quote. "
                    "Ask if they'd like to make any other adjustments or if they're ready "
                    "to end the quote process."
                ),
            }
        ],
        functions=[update_coverage, end_quote],
    )


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Thank the customer for their time and end the conversation. "
                    "Mention that a representative will contact them about the quote."
                ),
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the insurance quote bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
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

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
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
