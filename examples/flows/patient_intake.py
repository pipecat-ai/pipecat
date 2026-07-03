#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A patient intake flow example for Pipecat Flows.

This example demonstrates a medical intake system using flows with direct
functions where conversation paths are determined at runtime. The flow handles:

1. Patient identity verification through birthday
2. Prescription collection
3. Allergy information gathering
4. Medical conditions collection
5. Visit reason documentation
6. Information verification and confirmation

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai_responses (default), openai, anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (optionalfor transport)
- LLM API key (varies by provider - see env.example)
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from loguru import logger
from utils import create_llm

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import (
    ContextStrategy,
    ContextStrategyConfig,
    FlowManager,
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
class Prescription(TypedDict):
    medication: str
    dosage: str


class Allergy(TypedDict):
    name: str


class Condition(TypedDict):
    name: str


class VisitReason(TypedDict):
    name: str


# Result types for each handler
class BirthdayVerificationResult(TypedDict):
    verified: bool


class PrescriptionRecordResult(TypedDict):
    count: int


class AllergyRecordResult(TypedDict):
    count: int


class ConditionRecordResult(TypedDict):
    count: int


class VisitReasonRecordResult(TypedDict):
    count: int


# Functions
async def verify_birthday(
    flow_manager: FlowManager, birthday: str
) -> tuple[BirthdayVerificationResult, NodeConfig]:
    """Verify the user has provided their correct birthday. Once confirmed, the next step is to record the user's prescriptions.

    Args:
        birthday (str): The user's birthdate (convert to YYYY-MM-DD format).
    """
    # In a real app, this would verify against patient records
    is_valid = birthday == "1983-01-01"

    # Store verification result in flow state
    flow_manager.state["birthday_verified"] = is_valid
    flow_manager.state["birthday"] = birthday

    return BirthdayVerificationResult(verified=is_valid), create_prescriptions_node()


async def record_prescriptions(
    flow_manager: FlowManager, prescriptions: list[dict]
) -> tuple[PrescriptionRecordResult, NodeConfig]:
    """Record the user's prescriptions. Once confirmed, the next step is to collect allergy information.

    Args:
        prescriptions (list[dict]): List of prescription objects, each with "medication" (str, the medication's name) and "dosage" (str, the prescription's dosage).
    """
    # Store prescriptions in flow state
    flow_manager.state["prescriptions"] = prescriptions

    # In a real app, this would store in patient records
    return PrescriptionRecordResult(count=len(prescriptions)), create_allergies_node()


async def record_allergies(
    flow_manager: FlowManager, allergies: list[dict]
) -> tuple[AllergyRecordResult, NodeConfig]:
    """Record the user's allergies. Once confirmed, then next step is to collect medical conditions.

    Args:
        allergies (list[dict]): List of allergy objects, each with "name" (str, what the user is allergic to).
    """
    # Store allergies in flow state
    flow_manager.state["allergies"] = allergies

    # In a real app, this would store in patient records
    return AllergyRecordResult(count=len(allergies)), create_conditions_node()


async def record_conditions(
    flow_manager: FlowManager, conditions: list[dict]
) -> tuple[ConditionRecordResult, NodeConfig]:
    """Record the user's medical conditions. Once confirmed, the next step is to collect visit reasons.

    Args:
        conditions (list[dict]): List of condition objects, each with "name" (str, the user's medical condition).
    """
    # Store conditions in flow state
    flow_manager.state["conditions"] = conditions

    # In a real app, this would store in patient records
    return ConditionRecordResult(count=len(conditions)), create_visit_reasons_node()


async def record_visit_reasons(
    flow_manager: FlowManager, visit_reasons: list[dict]
) -> tuple[VisitReasonRecordResult, NodeConfig]:
    """Record the reasons for their visit. Once confirmed, the next step is to verify all information.

    Args:
        visit_reasons (list[dict]): List of visit reason objects, each with "name" (str, the user's reason for visiting).
    """
    # Store visit reasons in flow state
    flow_manager.state["visit_reasons"] = visit_reasons

    # In a real app, this would store in patient records
    return VisitReasonRecordResult(count=len(visit_reasons)), create_verification_node()


async def revise_information(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Return to prescriptions to revise information."""
    return None, create_prescriptions_node()


async def confirm_information(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Proceed with confirmed information."""
    return None, create_confirmation_node()


async def complete_intake(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Complete the intake process."""
    return None, create_end_node()


# Node creation functions
def create_initial_node() -> NodeConfig:
    """Create the initial node for patient identity verification."""
    return NodeConfig(
        name="start",
        role_message="You are Jessica, an agent for Tri-County Health Services. You must ALWAYS use one of the available functions to progress the conversation. Be professional but friendly.",
        task_messages=[
            {
                "role": "developer",
                "content": "Start by introducing yourself to Chad Bailey, then ask for their date of birth, including the year. Once they provide their birthday, use verify_birthday to check it. If verified (1983-01-01), proceed to prescriptions.",
            }
        ],
        functions=[verify_birthday],
    )


def create_prescriptions_node() -> NodeConfig:
    """Create the prescriptions collection node."""
    return NodeConfig(
        name="get_prescriptions",
        role_message="You are Jessica, an agent for Tri-County Health Services. You must ALWAYS use one of the available functions to progress the conversation. Be professional but friendly.",
        task_messages=[
            {
                "role": "developer",
                "content": "This step is for collecting prescriptions. Ask them what prescriptions they're taking, including the dosage. Get to the point by saying 'Thanks for confirming that. First up, what prescriptions are you currently taking, including the dosage for each medication?'. After recording prescriptions (or confirming none), proceed to allergies.",
            }
        ],
        context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        functions=[record_prescriptions],
    )


def create_allergies_node() -> NodeConfig:
    """Create the allergies collection node."""
    return NodeConfig(
        name="get_allergies",
        task_messages=[
            {
                "role": "developer",
                "content": "Collect allergy information. Ask about any allergies they have. After recording allergies (or confirming none), proceed to medical conditions.",
            }
        ],
        functions=[record_allergies],
    )


def create_conditions_node() -> NodeConfig:
    """Create the medical conditions collection node."""
    return NodeConfig(
        name="get_conditions",
        task_messages=[
            {
                "role": "developer",
                "content": "Collect medical condition information. Ask about any medical conditions they have. After recording conditions (or confirming none), proceed to visit reasons.",
            }
        ],
        functions=[record_conditions],
    )


def create_visit_reasons_node() -> NodeConfig:
    """Create the visit reasons collection node."""
    return NodeConfig(
        name="get_visit_reasons",
        task_messages=[
            {
                "role": "developer",
                "content": "Collect information about the reason for their visit. Ask what brings them to the doctor today. After recording their reasons, proceed to verification.",
            }
        ],
        functions=[record_visit_reasons],
    )


def create_verification_node() -> NodeConfig:
    """Create the information verification node with context reset and summary."""
    return NodeConfig(
        name="verify",
        task_messages=[
            {
                "role": "developer",
                "content": """Review all collected information with the patient. Follow these steps:
1. Summarize their prescriptions, allergies, conditions, and visit reasons
2. Ask if everything is correct
3. Use the appropriate function based on their response

Be thorough in reviewing all details and wait for explicit confirmation.""",
            }
        ],
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY,
            summary_prompt=(
                "Summarize the patient intake conversation, including their birthday, "
                "prescriptions, allergies, medical conditions, and reasons for visiting. "
                "Focus on the specific medical information provided."
            ),
        ),
        functions=[revise_information, confirm_information],
    )


def create_confirmation_node() -> NodeConfig:
    """Create the final confirmation node."""
    return NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "developer",
                "content": "Once confirmed, thank them, then use the complete_intake function to end the conversation.",
            }
        ],
        functions=[complete_intake],
    )


def create_end_node() -> NodeConfig:
    """Create the final end node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "developer",
                "content": "Thank them for their time and end the conversation.",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the patient intake bot."""
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
