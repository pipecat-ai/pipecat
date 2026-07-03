#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Multi-worker handoff: a free-form LLM router and a structured Flows worker.

This example demonstrates how Pipecat Flows composes with Pipecat's
multi-worker framework. Three workers share a single bus:

- A *main* worker owns the transport (STT, TTS) and the shared conversation
  context. It does not run an LLM itself; instead it bridges user/assistant
  frames onto the bus so other workers can take turns speaking to the user.
- A *router* worker (a plain ``LLMWorker``) handles open-ended chit-chat and
  general questions about the restaurant. When the user wants to book a table
  it hands off to the reservation worker.
- A *reservation* worker (``build_reservation_worker``) drives a structured
  Pipecat Flows conversation: party size, then time, then an availability
  check, then confirmation. When it's done — or if the user changes their
  mind — it hands control back to the router.

Only one worker is active at a time. Hand-offs are seamless: the user never
hears that they've been transferred.

The reservation worker is built as a plain ``PipelineWorker`` (no subclass),
the same way the sensor-controller example builds its worker. A ``FlowManager``
is wired onto the worker and the flow is (re)initialized from the worker's
``on_activated`` event handler each time control is handed to it. The shared
``LLMContextAggregatorPair`` is owned by the main worker, so every worker
speaks into the same conversation history.

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai_responses (default), openai, anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import asyncio
import os
from typing import Any, TypedDict

from dotenv import load_dotenv
from loguru import logger
from utils import create_llm

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.bus import BusBridgeProcessor
from pipecat.evals.transport import EvalTransportParams
from pipecat.flows import FlowManager, FlowResult, NodeConfig
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.llm import LLMWorker, LLMWorkerActivationArgs, tool
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

MAIN_NAME = "restaurant"
ROUTER_NAME = "router"
RESERVATION_NAME = "reservation"


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


# =============================================================================
# Mock reservation backend.
# =============================================================================


class MockReservationSystem:
    """Simulates a restaurant reservation API."""

    booked_times = {"7:00 PM", "8:00 PM"}

    async def check_availability(self, party_size: int, time: str) -> tuple[bool, list[str]]:
        """Return whether a time is open and, if not, some alternatives."""
        await asyncio.sleep(0.5)  # Simulate a network call.
        is_available = time not in self.booked_times
        alternatives: list[str] = []
        if not is_available:
            all_times = ["5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM"]
            alternatives = [t for t in all_times if t not in self.booked_times]
        return is_available, alternatives


# =============================================================================
# Reservation worker: a structured Pipecat Flows conversation.
# =============================================================================


class PartySizeResult(TypedDict):
    """Result of recording the party size."""

    size: int


class AvailabilityResult(TypedDict):
    """Result of an availability check."""

    time: str
    available: bool


def build_reservation_worker(
    *,
    llm: Any,
    context_aggregator: LLMContextAggregatorPair,
    reservation_system: MockReservationSystem,
) -> PipelineWorker:
    """Build the reservation worker: a Flows conversation as a ``PipelineWorker``.

    The worker's pipeline is just the LLM. ``bridged=()`` wraps it with bus
    edge processors so user frames arrive from the main worker and generated
    frames are sent back the same way. A ``FlowManager`` drives the
    conversation; it shares the main worker's ``context_aggregator`` so the
    whole session uses a single conversation history.

    The worker starts inactive (``active=False``) and stays quiet until the
    router hands it control. The ``on_activated`` event handler initializes the
    flow the first time and resumes it on subsequent hand-offs.
    """
    worker = PipelineWorker(
        Pipeline([llm]),
        name=RESERVATION_NAME,
        active=False,
        bridged=(),
    )

    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
    )

    # --- Nodes -------------------------------------------------------------

    def party_size_node() -> NodeConfig:
        return NodeConfig(
            name="party_size",
            role_message=(
                "You are a reservation assistant for La Maison, an upscale French "
                "restaurant. Be casual and friendly. This is a voice conversation, "
                "so avoid special characters and emojis."
            ),
            task_messages=[
                {"role": "developer", "content": "Ask how many people are in their party."}
            ],
            functions=[collect_party_size, transfer_to_router],
        )

    def time_selection_node() -> NodeConfig:
        return NodeConfig(
            name="get_time",
            task_messages=[
                {
                    "role": "developer",
                    "content": (
                        "Ask what time they would like to dine. The restaurant is "
                        "open from 5 PM to 10 PM."
                    ),
                }
            ],
            functions=[check_availability, transfer_to_router],
        )

    def confirmation_node() -> NodeConfig:
        return NodeConfig(
            name="confirm",
            task_messages=[
                {
                    "role": "developer",
                    "content": "Confirm the reservation details and ask if there is anything else.",
                }
            ],
            functions=[end_reservation, transfer_to_router],
        )

    def end_node() -> NodeConfig:
        return NodeConfig(
            name="end",
            task_messages=[
                {
                    "role": "developer",
                    "content": "Thank them for their reservation and say goodbye.",
                }
            ],
            post_actions=[{"type": "end_conversation"}],
        )

    # --- Flow functions ----------------------------------------------------

    async def collect_party_size(
        flow_manager: FlowManager, size: int
    ) -> tuple[PartySizeResult, NodeConfig]:
        """Record the number of people in the party.

        Args:
            size (int): Number of people in the party. Must be between 1 and 12.
        """
        flow_manager.state["party_size"] = size
        return PartySizeResult(size=size), time_selection_node()

    async def check_availability(
        flow_manager: FlowManager, time: str
    ) -> tuple[AvailabilityResult, NodeConfig]:
        """Check availability for the requested time.

        Args:
            time (str): Reservation time (e.g., '6:00 PM').
        """
        party_size = flow_manager.state.get("party_size", 2)
        is_available, alternatives = await reservation_system.check_availability(party_size, time)

        if is_available:
            flow_manager.state["time"] = time
            return AvailabilityResult(time=time, available=True), confirmation_node()

        times_list = ", ".join(alternatives)
        no_availability = NodeConfig(
            name="no_availability",
            task_messages=[
                {
                    "role": "developer",
                    "content": (
                        f"Apologize that {time} is not available. "
                        f"Suggest these alternative times: {times_list}."
                    ),
                }
            ],
            functions=[check_availability, transfer_to_router],
        )
        return AvailabilityResult(time=time, available=False), no_availability

    async def end_reservation(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Confirm and end the reservation."""
        return None, end_node()

    async def transfer_to_router(
        flow_manager: FlowManager, reason: str
    ) -> tuple[FlowResult, NodeConfig]:
        """Hand the conversation back to the general assistant.

        Call this when the user no longer wants to make a reservation, or asks
        a general question unrelated to booking a table.

        Args:
            reason (str): Why control is being handed back (e.g. 'user changed
                their mind', 'user asked about the menu').
        """
        logger.info(f"Worker '{RESERVATION_NAME}': handing back to '{ROUTER_NAME}' ({reason})")
        await worker.activate_worker(
            ROUTER_NAME,
            args=LLMWorkerActivationArgs(
                messages=[{"role": "developer", "content": reason}],
            ),
            deactivate_self=True,
        )
        return {"status": "transferred"}, party_size_node()

    # --- Activation: start or resume the flow ------------------------------

    async def end_conversation_action(action: dict) -> None:
        await worker.end(reason=action.get("reason"))

    flow_manager.register_action("end_conversation", end_conversation_action)

    initialized = {"done": False}

    @worker.event_handler("on_activated")
    async def on_activated(worker, args):
        if not initialized["done"]:
            initialized["done"] = True
            await flow_manager.initialize(party_size_node())
        else:
            # Control was handed back to us; restart the reservation flow.
            await flow_manager.set_node_from_config(party_size_node())

    return worker


# =============================================================================
# Router worker: free-form LLM that routes to the reservation flow.
# =============================================================================


class RouterWorker(LLMWorker):
    """Open-ended assistant that transfers to the reservation worker."""

    @tool(cancel_on_interruption=False)
    async def transfer_to_reservation(self, params: FunctionCallParams, reason: str):
        """Transfer the user to the reservation assistant.

        Call this as soon as the user wants to book, change, or ask about
        making a table reservation.

        Args:
            reason (str): Why the user is being transferred.
        """
        logger.info(f"Worker '{self.name}': transferring to '{RESERVATION_NAME}' ({reason})")
        await self.activate_worker(
            RESERVATION_NAME,
            args=LLMWorkerActivationArgs(
                messages=[{"role": "developer", "content": reason}],
            ),
            deactivate_self=True,
            result_callback=params.result_callback,
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"Worker '{self.name}': ending conversation ({reason})")
        await self.end(
            reason=reason,
            messages=[{"role": "developer", "content": reason}],
            result_callback=params.result_callback,
        )


def build_router(llm: Any) -> RouterWorker:
    """Build the free-form router worker."""
    return RouterWorker(ROUTER_NAME, llm=llm, bridged=())


# =============================================================================
# Bot setup.
# =============================================================================


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Wire up the transport, the shared context, and the three workers."""
    logger.info("Starting multi-worker handoff bot")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Jacqueline
        ),
    )

    # The shared conversation context lives in the main worker. Both the router
    # and the reservation worker speak into this same history via the bus.
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # The main bridge sends user-side context to the active worker and brings
    # its generated frames back so the TTS can speak them.
    bridge = BusBridgeProcessor(
        bus=runner.bus,
        worker_name=MAIN_NAME,
        name=f"{MAIN_NAME}::BusBridge",
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            bridge,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(
        pipeline,
        name=MAIN_NAME,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Each LLM worker gets its own LLM service instance.
    router = build_router(create_llm())
    reservation = build_reservation_worker(
        llm=create_llm(),
        context_aggregator=aggregators,
        reservation_system=MockReservationSystem(),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Start the conversation with the router.
        await worker.activate_worker(
            ROUTER_NAME,
            args=LLMWorkerActivationArgs(
                messages=[
                    {
                        "role": "developer",
                        "content": (
                            "You are a friendly assistant for La Maison restaurant. Greet the "
                            "user, mention you can answer questions or book a table, and ask how "
                            "you can help. When the user wants to make a reservation, call the "
                            "transfer_to_reservation tool. If the user says goodbye, call the "
                            "end_conversation tool. Do not mention transferring, just do it "
                            "seamlessly. Keep responses brief, this is a voice conversation."
                        ),
                    }
                ],
            ),
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.add_workers(router, reservation, worker)

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
