#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Talk to the Zoom Voice Agent, with a Decisions delegation and a transcript trigger.

A returns assistant hands the return decision to the Decisions API through the
session's ``assess_return`` delegation. The bot applies its own rules to the
answers, adds the outcome to the conversation and asks for the next reply. A
trigger checks every caller turn for frustration, apart from the conversation.
Set ``ZOOM_AI_API_KEY`` to a Zoom AI Services API key.

Requirements:
    uv sync --extra zoom-ai --extra daily --extra livekit --extra runner \
        --extra websocket --extra webrtc

Usage:
    uv run python examples/realtime/realtime-zoom-ai-decisions.py -t webrtc

Try: "I'd like to return the mug from order 1001. It arrived cracked." Then
choose a refund or a replacement.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMMessagesAppendFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.zoom_ai.realtime.events import (
    DecisionEvent,
    DecisionQuestion,
    Decisions,
    DecisionTrigger,
    Delegation,
    Speak,
)
from pipecat.services.zoom_ai.realtime.llm import ZoomAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

ORDERS = {"1001": "ceramic coffee mug", "1004": "gift card (final sale)"}

DECISIONS = Decisions(
    delegation=Delegation(
        name="assess_return",
        description="Decide what happens with the customer's return request.",
        questions=[
            DecisionQuestion(
                type="choice",
                name="reason",
                instructions="Why does the customer want to return the item?",
                choices=[
                    {"value": "damaged"},
                    {"value": "changed_mind"},
                    {"value": "unclear", "description": "Not stated yet."},
                ],
            ),
            DecisionQuestion(
                type="choice",
                name="resolution",
                instructions="Which resolution has the customer asked for, if any?",
                choices=[{"value": "refund"}, {"value": "replacement"}, {"value": "not_chosen"}],
            ),
        ],
    ),
    triggers=[
        DecisionTrigger(
            id="caller_mood",
            event_type="conversation.item.input_audio_transcription.completed",
            questions=[
                DecisionQuestion(
                    type="predicate", name="frustrated", instructions="Is the customer frustrated?"
                )
            ],
        )
    ],
)

transport_params = {
    "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "livekit": lambda: LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")
    order: dict[str, str] = {}

    async def lookup_order(params: FunctionCallParams, order_number: str):
        """Look up an order.

        Args:
            order_number: The order number, digits only, e.g. "1001".
        """
        item = ORDERS.get(order_number)
        if item:
            order["item"] = item
        await params.result_callback({"found": bool(item), "item": item})

    llm = ZoomAIRealtimeLLMService(
        api_key=os.environ["ZOOM_AI_API_KEY"],
        settings=ZoomAIRealtimeLLMService.Settings(
            system_instruction="""You are a returns assistant on a phone call. Keep replies short.
Call lookup_order for the customer's order number. Once you know what's wrong, delegate to
assess_return; delegate again when they choose a refund or a replacement. Never decide a return
yourself: after a delegation, a message tells you the outcome, and you tell the customer.""",
            speak=Speak(voice="jenna"),
            decisions=DECISIONS,
        ),
    )

    context = LLMContext(
        [{"role": "developer", "content": "Greet the customer and ask how you can help."}],
        [lookup_order],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [transport.input(), user_aggregator, llm, transport.output(), assistant_aggregator]
    )
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True))

    def outcome(decision: DecisionEvent) -> str:
        item = order.get("item")
        if not item:
            return "Ask for the order number first."
        if "final sale" in item:
            return f"The {item} can't be returned."
        reason = decision.answer("reason")
        if not reason or reason.type != "choice" or reason.choice == "unclear":
            return "Ask what's wrong with the item."
        resolution = decision.answer("resolution")
        chosen = resolution.choice if resolution and resolution.type == "choice" else "not_chosen"
        if chosen == "not_chosen":
            return f"The return of the {item} is approved. Ask: refund or replacement?"
        return f"A {chosen} for the {item} is confirmed. A confirmation email is on its way."

    @llm.event_handler("on_decision_completed")
    async def on_decision_completed(service, decision: DecisionEvent):
        if decision.trigger_id:
            frustrated = decision.answer("frustrated")
            logger.info(f"caller frustrated: {frustrated.probability if frustrated else None}")
            return
        # The delegated turn has ended; tell the agent what happened and ask for its reply.
        await worker.queue_frames(
            [
                LLMMessagesAppendFrame(
                    [{"role": "developer", "content": outcome(decision)}], run_llm=True
                )
            ]
        )

    @llm.event_handler("on_decision_failed")
    async def on_decision_failed(service, decision: DecisionEvent):
        logger.warning(f"decision failed: {decision.error}")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
