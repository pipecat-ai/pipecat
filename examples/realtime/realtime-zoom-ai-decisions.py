#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A returns assistant on the Zoom Voice Agent, deciding with a classifier.

The Voice Agent (Zoom's ``gemma-4`` model) talks with the caller and calls
three tools:

- ``lookup_order``: finds the order in a small in-memory store.
- ``search_policy``: searches a local knowledge base, the ``##`` sections of
  ``examples/assets/zoom-cafe-shop-returns.md``, ranked with BM25.
- ``assess_return``: asks OpenAI's Decisions API why the caller wants to
  return the item, then applies the return policy in plain code.

The voice model gathers the facts, the classifier judges them, and code
decides, so the outcome follows the policy whatever the model says.

Set ``ZOOM_AI_API_KEY`` to a Zoom AI Services API key from Zoom Platform Studio,
and ``OPENAI_API_KEY`` for the classifier.
Try order 1001 (a cracked mug) or 1002 (coffee beans you don't want any more).

Requirements:
    uv sync --extra zoom-ai --extra openai --extra daily --extra livekit \
        --extra runner --extra websocket --extra webrtc

Usage:
    uv run python examples/realtime/realtime-zoom-ai-decisions.py -t webrtc
"""

import math
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from pipecat.classifiers.base_classifier import (
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ClassifierQuestion,
    ScoreQuestion,
    ScoreResult,
)
from pipecat.classifiers.openai.decisions.classifier import OpenAIDecisionsClassifier
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.zoom_ai.realtime.events import Speak
from pipecat.services.zoom_ai.realtime.llm import ZoomAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

KNOWLEDGE_BASE = Path(__file__).parent.parent / "assets" / "zoom-cafe-shop-returns.md"

RETURN_WINDOW_DAYS = 30

ORDERS = {
    "1001": {
        "item": "ceramic coffee mug",
        "price_usd": 14.0,
        "days_since_delivery": 6,
        "food": False,
    },
    "1002": {
        "item": "house blend coffee beans, 1 lb",
        "price_usd": 18.0,
        "days_since_delivery": 12,
        "food": True,
    },
    "1003": {
        "item": "milk frother",
        "price_usd": 39.0,
        "days_since_delivery": 45,
        "food": False,
    },
}

INSTRUCTIONS = """You are the returns assistant for Zoom Cafe Shop, in a voice conversation.

Help the caller return an item. First ask for their order number and call lookup_order. Then ask
what's wrong with the item, in their own words, and call assess_return with their description. Tell
them the outcome the tool returns, and follow its guidance. For questions about the policy, refunds
or shipping, call search_policy and answer only from what it returns.

Call tools right away, without saying that you will. Keep each reply to one or two short sentences.
Never promise a refund, a replacement or an exception that a tool didn't give."""


class KnowledgeBase:
    """The ``##`` sections of a markdown file, ranked against a query with BM25."""

    def __init__(self, path: Path, *, k1: float = 1.5, b: float = 0.75):
        text = path.read_text()
        self._passages = [
            (title.strip(), body.strip())
            for title, body in re.findall(r"^## ([^\n]+)\n(.*?)(?=^## |\Z)", text, re.M | re.S)
        ]
        self._docs = [Counter(self._tokens(f"{t} {b}")) for t, b in self._passages]
        self._avg_len = sum(sum(d.values()) for d in self._docs) / len(self._docs)
        df = Counter(term for doc in self._docs for term in doc)
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._k1, self._b = k1, b

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # Plurals fold into the singular ("refunds" matches "refund").
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [
            w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words
        ]

    def search(self, query: str, k: int = 3) -> list[dict[str, str]]:
        """The ``k`` sections that best match ``query``, best first."""
        terms = self._tokens(query)
        scores = []
        for i, doc in enumerate(self._docs):
            length = sum(doc.values())
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self._k1 * (1 - self._b + self._b * length / self._avg_len)
                    score += self._idf[term] * tf * (self._k1 + 1) / norm
            scores.append((score, i))
        ranked = [i for score, i in sorted(scores, reverse=True) if score > 0][:k]
        return [{"title": self._passages[i][0], "text": self._passages[i][1]} for i in ranked]


RETURN_QUESTIONS: dict[str, ClassifierQuestion] = {
    "reason": ChoiceQuestion(
        instructions="Why does the customer want to return the item?",
        options={
            "damaged": "it arrived broken, cracked, chipped, dented, or doesn't work",
            "wrong_item": "it's a different item, size or color than ordered, or not as described",
            "changed_mind": "nothing is wrong with it; they no longer want it",
            "other": None,
        },
    ),
    "frustration": ScoreQuestion(
        instructions="How upset does the customer sound?",
        levels=["calm", "annoyed", "angry"],
    ),
}


def decide(order: dict, reason: ChoiceResult, frustration: ScoreResult) -> dict:
    """Apply the return policy to the classifier's reading of the customer."""
    tone = "Acknowledge their frustration first. " if frustration.score >= 1.0 else ""
    days = order["days_since_delivery"]
    if reason.confidence < 0.6 or reason.choice == "other":
        return {
            "outcome": "specialist_review",
            "guidance": tone + "Say a returns specialist will email them within one business day.",
        }
    if reason.choice == "damaged":
        if days > RETURN_WINDOW_DAYS:
            return {
                "outcome": "warranty",
                "guidance": tone + "It's past the 30-day window; offer a warranty claim by email.",
            }
        return {"outcome": "refund", "guidance": tone + "Confirm a refund right away."}
    if reason.choice == "wrong_item":
        return {
            "outcome": "exchange_or_refund",
            "guidance": tone + "Offer the right item or a full refund, with a free return label.",
        }
    # Changed their mind.
    if order["food"]:
        return {
            "outcome": "not_returnable",
            "guidance": tone + "Explain that opened food can't be returned for hygiene reasons.",
        }
    if days > RETURN_WINDOW_DAYS:
        return {
            "outcome": "outside_window",
            "guidance": tone + "Explain that change-of-mind returns close 30 days after delivery.",
        }
    return {
        "outcome": "return_label",
        "guidance": tone + "Say a prepaid return label is on its way; the refund follows arrival.",
    }


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "livekit": lambda: LiveKitParams(
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
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    knowledge = KnowledgeBase(KNOWLEDGE_BASE)
    classifier = OpenAIDecisionsClassifier(api_key=os.environ["OPENAI_API_KEY"])

    async def lookup_order(params: FunctionCallParams, order_number: str):
        """Look up an order.

        Args:
            order_number: The order number the caller gives, digits only, e.g. "1001".
        """
        order = ORDERS.get(re.sub(r"\D", "", order_number))
        if order is None:
            await params.result_callback({"found": False, "guidance": "Ask them to check it."})
            return
        await params.result_callback({"found": True, **order})

    async def search_policy(params: FunctionCallParams, question: str):
        """Search the store's return policy, refund and shipping information.

        Args:
            question: The caller's question, in a few words.
        """
        passages = knowledge.search(question)
        await params.result_callback(
            {
                "passages": passages,
                "guidance": "Answer from these passages only, in one or two sentences.",
            }
        )

    async def assess_return(params: FunctionCallParams, order_number: str, description: str):
        """Decide what happens to a return.

        Args:
            order_number: The order number, digits only.
            description: What the caller said is wrong with the item, in their words.
        """
        order = ORDERS.get(re.sub(r"\D", "", order_number))
        if order is None:
            await params.result_callback({"guidance": "Look up the order first."})
            return
        state = f"Item: {order['item']}\nCustomer: {description}"
        try:
            answers = await classifier.ask(state, RETURN_QUESTIONS)
        except ClassifierError as e:
            logger.error(f"Classifier unavailable: {e}")
            await params.result_callback(
                {
                    "outcome": "specialist_review",
                    "guidance": "Say a returns specialist will email them within one business day.",
                }
            )
            return
        reason, frustration = answers["reason"], answers["frustration"]
        assert isinstance(reason, ChoiceResult) and isinstance(frustration, ScoreResult)
        logger.info(
            f"Classifier: reason={reason.choice} ({reason.confidence:.2f}), "
            f"frustration={frustration.score:.2f}"
        )
        await params.result_callback({"item": order["item"], **decide(order, reason, frustration)})

    llm = ZoomAIRealtimeLLMService(
        api_key=os.environ["ZOOM_AI_API_KEY"],
        settings=ZoomAIRealtimeLLMService.Settings(
            system_instruction=INSTRUCTIONS,
            speak=Speak(voice="jenna"),
        ),
    )

    context = LLMContext(
        [{"role": "developer", "content": "Greet the caller and ask how you can help."}],
        [lookup_order, search_policy, assess_return],
    )

    # The Voice Agent detects turns and handles barge-in itself, so the
    # aggregators need no local VAD.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            llm,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    try:
        await runner.run()
    finally:
        await classifier.cleanup()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
