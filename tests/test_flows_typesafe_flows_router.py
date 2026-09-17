#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import functools
import unittest
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import ChoiceAnswer, NoulAnswer, SystemOneResponse, TypeSafeError, Usage

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.flows.manager import FlowManager, NodeConfig
from pipecat.flows.typesafe_flows_router import MORE_QUESTION_ID, TypeSafeFlowsRouter
from pipecat.flows.typesafe_llm import NO_TOOL, STATED, TOOL_QUESTION_ID, NodeLines, ToolLines
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.typesafe import TypeSafeJudge
from pipecat.tests.utils import SleepFrame
from pipecat.tests.utils import run_test as _run_test
from tests.flows_test_helpers import make_mock_worker
from tests.typesafe_test_helpers import FakeTypeSafeClient

# The first pipeline in a process can take over a second to start.
run_test = functools.partial(_run_test, start_timeout=5.0)

SELECT_PIZZA = FunctionSchema(
    name="select_pizza_order",
    description="Record the pizza order details.",
    properties={
        "size": {"type": "string", "description": "Size of the pizza."},
        "pizza_type": {"type": "string", "enum": ["cheese", "pepperoni"]},
    },
    required=["size", "pizza_type"],
)

GET_PRICES = FunctionSchema(
    name="get_prices", description="Provide the menu prices.", properties={}, required=[]
)

NODE_LINES = {
    "choose_pizza": NodeLines(say="What size and kind of pizza would you like?"),
}

TOOL_LINES = {
    "select_pizza_order": ToolLines(
        description="The caller gives pizza details",
        options={"size": ["small", "medium", "large"]},
        ask={"size": "What size?", "pizza_type": "A {{ args.size }}, got it. What kind?"},
    ),
}


class RecordingLLM(LLMService):
    """An LLM that keeps every context frame it is handed and never speaks."""

    def __init__(self):
        super().__init__(
            settings=LLMSettings(
                model="stub",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=None,
                user_turn_completion_config=None,
            )
        )
        self.contexts: list[LLMContext] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.contexts.append(frame.context)
        else:
            await self.push_frame(frame, direction)


def response(answers: dict[str, str | float | tuple[str, float]]) -> SystemOneResponse:
    """A response answering each question.

    A string is a fully confident choice, a ``(choice, confidence)`` pair a
    choice at that confidence, and a float a Noul probability. The "said
    more" Noul defaults to 0.
    """
    built: dict[str, Any] = {MORE_QUESTION_ID: NoulAnswer(noul=0.0)}
    for question_id, answer in answers.items():
        if isinstance(answer, float):
            built[question_id] = NoulAnswer(noul=answer)
        else:
            choice, confidence = answer if isinstance(answer, tuple) else (answer, 1.0)
            built[question_id] = ChoiceAnswer(
                choice=choice, confidence=confidence, probabilities={choice: confidence}
            )
    return SystemOneResponse(
        model="jev-test", usage=Usage(input_tokens=10, output_tokens=2), answers=built
    )


def stated(*arguments: str) -> dict[str, float]:
    """Presence answers: the named arguments given, every other option-bearing one not."""
    return {
        f"select_pizza_order.{name}.{STATED}": (1.0 if name in arguments else 0.0)
        for name in ("size", "pizza_type")
    }


def node_entry() -> LLMContext:
    return LLMContext(messages=[{"role": "developer", "content": "Get the order."}])


def texts(frames) -> list[str]:
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


class TestTypeSafeFlowsRouter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        assistant = MagicMock()
        type(assistant).has_function_calls_in_progress = PropertyMock(return_value=False)
        aggregator = MagicMock()
        aggregator.user = MagicMock(return_value=MagicMock())
        aggregator.assistant = MagicMock(return_value=assistant)
        self.client = FakeTypeSafeClient()
        self.llm = RecordingLLM()
        self.router = TypeSafeFlowsRouter(
            judge=TypeSafeJudge(client=self.client),
            llm=self.llm,
            nodes=NODE_LINES,
            tools=TOOL_LINES,
            warm_up=False,
        )
        self.pipeline = Pipeline([self.router, self.llm])
        self.flow_manager = FlowManager(
            worker=make_mock_worker(), llm=self.llm, context_aggregator=aggregator
        )
        self.router.flow_manager = self.flow_manager
        await self.flow_manager.initialize(
            NodeConfig(name="choose_pizza", task_messages=[], respond_immediately=False)
        )
        self.calls: list[dict[str, Any]] = []

    async def record(self, params):
        self.calls.append(params.arguments)
        await params.result_callback({"ok": True})

    def user_turn(self, *messages: dict[str, Any]) -> LLMContext:
        """A caller's turn in a node whose tools carry their handlers, as Flows advertises them."""
        select_pizza = FunctionSchema(
            name=SELECT_PIZZA.name,
            description=SELECT_PIZZA.description,
            properties=SELECT_PIZZA.properties,
            required=SELECT_PIZZA.required,
            handler=self.record,
        )
        context = LLMContext(
            messages=[{"role": "assistant", "content": "What size and kind of pizza?"}, *messages]
        )
        context.set_tools([select_pizza, GET_PRICES])
        return context

    async def send(self, *frames: Frame, **kwargs):
        return await run_test(self.pipeline, frames_to_send=[*frames, SleepFrame()], **kwargs)

    async def test_the_first_node_entry_speaks_its_line(self):
        down, _ = await self.send(LLMContextFrame(context=node_entry()))

        self.assertEqual(texts(down), ["What size and kind of pizza would you like?"])
        self.assertEqual(self.llm.contexts, [])
        self.assertEqual(self.client.requests, [])

    async def test_a_sure_turn_runs_the_tool_and_speaks_the_next_node(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "pepperoni",
                **stated("size", "pizza_type"),
            }
        )
        turn = self.user_turn({"role": "user", "content": "A large pepperoni please"})

        down, _ = await self.send(
            LLMContextFrame(context=node_entry()),
            SleepFrame(),
            LLMContextFrame(context=turn),
            SleepFrame(),
            LLMContextFrame(context=node_entry()),
        )

        self.assertTrue(any(isinstance(f, FunctionCallInProgressFrame) for f in down))
        self.assertEqual(self.calls, [{"size": "large", "pizza_type": "pepperoni"}])
        # Both entries were spoken from the line; the LLM never ran, so the
        # tool's handler reached it only through the router's sync.
        self.assertEqual(texts(down), ["What size and kind of pizza would you like?"] * 2)
        self.assertEqual(self.llm.contexts, [])
        _, questions = self.client.requests[0]
        self.assertIn(MORE_QUESTION_ID, questions)
        self.assertIn(TOOL_QUESTION_ID, questions)

    async def test_a_less_sure_value_runs_the_tool_but_the_llm_speaks(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": ("large", 0.7),
                "select_pizza_order.pizza_type": "pepperoni",
                **stated("size", "pizza_type"),
            }
        )
        turn = self.user_turn({"role": "user", "content": "A biggish pepperoni"})

        down, _ = await self.send(
            LLMContextFrame(context=turn), SleepFrame(), LLMContextFrame(context=node_entry())
        )

        self.assertEqual(self.calls, [{"size": "large", "pizza_type": "pepperoni"}])
        self.assertEqual(texts(down), [])
        self.assertEqual(len(self.llm.contexts), 1)

    async def test_a_caller_who_also_asks_something_gets_the_llm(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "pepperoni",
                **stated("size", "pizza_type"),
                MORE_QUESTION_ID: 0.8,
            }
        )
        turn = self.user_turn({"role": "user", "content": "A large pepperoni, is it gluten free?"})

        down, _ = await self.send(
            LLMContextFrame(context=turn), SleepFrame(), LLMContextFrame(context=node_entry())
        )

        self.assertEqual(self.calls, [{"size": "large", "pizza_type": "pepperoni"}])
        self.assertEqual(texts(down), [])
        self.assertEqual(len(self.llm.contexts), 1)

    async def test_a_plainly_missing_argument_gets_the_written_ask_and_is_remembered(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "cheese",
                **stated("size"),
            }
        )

        down, _ = await self.send(
            LLMContextFrame(context=self.user_turn({"role": "user", "content": "Large."})),
            expected_down_frames=[
                LLMServiceMetadataFrame,
                RTVIServerMessageFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )
        self.assertEqual(texts(down), ["A large, got it. What kind?"])
        message = next(f for f in down if isinstance(f, RTVIServerMessageFrame))
        self.assertEqual(message.data["type"], "typesafe-router")
        self.assertEqual(message.data["tier"], "canned")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.llm.contexts, [])

        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "small",
                "select_pizza_order.pizza_type": "cheese",
                **stated("pizza_type"),
            }
        )
        second = self.user_turn(
            {"role": "user", "content": "Large."},
            {"role": "assistant", "content": "A large, got it. What kind?"},
            {"role": "user", "content": "Cheese."},
        )
        await self.send(LLMContextFrame(context=second))

        self.assertEqual(self.calls, [{"size": "large", "pizza_type": "cheese"}])
        # The router spoke last, so the judgment saw its whole line.
        state, _ = self.client.requests[1]
        self.assertEqual(state["bot_message"], "A large, got it. What kind?")

    async def test_a_less_sure_missing_argument_is_asked_for_by_the_llm(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": ("large", 0.7),
                "select_pizza_order.pizza_type": "cheese",
                **stated("size"),
            }
        )
        turn = self.user_turn({"role": "user", "content": "A big one."})

        down, _ = await self.send(LLMContextFrame(context=turn))

        self.assertEqual(texts(down), [])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.llm.contexts, [turn])
        self.assertEqual(
            self.router._remembered[("choose_pizza", "select_pizza_order")], {"size": "large"}
        )

    async def test_no_matching_tool_passes_the_turn_to_the_llm(self):
        self.client.response = response({TOOL_QUESTION_ID: NO_TOOL, **stated()})
        turn = self.user_turn({"role": "user", "content": "What do you recommend?"})

        down, _ = await self.send(LLMContextFrame(context=turn))

        self.assertEqual(texts(down), [])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.llm.contexts, [turn])

    async def test_a_tool_threshold_overrides_the_router_thresholds(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: ("select_pizza_order", 0.95),
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "pepperoni",
                **stated("size", "pizza_type"),
            }
        )
        self.router._tools["select_pizza_order"] = ToolLines(
            options={"size": ["small", "large"]}, canned_threshold=0.99
        )
        turn = self.user_turn({"role": "user", "content": "A large pepperoni"})

        down, _ = await self.send(
            LLMContextFrame(context=turn), SleepFrame(), LLMContextFrame(context=node_entry())
        )

        self.assertEqual(self.calls, [{"size": "large", "pizza_type": "pepperoni"}])
        self.assertEqual(texts(down), [])
        self.assertEqual(len(self.llm.contexts), 1)

    async def test_a_failed_judgment_passes_the_turn_to_the_llm(self):
        self.client.error = TypeSafeError("boom")
        turn = self.user_turn({"role": "user", "content": "A large pepperoni"})

        _, up = await self.send(LLMContextFrame(context=turn))

        self.assertTrue(any(isinstance(f, ErrorFrame) for f in up))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.llm.contexts, [turn])

    async def test_an_interruption_cancels_the_judgment_and_drops_the_turn(self):
        self.client.delay = 0.5
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "pepperoni",
                **stated("size", "pizza_type"),
            }
        )
        turn = self.user_turn({"role": "user", "content": "A large pepperoni"})

        down, _ = await self.send(
            LLMContextFrame(context=turn), SleepFrame(0.1), InterruptionFrame(), SleepFrame(0.6)
        )

        self.assertEqual(texts(down), [])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.llm.contexts, [])
