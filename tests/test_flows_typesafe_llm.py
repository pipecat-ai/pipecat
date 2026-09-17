#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import functools
import json
import unittest
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import ChoiceAnswer, NoulAnswer, SystemOneResponse, Usage

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.flows.manager import FlowManager, NodeConfig
from pipecat.flows.typesafe_llm import (
    NO_TOOL,
    STATED,
    TOOL_QUESTION_ID,
    NodeLines,
    ToolLines,
    TypeSafeFlowsLLMService,
)
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
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
    "choose_pizza": NodeLines(
        say="What size and kind of pizza would you like?",
        reprompt="What size and what kind of pizza?",
    ),
    "confirm": NodeLines(say="So that's {{ order.summary }}. Sound good?"),
}

TOOL_LINES = {
    "select_pizza_order": ToolLines(
        description="The caller gives pizza details",
        examples=["Pizza please"],
        options={"size": ["small", "medium", "large"]},
        option_examples={"size": {"large": ["big", "the biggest"]}},
        ask={
            "size": "What size?",
            "pizza_type": "A {{ args.size }}, got it. What kind?",
        },
    ),
    "get_prices": ToolLines(result="Pizzas are {{ result.small }} and up."),
}


def response(answers: dict[str, str | float | tuple[str, float]]) -> SystemOneResponse:
    """A response answering each question.

    A string is a fully confident choice, a ``(choice, confidence)`` pair a
    choice at that confidence, and a float a Noul probability.
    """
    built: dict[str, Any] = {}
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


def stated(tool: str, *arguments: str) -> dict[str, float]:
    """Presence answers: the named arguments given, every other option-bearing one not."""
    return {
        f"{tool}.{name}.{STATED}": (1.0 if name in arguments else 0.0)
        for name in ("size", "pizza_type")
    }


def user_turn(*messages: dict[str, Any]) -> LLMContext:
    context = LLMContext(
        messages=[{"role": "assistant", "content": "What size and kind of pizza?"}, *messages]
    )
    context.set_tools([SELECT_PIZZA, GET_PRICES])
    return context


def texts(frames) -> list[str]:
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


class TestTypeSafeFlowsLLMService(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        assistant = MagicMock()
        type(assistant).has_function_calls_in_progress = PropertyMock(return_value=False)
        aggregator = MagicMock()
        aggregator.user = MagicMock(return_value=MagicMock())
        aggregator.assistant = MagicMock(return_value=assistant)
        self.client = FakeTypeSafeClient()
        self.llm = TypeSafeFlowsLLMService(
            judge=TypeSafeJudge(client=self.client),
            nodes=NODE_LINES,
            tools=TOOL_LINES,
            warm_up=False,
        )
        self.flow_manager = FlowManager(
            worker=make_mock_worker(), llm=self.llm, context_aggregator=aggregator
        )
        self.llm.flow_manager = self.flow_manager
        await self.flow_manager.initialize(
            NodeConfig(name="choose_pizza", task_messages=[], respond_immediately=False)
        )
        self.calls: list[dict[str, Any]] = []

        async def record(params):
            self.calls.append(params.arguments)
            await params.result_callback({"ok": True})

        self.llm.register_function("select_pizza_order", record)

    async def test_entering_a_node_speaks_its_line_as_llm_text(self):
        context = LLMContext(messages=[{"role": "developer", "content": "Get the order."}])

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

        self.assertEqual(texts(down), ["What size and kind of pizza would you like?"])
        self.assertEqual(self.client.requests, [])

    async def test_a_complete_turn_runs_the_chosen_tool_with_its_arguments(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "pepperoni",
                **stated("select_pizza_order", "size", "pizza_type"),
            }
        )
        context = user_turn({"role": "user", "content": "A large pepperoni please"})

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
        )

        self.assertTrue(any(isinstance(f, FunctionCallInProgressFrame) for f in down))
        self.assertEqual(self.calls, [{"size": "large", "pizza_type": "pepperoni"}])
        state, questions = self.client.requests[0]
        self.assertEqual(state["user_reply"], "A large pepperoni please")
        tool_question = questions[TOOL_QUESTION_ID]
        self.assertIn("select_pizza_order", tool_question.criteria)
        self.assertIn("get_prices", tool_question.criteria)
        self.assertIn(NO_TOOL, tool_question.criteria)
        # Presence and value are separate questions; the value choice lists only values.
        self.assertIn(f"select_pizza_order.pizza_type.{STATED}", questions)
        self.assertEqual(
            set(questions["select_pizza_order.pizza_type"].criteria), {"cheese", "pepperoni"}
        )
        # Examples make a criterion an object the model compares on.
        self.assertEqual(
            questions["select_pizza_order.size"].criteria["large"],
            {"what": "The caller says large", "examples": ["big", "the biggest"]},
        )
        self.assertEqual(tool_question.criteria["select_pizza_order"]["examples"], ["Pizza please"])

    async def test_a_weak_argument_value_is_asked_for_rather_than_guessed(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": ("large", 0.4),
                "select_pizza_order.pizza_type": "pepperoni",
                **stated("select_pizza_order", "size", "pizza_type"),
            }
        )
        context = user_turn({"role": "user", "content": "A regular pepperoni"})

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
        )

        self.assertEqual(texts(down), ["What size?"])
        self.assertEqual(self.calls, [])

    async def test_a_tool_threshold_overrides_the_service_threshold(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: ("get_prices", 0.7),
                **stated("select_pizza_order"),
            }
        )
        self.llm._tools["get_prices"] = ToolLines(
            result="Pizzas are {{ result.small }} and up.", confidence_threshold=0.9
        )
        context = user_turn({"role": "user", "content": "Um, what's the, uh, cost"})

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
        )

        self.assertEqual(texts(down), ["What size and what kind of pizza?"])
        self.assertEqual(self.calls, [])

    async def test_a_partial_turn_asks_for_the_missing_argument_and_remembers_the_rest(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "cheese",
                **stated("select_pizza_order", "size"),
            }
        )
        first = user_turn({"role": "user", "content": "Large."})

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=first), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )
        self.assertEqual(texts(down), ["A large, got it. What kind?"])
        self.assertEqual(self.calls, [])

        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "small",
                "select_pizza_order.pizza_type": "cheese",
                **stated("select_pizza_order", "pizza_type"),
            }
        )
        second = user_turn(
            {"role": "user", "content": "Large."},
            {"role": "assistant", "content": "A large, got it. What kind?"},
            {"role": "user", "content": "Cheese."},
        )

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=second), SleepFrame()],
        )

        self.assertTrue(any(isinstance(f, FunctionCallInProgressFrame) for f in down))
        self.assertEqual(self.calls, [{"size": "large", "pizza_type": "cheese"}])

    async def test_no_matching_tool_speaks_the_reprompt(self):
        self.client.response = response({TOOL_QUESTION_ID: NO_TOOL, **stated("select_pizza_order")})
        context = user_turn({"role": "user", "content": "Hang on a second"})

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

        self.assertEqual(texts(down), ["What size and what kind of pizza?"])
        self.assertEqual(self.calls, [])

    async def test_a_tool_result_speaks_the_tool_result_line(self):
        context = LLMContext(
            messages=[
                {"role": "user", "content": "How much?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "get_prices", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": json.dumps({"small": 10})},
            ]
        )

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

        self.assertEqual(texts(down), ["Pizzas are 10 and up."])
        self.assertEqual(self.client.requests, [])

    async def test_lines_render_flow_state(self):
        self.flow_manager.state["order"] = {"summary": "one large pepperoni pizza"}
        await self.flow_manager.set_node_from_config(
            NodeConfig(name="confirm", task_messages=[], respond_immediately=False)
        )
        context = LLMContext(messages=[{"role": "developer", "content": "Read it back."}])

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
            expected_down_frames=[
                LLMServiceMetadataFrame,
                LLMFullResponseStartFrame,
                LLMTextFrame,
                LLMFullResponseEndFrame,
            ],
        )

        self.assertEqual(texts(down), ["So that's one large pepperoni pizza. Sound good?"])

    async def test_a_judged_turn_reports_ttfb_ttfat_processing_and_usage(self):
        self.client.response = response(
            {
                TOOL_QUESTION_ID: "select_pizza_order",
                "select_pizza_order.size": "large",
                "select_pizza_order.pizza_type": "pepperoni",
            }
        )
        context = user_turn({"role": "user", "content": "A large pepperoni please"})

        down, _ = await run_test(
            self.llm,
            frames_to_send=[LLMContextFrame(context=context), SleepFrame()],
            pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        reported = {
            type(data) for frame in down if isinstance(frame, MetricsFrame) for data in frame.data
        }
        self.assertLessEqual(
            {TTFBMetricsData, TTFATMetricsData, ProcessingMetricsData, LLMUsageMetricsData},
            reported,
        )
        usage = next(
            data
            for frame in down
            if isinstance(frame, MetricsFrame)
            for data in frame.data
            if isinstance(data, LLMUsageMetricsData)
        )
        self.assertEqual(usage.value.prompt_tokens, 10)
        self.assertEqual(usage.value.completion_tokens, 2)
        self.assertEqual(usage.model, "jev-latest")
