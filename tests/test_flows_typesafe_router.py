#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import functools
import unittest
from unittest.mock import MagicMock, PropertyMock

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import Noul

from pipecat.flows.manager import FlowManager, NodeConfig
from pipecat.flows.typesafe_router import STATE_KEY, FlowsTypeSafeRouter, TypeSafeRoute
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.typesafe import TypeSafeJudge
from pipecat.tests.utils import SleepFrame
from pipecat.tests.utils import run_test as _run_test
from tests.flows_test_helpers import make_mock_worker
from tests.typesafe_test_helpers import FakeTypeSafeClient, choice_response

# The first pipeline in a process can take over a second to start.
run_test = functools.partial(_run_test, start_timeout=5.0)


def verify_node() -> NodeConfig:
    return NodeConfig(
        name="verify",
        task_messages=[{"role": "developer", "content": "Ask again."}],
        respond_immediately=False,
    )


def confirmed_node() -> NodeConfig:
    return NodeConfig(name="confirmed", task_messages=[], respond_immediately=False)


def declined_node() -> NodeConfig:
    return NodeConfig(name="declined", task_messages=[], respond_immediately=False)


def make_context() -> LLMContext:
    return LLMContext(
        messages=[
            {"role": "assistant", "content": "Are you over 18?"},
            {"role": "user", "content": "Yes"},
        ]
    )


class TestFlowsTypeSafeRouter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        assistant = MagicMock()
        type(assistant).has_function_calls_in_progress = PropertyMock(return_value=False)
        aggregator = MagicMock()
        aggregator.user = MagicMock(return_value=MagicMock())
        aggregator.assistant = MagicMock(return_value=assistant)
        self.flow_manager = FlowManager(
            worker=make_mock_worker(),
            llm=OpenAILLMService(api_key="test-key"),
            context_aggregator=aggregator,
        )
        await self.flow_manager.initialize(verify_node())

    def make_router(self, client, **kwargs) -> FlowsTypeSafeRouter:
        router = FlowsTypeSafeRouter(
            judge=TypeSafeJudge(client=client),
            instructions="How did the user answer `bot_message`? Judge `user_reply`.",
            routes={
                "yes": TypeSafeRoute("Confirms over 18", confirmed_node),
                "no": TypeSafeRoute("Says not over 18", declined_node()),
            },
            active_nodes={"verify"},
            warm_up=False,
            **kwargs,
        )
        router.flow_manager = self.flow_manager
        return router

    async def test_confident_choice_moves_to_the_route_target(self):
        router = self.make_router(FakeTypeSafeClient(choice_response("yes", 0.95)))

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[],
        )

        self.assertEqual(self.flow_manager.current_node, "confirmed")
        self.assertEqual(self.flow_manager.state[STATE_KEY]["choices"]["route"]["choice"], "yes")

    async def test_node_config_target(self):
        router = self.make_router(FakeTypeSafeClient(choice_response("no", 0.95)))

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[],
        )

        self.assertEqual(self.flow_manager.current_node, "declined")

    async def test_fallback_leaves_the_turn_to_the_llm(self):
        router = self.make_router(FakeTypeSafeClient(choice_response("other", 1.0)))

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(self.flow_manager.current_node, "verify")
        self.assertNotIn(STATE_KEY, self.flow_manager.state)

    async def test_inactive_node_passes_through(self):
        await self.flow_manager.set_node_from_config(
            NodeConfig(name="chat", task_messages=[], respond_immediately=False)
        )
        client = FakeTypeSafeClient(choice_response("yes", 1.0))
        router = self.make_router(client)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context())],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(client.requests, [])
        self.assertEqual(self.flow_manager.current_node, "chat")

    async def test_async_target_can_yield_to_the_llm(self):
        async def guarded(flow_manager, result):
            if result.nouls["asked"].probability > 0.5:
                return None
            return confirmed_node()

        client = FakeTypeSafeClient(choice_response("yes", 1.0, nouls={"asked": 0.9}))
        router = FlowsTypeSafeRouter(
            judge=TypeSafeJudge(client=client),
            instructions="q",
            routes={"yes": TypeSafeRoute("Confirms", guarded)},
            active_nodes={"verify"},
            extra_questions={"asked": Noul(instructions="Did they ask something?")},
            warm_up=False,
        )
        router.flow_manager = self.flow_manager

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(self.flow_manager.current_node, "verify")
        self.assertIn(STATE_KEY, self.flow_manager.state)
