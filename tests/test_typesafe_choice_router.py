#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import functools
import unittest

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import Noul, TypeSafeAPITimeoutError

from pipecat.frames.frames import ErrorFrame, InterruptionFrame, LLMContextFrame, MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.typesafe_choice_router import (
    TypeSafeChoiceRouter,
    default_state_builder,
)
from pipecat.services.typesafe import TypeSafeJudge
from pipecat.tests.utils import SleepFrame
from pipecat.tests.utils import run_test as _run_test
from tests.typesafe_test_helpers import FakeTypeSafeClient, choice_response

CRITERIA = {"yes": "The user agrees", "no": "The user declines"}

# The first pipeline in a process can take over a second to start.
run_test = functools.partial(_run_test, start_timeout=5.0)


def make_context() -> LLMContext:
    return LLMContext(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "assistant", "content": "Are you over 18?"},
            {"role": "user", "content": [{"type": "text", "text": "Yeah I am"}]},
        ]
    )


class TestDefaultStateBuilder(unittest.TestCase):
    def test_last_bot_and_user_messages(self):
        state = default_state_builder(make_context())
        self.assertEqual(state, {"bot_message": "Are you over 18?", "user_reply": "Yeah I am"})

    def test_empty_context(self):
        state = default_state_builder(LLMContext())
        self.assertEqual(state, {"bot_message": "", "user_reply": ""})


class TestTypeSafeChoiceRouter(unittest.IsolatedAsyncioTestCase):
    def make_router(self, client, *, handled=True, active=True, **kwargs):
        self.calls = []

        async def on_choice(result, context):
            self.calls.append(result)
            return handled

        return TypeSafeChoiceRouter(
            judge=TypeSafeJudge(client=client),
            instructions="How did the user answer `bot_message`? Judge `user_reply`.",
            criteria=CRITERIA,
            on_choice=on_choice,
            is_active=lambda: active,
            warm_up=False,
            **kwargs,
        )

    async def test_confident_choice_consumes_the_turn(self):
        client = FakeTypeSafeClient(choice_response("yes", 0.95))
        router = self.make_router(client)

        down, _ = await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[],
        )

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0].choices["route"].choice, "yes")
        self.assertFalse(any(isinstance(f, LLMContextFrame) for f in down))
        state, questions = client.requests[0]
        self.assertEqual(state["user_reply"], "Yeah I am")
        self.assertEqual(set(questions["route"].criteria), {"yes", "no", "other"})

    async def test_handler_declining_forwards_the_frame(self):
        client = FakeTypeSafeClient(choice_response("yes", 0.95))
        router = self.make_router(client, handled=False)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(len(self.calls), 1)

    async def test_low_confidence_forwards_the_frame(self):
        client = FakeTypeSafeClient(choice_response("no", 0.4, {"no": 0.6, "other": 0.4}))
        router = self.make_router(client)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(self.calls, [])

    async def test_fallback_option_forwards_the_frame(self):
        client = FakeTypeSafeClient(choice_response("other", 1.0))
        router = self.make_router(client)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(self.calls, [])

    async def test_speculative_frame_is_never_judged(self):
        client = FakeTypeSafeClient(choice_response("yes", 1.0))
        router = self.make_router(client)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context(), speculation=True)],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(client.requests, [])

    async def test_inactive_router_passes_frames_through(self):
        client = FakeTypeSafeClient(choice_response("yes", 1.0))
        router = self.make_router(client, active=False)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context())],
            expected_down_frames=[LLMContextFrame],
        )

        self.assertEqual(client.requests, [])

    async def test_interruption_cancels_the_judgment(self):
        client = FakeTypeSafeClient(choice_response("yes", 1.0), delay=0.5)
        router = self.make_router(client)

        down, _ = await run_test(
            router,
            frames_to_send=[
                LLMContextFrame(context=make_context()),
                SleepFrame(sleep=0.1),
                InterruptionFrame(),
                SleepFrame(sleep=0.6),
            ],
            expected_down_frames=[InterruptionFrame],
        )

        self.assertEqual(self.calls, [])
        self.assertFalse(any(isinstance(f, LLMContextFrame) for f in down))

    async def test_error_falls_through_to_the_llm(self):
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        router = self.make_router(client)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[LLMContextFrame],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(self.calls, [])

    async def test_fallthrough_handler_consumes_unsettled_turns(self):
        fallthroughs = []

        async def on_fallthrough(result, context):
            fallthroughs.append(result)
            return True

        # Fallback option: the handler gets the judgment and consumes the turn.
        client = FakeTypeSafeClient(choice_response("other", 1.0))
        router = self.make_router(client, on_fallthrough=on_fallthrough)
        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[],
        )
        self.assertEqual(self.calls, [])
        self.assertEqual(fallthroughs[0].choices["route"].choice, "other")

        # Failed request: the handler gets None and still consumes the turn.
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        router = self.make_router(client, on_fallthrough=on_fallthrough)
        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[],
            expected_up_frames=[ErrorFrame],
        )
        self.assertIsNone(fallthroughs[1])

    async def test_fallthrough_handler_declining_forwards_the_frame(self):
        async def on_fallthrough(result, context):
            return False

        client = FakeTypeSafeClient(choice_response("other", 1.0))
        router = self.make_router(client, on_fallthrough=on_fallthrough)

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[LLMContextFrame],
        )

    async def test_extra_questions_ride_along(self):
        client = FakeTypeSafeClient(choice_response("yes", 1.0, nouls={"asked": 0.8}))
        router = self.make_router(
            client, extra_questions={"asked": Noul(instructions="Did they ask something?")}
        )

        await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[],
        )

        self.assertAlmostEqual(self.calls[0].nouls["asked"].probability, 0.8)
        _, questions = client.requests[0]
        self.assertEqual(set(questions), {"route", "asked"})

    async def test_reserved_question_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_router(
                FakeTypeSafeClient(), extra_questions={"route": Noul(instructions="?")}
            )

    async def test_metrics_are_reported(self):
        client = FakeTypeSafeClient(choice_response("yes", 1.0))
        router = self.make_router(client)

        # The worker sends one initial zeroed MetricsFrame with no model name;
        # the router adds processing time and token usage tagged with its model.
        down, _ = await run_test(
            router,
            frames_to_send=[LLMContextFrame(context=make_context()), SleepFrame()],
            expected_down_frames=[MetricsFrame, MetricsFrame, MetricsFrame],
            pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        router_metrics = [
            d
            for f in down
            if isinstance(f, MetricsFrame)
            for d in f.data
            if d.processor == router.name and d.model == "jev-latest"
        ]
        self.assertEqual(
            {type(m) for m in router_metrics}, {ProcessingMetricsData, LLMUsageMetricsData}
        )
