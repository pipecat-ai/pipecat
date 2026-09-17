#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    SystemOneResponse,
    TypeSafeAPITimeoutError,
    Usage,
)

from pipecat.evals.judge import EvalJudge
from pipecat.evals.typesafe_judge import (
    GOAL_QUESTION_ID,
    VERDICT_QUESTION_ID,
    TypeSafeEvalJudge,
)
from pipecat.services.typesafe import TypeSafeJudge
from tests.typesafe_test_helpers import FakeTypeSafeClient


def response(answers: dict) -> SystemOneResponse:
    return SystemOneResponse(
        model="jev-test", usage=Usage(input_tokens=10, output_tokens=2), answers=answers
    )


def verdict_response(probabilities: dict[str, float]) -> SystemOneResponse:
    choice = max(probabilities, key=lambda k: probabilities[k])
    return response(
        {
            VERDICT_QUESTION_ID: ChoiceAnswer(
                choice=choice, confidence=probabilities[choice], probabilities=probabilities
            )
        }
    )


def make_judge(client) -> TypeSafeEvalJudge:
    return TypeSafeEvalJudge(TypeSafeJudge(client=client))


class TestFromConfig(unittest.TestCase):
    def test_service_typesafe_builds_the_typesafe_judge(self):
        judge = EvalJudge.from_config({"service": "typesafe", "threshold": 0.7})
        self.assertIsInstance(judge, TypeSafeEvalJudge)
        self.assertEqual(judge._threshold, 0.7)


class TestEvaluate(unittest.IsolatedAsyncioTestCase):
    async def test_the_most_probable_option_is_the_verdict(self):
        client = FakeTypeSafeClient(verdict_response({"yes": 0.91, "no": 0.07, "continue": 0.02}))
        judge = make_judge(client)
        judge.add_user_message("What is the capital of France?")
        judge.add_assistant_message("It's Paris.")

        verdict = await judge.evaluate("says the capital is Paris")

        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.reason, "yes 0.91, no 0.07, continue 0.02")
        state, questions = client.requests[0]
        self.assertEqual(
            state,
            {
                "conversation": [{"speaker": "user", "text": "What is the capital of France?"}],
                "bot_reply": "It's Paris.",
                "criterion": "says the capital is Paris",
            },
        )
        self.assertEqual(set(questions), {VERDICT_QUESTION_ID})

    async def test_streamed_segments_form_one_reply_and_tool_calls_are_left_out(self):
        client = FakeTypeSafeClient(verdict_response({"yes": 0.2, "no": 0.1, "continue": 0.7}))
        judge = make_judge(client)
        judge.add_user_message("Weather in Paris?")
        judge.add_tool_call("get_weather()")
        judge.add_assistant_message("Let me")
        judge.add_assistant_message("check that.")

        verdict = await judge.evaluate("gives the weather")

        self.assertEqual(verdict.verdict, "continue")
        state, _ = client.requests[0]
        self.assertEqual(state["bot_reply"], "Let me check that.")
        self.assertEqual(state["conversation"], [{"speaker": "user", "text": "Weather in Paris?"}])

    async def test_same_question_is_asked_once(self):
        client = FakeTypeSafeClient(verdict_response({"yes": 0.9, "no": 0.1, "continue": 0.0}))
        judge = make_judge(client)
        judge.add_assistant_message("Hello!")

        await judge.evaluate("greets")
        await judge.evaluate("greets")

        self.assertEqual(len(client.requests), 1)

    async def test_a_failed_call_is_a_no_with_the_reason(self):
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        judge = make_judge(client)
        judge.add_assistant_message("Hello!")

        verdict = await judge.evaluate("greets")

        self.assertEqual(verdict.verdict, "no")
        self.assertIn("judge call failed", verdict.reason)


class TestEvaluateRun(unittest.IsolatedAsyncioTestCase):
    async def test_one_request_decides_the_goal_and_every_turn(self):
        client = FakeTypeSafeClient(
            response(
                {
                    GOAL_QUESTION_ID: NoulAnswer(noul=0.95),
                    "polite:1": NoulAnswer(noul=0.9),
                    "polite:2": NoulAnswer(noul=0.2),
                }
            )
        )
        judge = make_judge(client)
        judge.add_user_message("Book a table at six.")
        judge.add_tool_call('book({"time": "6pm"})')
        judge.add_assistant_message("Let me check.")
        judge.add_assistant_message("Done, six o'clock.")
        judge.add_user_message("Thanks.")
        judge.add_assistant_message("Bye.")

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")

        self.assertTrue(verdicts.goal.passed)
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["yes", "no"])
        self.assertEqual(verdicts.turns["polite"][1].reason, "yes 0.20")
        state, questions = client.requests[0]
        self.assertEqual(
            state["transcript"],
            [
                "User: Book a table at six.",
                '[tool call] book({"time": "6pm"})',
                "Bot turn 1: Let me check. Done, six o'clock.",
                "User: Thanks.",
                "Bot turn 2: Bye.",
            ],
        )
        self.assertEqual(set(questions), {GOAL_QUESTION_ID, "polite:1", "polite:2"})
        self.assertIn("a table is booked", questions[GOAL_QUESTION_ID].instructions)
        self.assertIn("Bot turn 2", questions["polite:2"].instructions)

    async def test_a_failed_call_gives_no_verdicts(self):
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        judge = make_judge(client)
        judge.add_assistant_message("Hi.")

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "done")

        self.assertEqual(verdicts.goal.verdict, "none")
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["none"])
