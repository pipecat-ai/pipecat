#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import pytest

pytest.importorskip("typesafe_sdk")

from typesafe_sdk import NoulAnswer, SystemOneResponse, TypeSafeAPITimeoutError, Usage

from pipecat.evals.judge import EvalJudge
from pipecat.evals.typesafe_judge import (
    FINISHED_QUESTION_ID,
    GOAL_QUESTION_ID,
    SATISFIES_QUESTION_ID,
    TypeSafeEvalJudge,
)
from pipecat.services.typesafe import TypeSafeJudge
from tests.typesafe_test_helpers import FakeTypeSafeClient


def response(probabilities: dict[str, float]) -> SystemOneResponse:
    return SystemOneResponse(
        model="jev-test",
        usage=Usage(input_tokens=10, output_tokens=2),
        answers={k: NoulAnswer(noul=p) for k, p in probabilities.items()},
    )


def reply_response(finished: float, satisfies: float) -> SystemOneResponse:
    return response({FINISHED_QUESTION_ID: finished, SATISFIES_QUESTION_ID: satisfies})


def make_judge(client, **kwargs) -> TypeSafeEvalJudge:
    return TypeSafeEvalJudge(TypeSafeJudge(client=client), **kwargs)


class TestFromConfig(unittest.TestCase):
    def test_service_typesafe_builds_the_typesafe_judge(self):
        judge = EvalJudge.from_config(
            {"service": "typesafe", "threshold": 0.7, "uncertain_band": 0.1}
        )
        self.assertIsInstance(judge, TypeSafeEvalJudge)
        self.assertEqual(judge._threshold, 0.7)
        self.assertEqual(judge._uncertain_band, 0.1)


class TestEvaluate(unittest.IsolatedAsyncioTestCase):
    async def test_a_finished_reply_that_satisfies_is_a_yes(self):
        client = FakeTypeSafeClient(reply_response(finished=0.97, satisfies=0.91))
        judge = make_judge(client)
        judge.add_user_message("What is the capital of France?")
        judge.add_assistant_message("It's Paris.")

        verdict = await judge.evaluate("says the capital is Paris")

        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.reason, "finished 0.97, satisfies 0.91")
        state, questions = client.requests[0]
        self.assertEqual(
            state,
            {
                "conversation": [{"speaker": "user", "text": "What is the capital of France?"}],
                "bot_reply": "It's Paris.",
                "criterion": "says the capital is Paris",
            },
        )
        self.assertEqual(set(questions), {FINISHED_QUESTION_ID, SATISFIES_QUESTION_ID})

    async def test_a_finished_reply_that_fails_is_a_no(self):
        client = FakeTypeSafeClient(reply_response(finished=0.95, satisfies=0.1))
        judge = make_judge(client)
        judge.add_assistant_message("It's Lyon.")

        verdict = await judge.evaluate("says the capital is Paris")

        self.assertEqual(verdict.verdict, "no")

    async def test_an_unfinished_reply_is_continue_whatever_it_says_so_far(self):
        client = FakeTypeSafeClient(reply_response(finished=0.2, satisfies=0.05))
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
        client = FakeTypeSafeClient(reply_response(finished=0.9, satisfies=0.9))
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
            response({GOAL_QUESTION_ID: 0.95, "polite:1": 0.9, "polite:2": 0.2})
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
        self.assertIsNotNone(questions["polite:2"].criteria)

    async def test_a_probability_near_the_threshold_is_no_verdict(self):
        client = FakeTypeSafeClient(response({GOAL_QUESTION_ID: 0.56, "polite:1": 0.44}))
        judge = make_judge(client, threshold=0.5, uncertain_band=0.15)
        judge.add_assistant_message("Hi.")

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "done")

        self.assertEqual(verdicts.goal.verdict, "none")
        self.assertEqual(verdicts.goal.reason, "yes 0.56: too close to call")
        self.assertEqual(verdicts.turns["polite"][0].verdict, "none")

    async def test_a_failed_call_gives_no_verdicts(self):
        client = FakeTypeSafeClient(error=TypeSafeAPITimeoutError("slow"))
        judge = make_judge(client)
        judge.add_assistant_message("Hi.")

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "done")

        self.assertEqual(verdicts.goal.verdict, "none")
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["none"])
