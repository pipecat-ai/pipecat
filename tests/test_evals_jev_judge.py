#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import os
import unittest
from collections.abc import Callable
from unittest.mock import patch

import httpx2

from pipecat.evals.base_judge import judge_from_config
from pipecat.evals.jev_judge import JevEvalJudge
from pipecat.evals.judge import EvalJudge


class _FakeApi:
    """Stands in for Jev's API behind an httpx2 mock transport: records each request and answers it.

    Answers come from a queue, or from a function of the request body when
    the order requests arrive in isn't fixed. A queued exception is raised,
    as the transport would on a timeout.
    """

    def __init__(self, responses: list | Callable[[dict], tuple[int, dict]]):
        self._responses = responses if callable(responses) else list(responses)
        self.requests: list[dict] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            self.requests.append({"method": "GET", "url": str(request.url)})
            return httpx2.Response(200, json={"models": []})
        body = json.loads(request.content)
        self.requests.append(
            {"method": "POST", "url": str(request.url), "body": body, "headers": request.headers}
        )
        response = self._responses(body) if callable(self._responses) else self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status, payload = response
        return httpx2.Response(status, json=payload)

    @property
    def posts(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle))


class _FakeLLMService:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    async def run_inference(self, context, max_tokens=None, system_instruction=None):
        self.calls += 1
        return self._responses.pop(0)


def _choice(choice: str, confidence: float = 0.95) -> dict:
    others = [c for c in ("yes", "no", "continue") if c != choice]
    rest = (1 - confidence) / 2
    probabilities = {choice: confidence, **{c: rest for c in others}}
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def _noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def _turn(meets: float | dict) -> dict:
    probabilities = (
        meets
        if isinstance(meets, dict)
        else {"meets": meets, "fails": 1 - meets, "not_applicable": 0.0}
    )
    choice = max(probabilities, key=probabilities.get)
    return {
        "type": "choice",
        "choice": choice,
        "confidence": probabilities[choice],
        "probabilities": probabilities,
    }


def _ok(answers: dict) -> tuple[int, dict]:
    return 200, {"model": "jev-1", "answers": answers, "usage": {}}


def _judge(responses, explainer=None, **kwargs) -> tuple[JevEvalJudge, _FakeApi]:
    api = _FakeApi(responses)
    judge = JevEvalJudge(api_key="k", client=api.client(), explainer=explainer, **kwargs)
    return judge, api


class TestJevEvaluate(unittest.IsolatedAsyncioTestCase):
    async def test_the_latest_reply_is_judged_after_the_spoken_conversation(self):
        judge, api = _judge([_ok({"verdict": _choice("yes")})])
        judge.add_user_message("What's the weather?")
        judge.add_tool_call("lookup()")
        judge.add_assistant_message("Let me check.")
        judge.add_assistant_message("It's 72 and sunny.")

        verdict = await judge.evaluate("describes the weather")

        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.confidence, 0.95)
        self.assertIn("P(yes)=0.95", verdict.reason)
        request = api.posts[0]
        self.assertTrue(request["url"].endswith("/v1/systemone"))
        self.assertEqual(request["headers"]["Authorization"], "Bearer k")
        state = request["body"]["state"]
        self.assertEqual(state["latest_bot_reply"], "Let me check. It's 72 and sunny.")
        self.assertEqual(
            state["conversation"], [{"speaker": "user", "text": "What's the weather?"}]
        )
        question = request["body"]["questions"]["verdict"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), {"yes", "no", "continue"})

    async def test_a_verdict_is_cached_by_criterion_and_conversation(self):
        judge, api = _judge([_ok({"verdict": _choice("yes")})])
        judge.add_assistant_message("It rains.")
        await judge.evaluate("mentions weather")
        await judge.evaluate("mentions weather")
        self.assertEqual(len(api.posts), 1)

    async def test_a_failed_request_is_a_no(self):
        judge, _ = _judge([(500, {"error": "boom"})] * 2)
        judge.add_assistant_message("anything")
        verdict = await judge.evaluate("anything")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(verdict.reason, "judge call failed")

    async def test_an_answer_of_the_wrong_type_is_a_failed_request(self):
        judge, _ = _judge([_ok({"verdict": _noul(0.9)})])
        judge.add_assistant_message("anything")
        verdict = await judge.evaluate("anything")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(verdict.reason, "judge call failed")


class TestJevExplainer(unittest.IsolatedAsyncioTestCase):
    async def test_a_no_takes_the_explainers_reason(self):
        llm = _FakeLLMService(['{"verdict": "no", "reason": "it never mentions rain"}'])
        judge, _ = _judge([_ok({"verdict": _choice("no")})], explainer=EvalJudge(llm))
        judge.add_assistant_message("Hello.")
        verdict = await judge.evaluate("mentions weather")
        self.assertEqual(verdict.verdict, "no")
        self.assertTrue(verdict.reason.startswith("it never mentions rain"))
        self.assertIn("Jev:", verdict.reason)

    async def test_a_disagreeing_explainer_is_noted_and_jev_stands(self):
        llm = _FakeLLMService(['{"verdict": "yes", "reason": "it says it rains"}'])
        judge, _ = _judge([_ok({"verdict": _choice("no")})], explainer=EvalJudge(llm))
        judge.add_assistant_message("It rains.")
        verdict = await judge.evaluate("mentions weather")
        self.assertEqual(verdict.verdict, "no")
        self.assertIn("the explainer judged yes: it says it rains", verdict.reason)

    async def test_an_agreeing_explainer_without_a_reason_leaves_jevs(self):
        llm = _FakeLLMService(['{"verdict": "no"}'])
        judge, _ = _judge([_ok({"verdict": _choice("no")})], explainer=EvalJudge(llm))
        judge.add_assistant_message("Hello.")
        verdict = await judge.evaluate("mentions weather")
        self.assertTrue(verdict.reason.startswith("Jev:"))

    async def test_a_sure_yes_and_a_continue_are_not_explained(self):
        llm = _FakeLLMService([])
        judge, _ = _judge(
            [_ok({"verdict": _choice("continue")}), _ok({"verdict": _choice("yes")})],
            explainer=EvalJudge(llm),
        )
        judge.add_assistant_message("Let me check.")
        self.assertEqual((await judge.evaluate("gives the weather")).verdict, "continue")
        judge.add_assistant_message("It's sunny.")
        self.assertTrue((await judge.evaluate("gives the weather")).passed)
        self.assertEqual(llm.calls, 0)

    async def test_an_unsure_yes_is_explained(self):
        llm = _FakeLLMService(['{"verdict": "yes", "reason": "close enough"}'])
        judge, _ = _judge([_ok({"verdict": _choice("yes", 0.6)})], explainer=EvalJudge(llm))
        judge.add_assistant_message("Sunny-ish.")
        verdict = await judge.evaluate("gives the weather")
        self.assertTrue(verdict.passed)
        self.assertTrue(verdict.reason.startswith("close enough"))
        self.assertEqual(llm.calls, 1)


class TestJevEvaluateCall(unittest.IsolatedAsyncioTestCase):
    async def test_a_call_is_judged_by_name_and_arguments(self):
        judge, api = _judge([_ok({"verdict": _noul(0.2)})])
        judge.add_user_message("Book six o'clock.")
        verdict = await judge.evaluate_call("book", {"time": "7pm"}, "books six o'clock")
        self.assertEqual(verdict.verdict, "no")
        self.assertAlmostEqual(verdict.confidence, 0.8)
        state = api.posts[0]["body"]["state"]
        self.assertEqual(state["call"], {"name": "book", "arguments": {"time": "7pm"}})


class TestJevEvaluateRun(unittest.IsolatedAsyncioTestCase):
    def _converse(self, judge: JevEvalJudge) -> None:
        judge.add_user_message("Book a table at six.")
        judge.add_tool_call('book({"time": "6pm"})')
        judge.add_assistant_message("Let me check.")
        judge.add_assistant_message("Done, six o'clock.")
        judge.add_user_message("Thanks.")
        judge.add_assistant_message("You're welcome.")

    @staticmethod
    def _answer(goal: float, turns: dict[str, float | dict], failing: str | None = None):
        """Answers a run's requests: the goal's, and each turn's by its reply text.

        A turn's answer is its probability of meeting the criterion (the rest
        failing it), or the choice's full probabilities.
        """

        def respond(body: dict) -> tuple[int, dict]:
            if "goal" in body["questions"]:
                return (503, {}) if failing == "goal" else _ok({"goal": _noul(goal)})
            reply = body["state"]["latest_bot_reply"]
            if reply == failing:
                return 503, {}
            return _ok({"c0": _turn(turns[reply])})

        return respond

    async def test_the_goal_is_asked_over_the_whole_conversation(self):
        judge, api = _judge(
            self._answer(0.9, {"Let me check. Done, six o'clock.": 0.8, "You're welcome.": 0.3})
        )
        self._converse(judge)

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")

        self.assertTrue(verdicts.goal.passed)
        goal = next(r["body"] for r in api.posts if "goal" in r["body"]["questions"])
        self.assertEqual(
            goal["state"]["conversation"],
            [
                {"speaker": "user", "text": "Book a table at six."},
                {"speaker": "tool", "text": 'book({"time": "6pm"})'},
                {"speaker": "bot", "turn": 1, "text": "Let me check. Done, six o'clock."},
                {"speaker": "user", "text": "Thanks."},
                {"speaker": "bot", "turn": 2, "text": "You're welcome."},
            ],
        )

    async def test_each_turn_is_judged_after_only_the_conversation_before_it(self):
        judge, api = _judge(
            self._answer(0.9, {"Let me check. Done, six o'clock.": 0.8, "You're welcome.": 0.3})
        )
        self._converse(judge)

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")

        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["yes", "no"])
        turns = [r["body"] for r in api.posts if "goal" not in r["body"]["questions"]]
        self.assertEqual(len(turns), 2)
        by_reply = {t["state"]["latest_bot_reply"]: t for t in turns}
        self.assertEqual(
            by_reply["Let me check. Done, six o'clock."]["state"]["conversation"],
            [
                {"speaker": "user", "text": "Book a table at six."},
                {"speaker": "tool", "text": 'book({"time": "6pm"})'},
            ],
        )
        self.assertEqual(len(by_reply["You're welcome."]["state"]["conversation"]), 4)
        self.assertEqual(set(by_reply["You're welcome."]["questions"]), {"c0"})

    async def test_a_turn_the_criterion_does_not_apply_to_passes(self):
        judge, api = _judge(
            self._answer(
                0.9,
                {
                    "Let me check. Done, six o'clock.": {
                        "meets": 0.3,
                        "fails": 0.1,
                        "not_applicable": 0.6,
                    },
                    "You're welcome.": {"meets": 0.05, "fails": 0.9, "not_applicable": 0.05},
                },
            )
        )
        self._converse(judge)

        verdicts = await judge.evaluate_run(
            {"apology": "when a time is taken, apologises"}, "booked"
        )

        first, second = verdicts.turns["apology"]
        self.assertEqual(first.verdict, "yes")
        self.assertAlmostEqual(first.confidence, 0.9)
        self.assertEqual(second.verdict, "no")
        self.assertAlmostEqual(second.confidence, 0.9)
        question = next(
            r["body"]["questions"]["c0"] for r in api.posts if "goal" not in r["body"]["questions"]
        )
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), {"meets", "fails", "not_applicable"})

    async def test_a_run_with_no_criteria_asks_only_the_goal(self):
        judge, api = _judge(self._answer(0.9, {}))
        self._converse(judge)
        verdicts = await judge.evaluate_run({}, "a table is booked")
        self.assertTrue(verdicts.goal.passed)
        self.assertEqual(len(api.posts), 1)

    async def test_a_failed_request_blanks_only_its_own_verdicts(self):
        judge, _ = _judge(
            self._answer(0.9, {"You're welcome.": 0.8}, failing="Let me check. Done, six o'clock.")
        )
        self._converse(judge)
        verdicts = await judge.evaluate_run({"polite": "is polite"}, "booked")
        self.assertTrue(verdicts.goal.passed)
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["none", "yes"])

    async def test_a_failed_goal_request_gives_no_goal_verdict(self):
        judge, _ = _judge(
            self._answer(
                0.9,
                {"Let me check. Done, six o'clock.": 0.8, "You're welcome.": 0.8},
                failing="goal",
            )
        )
        self._converse(judge)
        verdicts = await judge.evaluate_run({"polite": "is polite"}, "booked")
        self.assertEqual(verdicts.goal.verdict, "none")
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["yes", "yes"])

    async def test_the_explainer_reasons_only_the_verdicts_that_need_one(self):
        llm = _FakeLLMService(
            [
                '{"goal": {"verdict": "yes", "reason": "booked"}, '
                '"turns": {"polite": ["yes", "no"]}, '
                '"reasons": {"polite": {"2": "curt"}}}'
            ]
        )
        judge, _ = _judge(
            self._answer(0.95, {"Let me check. Done, six o'clock.": 0.9, "You're welcome.": 0.1}),
            explainer=EvalJudge(llm),
        )
        self._converse(judge)

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "booked")

        self.assertEqual(llm.calls, 1)
        self.assertEqual(verdicts.goal.reason, "Jev: P(yes)=0.95")
        first, second = verdicts.turns["polite"]
        self.assertEqual(first.reason, "Jev: P(meets)=0.90, P(fails)=0.10, P(not_applicable)=0.00")
        self.assertTrue(second.reason.startswith("curt"))


class TestJevConfig(unittest.IsolatedAsyncioTestCase):
    async def test_service_typesafe_builds_a_jev_judge_with_the_default_explainer(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "k"}):
            judge = judge_from_config({"service": "typesafe"})
        self.assertIsInstance(judge, JevEvalJudge)
        self.assertIsInstance(judge._explainer, EvalJudge)
        await judge.close()

    async def test_explainer_false_builds_no_explainer(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "k"}):
            judge = judge_from_config({"service": "typesafe", "explainer": False})
        self.assertIsNone(judge._explainer)
        await judge.close()

    def test_a_missing_api_key_is_an_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                JevEvalJudge()

    async def test_close_leaves_a_client_it_was_given_open(self):
        client = _FakeApi([]).client()
        judge = JevEvalJudge(api_key="k", client=client)
        await judge.close()
        self.assertFalse(client.is_closed)


def _timeout() -> httpx2.ReadTimeout:
    return httpx2.ReadTimeout("stalled")


class _OwnClientJudge(JevEvalJudge):
    """A judge that builds its own clients, each a new mock connection to one API."""

    def __init__(self, api: _FakeApi, **kwargs):
        self.api = api
        self.clients: list[httpx2.AsyncClient] = []
        super().__init__(api_key="k", **kwargs)

    def _new_client(self) -> httpx2.AsyncClient:
        client = self.api.client()
        self.clients.append(client)
        return client


class TestJevConnection(unittest.IsolatedAsyncioTestCase):
    async def _ask(self, judge: JevEvalJudge):
        judge.add_assistant_message("It rains.")
        return await judge.evaluate("mentions weather")

    async def test_a_timeout_is_retried_once(self):
        judge, api = _judge([_timeout(), _ok({"verdict": _choice("yes")})])
        self.assertTrue((await self._ask(judge)).passed)
        self.assertEqual(len(api.posts), 2)

    async def test_a_server_error_is_retried_once(self):
        judge, api = _judge([(503, {}), _ok({"verdict": _choice("yes")})])
        self.assertTrue((await self._ask(judge)).passed)
        self.assertEqual(len(api.posts), 2)

    async def test_a_second_failure_fails_the_request(self):
        judge, api = _judge([_timeout(), _timeout()])
        verdict = await self._ask(judge)
        self.assertEqual(verdict.reason, "judge call failed")
        self.assertEqual(len(api.posts), 2)

    async def test_a_client_error_is_not_retried(self):
        judge, api = _judge([(400, {"error": "bad question"})])
        verdict = await self._ask(judge)
        self.assertEqual(verdict.reason, "judge call failed")
        self.assertEqual(len(api.posts), 1)

    async def test_the_connection_is_opened_when_the_judge_is_created(self):
        api = _FakeApi([_ok({"verdict": _choice("yes")})])
        judge = _OwnClientJudge(api)
        await self._ask(judge)
        self.assertEqual([r["method"] for r in api.requests], ["GET", "POST"])
        self.assertTrue(api.requests[0]["url"].endswith("/v1/models"))
        await judge.close()

    async def test_a_retry_goes_out_on_a_new_connection_and_close_closes_both(self):
        api = _FakeApi([_timeout(), _ok({"verdict": _choice("yes")})])
        judge = _OwnClientJudge(api)
        self.assertTrue((await self._ask(judge)).passed)
        self.assertEqual(len(judge.clients), 2)
        await judge.close()
        self.assertTrue(all(c.is_closed for c in judge.clients))


if __name__ == "__main__":
    unittest.main()
