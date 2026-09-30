#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The judge over a Jev classifier: what goes on the wire, and what comes back."""

import json
import os
import unittest
from collections.abc import Callable
from unittest.mock import patch

import httpx

from pipecat.classifiers.jev.classifier import JevClassifier
from pipecat.classifiers.jev.client import DEFAULT_MODEL, JevClient
from pipecat.evals.judge import EvalJudge


class _FakeApi:
    """Stands in for Jev's API behind an httpx mock transport: records each request and answers it.

    Answers come from a queue, or from a function of the request body when
    the order requests arrive in isn't fixed. A queued exception is raised,
    as the transport would on a timeout.
    """

    def __init__(self, responses: list | Callable[[dict], tuple[int, dict]]):
        self._responses = responses if callable(responses) else list(responses)
        self.requests: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.requests.append({"method": "GET", "url": str(request.url)})
            return httpx.Response(200, json={"models": []})
        body = json.loads(request.content)
        self.requests.append(
            {"method": "POST", "url": str(request.url), "body": body, "headers": request.headers}
        )
        response = self._responses(body) if callable(self._responses) else self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status, payload = response
        return httpx.Response(status, json=payload)

    @property
    def posts(self) -> list[dict]:
        return [r for r in self.requests if r["method"] == "POST"]

    def client(self, **kwargs) -> JevClient:
        client = JevClient(**{"api_key": "k", **kwargs})
        client._http._transport = httpx.MockTransport(self.handle)
        return client

    def classifier(self) -> JevClassifier:
        return JevClassifier(client=self.client())


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


def _turn(meets: float) -> dict:
    probabilities = {"meets": meets, "fails": 1 - meets, "not_applicable": 0.0}
    choice = max(probabilities, key=probabilities.get)
    return {
        "type": "choice",
        "choice": choice,
        "confidence": probabilities[choice],
        "probabilities": probabilities,
    }


def _ok(answers: dict) -> tuple[int, dict]:
    return 200, {"model": "jev-1", "answers": answers, "usage": {}}


def _judge(responses, **kwargs) -> tuple[EvalJudge, _FakeApi]:
    api = _FakeApi(responses)
    return EvalJudge(classifier=api.classifier(), **kwargs), api


_FACTORY = "tests.test_evals_judge_jev.typesafe_classifier"


def typesafe_classifier(config: dict) -> JevClassifier:
    """A factory like the release evals' own, over the patched client."""
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise ValueError("no key")
    return JevClassifier(api_key=api_key, model=config.get("model") or DEFAULT_MODEL, timeout=2.5)


def _config_judge(api: _FakeApi, config: dict) -> EvalJudge:
    """The judge a ``judge.eval:`` block builds, over a mock connection to ``api``."""
    with patch.dict(os.environ, {"TYPESAFE_API_KEY": "k"}):
        with patch("pipecat.classifiers.jev.classifier.JevClient", api.client):
            return EvalJudge.from_config(config)


class TestJevQuestions(unittest.IsolatedAsyncioTestCase):
    async def test_a_reply_is_asked_as_a_choice_over_the_conversation(self):
        judge, api = _judge([_ok({"verdict": _choice("yes")})])
        judge.add_user_message("What's the weather?")
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

    async def test_a_call_is_asked_as_a_yes_or_no(self):
        judge, api = _judge([_ok({"answer": _noul(0.2)})])
        judge.add_user_message("Book six o'clock.")
        verdict = await judge.evaluate_call("book", {"time": "7pm"}, "books six o'clock")
        self.assertEqual(verdict.verdict, "no")
        self.assertAlmostEqual(verdict.confidence, 0.8)
        request = api.posts[0]["body"]
        self.assertEqual(request["state"]["call"], {"name": "book", "arguments": {"time": "7pm"}})
        self.assertEqual(request["questions"]["answer"]["type"], "noul")

    async def test_a_run_asks_the_goal_and_every_turn_at_once(self):
        def respond(body: dict) -> tuple[int, dict]:
            reply = body["state"].get("latest_bot_reply")
            if reply is None:
                return _ok({"answer": _noul(0.9)})
            return _ok(
                {name: _turn(0.9 if reply == "Sure." else 0.1) for name in body["questions"]}
            )

        judge, api = _judge(respond)
        judge.add_user_message("Book a table at six.")
        judge.add_assistant_message("Sure.")
        judge.add_user_message("Thanks.")
        judge.add_assistant_message("Bye.")

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")

        self.assertTrue(verdicts.goal.passed)
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["yes", "no"])
        self.assertEqual(len(api.posts), 3)
        turn = next(p for p in api.posts if "latest_bot_reply" in p["body"]["state"])
        self.assertEqual(
            set(turn["body"]["questions"]["polite"]["criteria"]),
            {"meets", "fails", "not_applicable"},
        )


class TestJevConnection(unittest.IsolatedAsyncioTestCase):
    async def _ask(self, judge: EvalJudge):
        judge.add_assistant_message("It rains.")
        return await judge.evaluate("mentions weather")

    async def test_a_timeout_is_asked_again(self):
        judge, api = _judge([httpx.ReadTimeout("stalled"), _ok({"verdict": _choice("yes")})])
        self.assertTrue((await self._ask(judge)).passed)
        self.assertEqual(len(api.posts), 2)

    async def test_a_rejected_question_is_asked_again(self):
        judge, api = _judge([(503, {}), _ok({"verdict": _choice("yes")})])
        self.assertTrue((await self._ask(judge)).passed)
        self.assertEqual(len(api.posts), 2)

    async def test_a_second_failure_fails_the_question(self):
        judge, api = _judge([httpx.ReadTimeout("stalled")] * 2)
        verdict = await self._ask(judge)
        self.assertEqual(verdict.reason, "judge call failed")
        self.assertEqual(len(api.posts), 2)

    async def test_the_connection_is_opened_when_the_judge_is_created(self):
        api = _FakeApi([_ok({"verdict": _choice("yes")})])
        judge = _config_judge(api, {"factory": _FACTORY, "explainer": False})
        await self._ask(judge)
        self.assertEqual([r["method"] for r in api.requests], ["GET", "POST"])
        self.assertTrue(api.requests[0]["url"].endswith("/v1/models"))
        await judge.close()

    async def test_close_releases_a_connection_the_judge_opened(self):
        api = _FakeApi([])
        judge = _config_judge(api, {"factory": _FACTORY, "explainer": False})
        await judge.close()
        self.assertTrue(judge.classifier.client._http.is_closed)

    async def test_close_leaves_a_connection_the_judge_was_given_open(self):
        api = _FakeApi([])
        classifier = api.classifier()
        await EvalJudge(classifier=classifier).close()
        self.assertFalse(classifier.client._http.is_closed)


class TestJevConfig(unittest.IsolatedAsyncioTestCase):
    async def test_service_typesafe_judges_with_jev_and_explains_with_an_llm(self):
        judge = _config_judge(_FakeApi([]), {"factory": _FACTORY, "model": "jev-9"})
        self.assertIsInstance(judge.classifier, JevClassifier)
        self.assertEqual(judge.classifier.model, "jev-9")
        self.assertIsNotNone(judge._explainer)
        await judge.close()

    async def test_explainer_false_builds_no_explainer(self):
        judge = _config_judge(_FakeApi([]), {"factory": _FACTORY, "explainer": False})
        self.assertIsNone(judge._explainer)
        await judge.close()

    async def test_the_explainer_follows_allow_continue(self):
        judge = _config_judge(_FakeApi([]), {"factory": _FACTORY, "allow_continue": False})
        self.assertNotIn("continue", judge._reply_outcomes)
        self.assertFalse(judge._explainer._allow_continue)
        await judge.close()

    def test_a_missing_api_key_is_an_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                EvalJudge.from_config({"factory": _FACTORY})


if __name__ == "__main__":
    unittest.main()
