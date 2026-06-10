#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.evals.judge import EvalJudge, JudgeVerdict, _parse_verdict


class TestParseVerdict(unittest.TestCase):
    def test_clean_json_yes(self):
        v = _parse_verdict('{"verdict": "yes", "reason": "It mentions weather."}')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "It mentions weather.")

    def test_clean_json_no(self):
        v = _parse_verdict('{"verdict": "no", "reason": "Does not mention it."}')
        self.assertFalse(v.passed)
        self.assertEqual(v.verdict, "no")
        self.assertEqual(v.reason, "Does not mention it.")

    def test_clean_json_continue(self):
        v = _parse_verdict('{"verdict": "continue", "reason": "Just a filler so far."}')
        self.assertEqual(v.verdict, "continue")
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, "Just a filler so far.")

    def test_unknown_verdict_fails_closed(self):
        v = _parse_verdict('{"verdict": "maybe", "reason": "x"}')
        self.assertEqual(v.verdict, "no")

    def test_unstructured_continue_fallback(self):
        v = _parse_verdict("continue, more text is needed")
        self.assertEqual(v.verdict, "continue")

    def test_fenced_json(self):
        v = _parse_verdict('```json\n{"verdict": "yes", "reason": "ok"}\n```')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "ok")

    def test_fenced_json_without_lang(self):
        v = _parse_verdict('```\n{"verdict": "yes", "reason": "ok"}\n```')
        self.assertTrue(v.passed)

    def test_trailing_prose_after_json(self):
        # gemma2 and similar models sometimes append chatty text after the JSON
        # object (and "know" contains "no", which used to defeat the fallback).
        v = _parse_verdict(
            ' {"verdict": "yes", "reason": "The bot greets the user."}\n\n'
            "Let me know if you'd like to evaluate any further turns!"
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "The bot greets the user.")

    def test_leading_prose_before_json(self):
        v = _parse_verdict('Sure, here is my verdict: {"verdict": "no", "reason": "wrong"}')
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, "wrong")

    def test_unstructured_yes_fallback(self):
        v = _parse_verdict("yes, this satisfies the criterion")
        self.assertTrue(v.passed)

    def test_unstructured_no_fallback(self):
        v = _parse_verdict("no, it does not")
        self.assertFalse(v.passed)

    def test_ambiguous_response_fails_closed(self):
        v = _parse_verdict("the answer is yes or possibly no")
        self.assertFalse(v.passed)
        self.assertIn("could not parse", v.reason)

    def test_garbage_response(self):
        v = _parse_verdict("???")
        self.assertFalse(v.passed)

    def test_extra_whitespace(self):
        v = _parse_verdict('  \n {"verdict": "yes", "reason": "x"}  \n ')
        self.assertTrue(v.passed)

    def test_missing_reason(self):
        v = _parse_verdict('{"verdict": "yes"}')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "(no reason given)")


class _FakeLLMService:
    """In-memory stand-in for a pipecat LLM service.

    Records every call and returns a queued response.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def run_inference(
        self,
        context,
        max_tokens=None,
        system_instruction=None,
    ) -> str:
        self.calls.append(
            {
                "messages": list(context._messages),
                "max_tokens": max_tokens,
                "system_instruction": system_instruction,
            }
        )
        if not self._responses:
            raise RuntimeError("FakeLLMService: no more queued responses")
        return self._responses.pop(0)


class TestJudgeEvaluate(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_pass(self):
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "yes it does"}'])
        judge = EvalJudge(svc)
        judge.add_assistant_message("It's raining in Paris.")
        v = await judge.evaluate("mentions weather")
        self.assertTrue(v.passed)
        self.assertEqual(len(svc.calls), 1)

    async def test_evaluate_fail(self):
        svc = _FakeLLMService(['{"verdict": "no", "reason": "no it does not"}'])
        judge = EvalJudge(svc)
        judge.add_assistant_message("Hello there.")
        v = await judge.evaluate("mentions weather")
        self.assertFalse(v.passed)

    async def test_evaluate_judges_in_conversation_context(self):
        """The user turn and reply are both sent to the judge as messages."""
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "four"}'])
        judge = EvalJudge(svc)
        judge.add_user_message("What is two plus two?")
        judge.add_assistant_message("That's for")  # terse + STT homophone
        v = await judge.evaluate("answers that two plus two is four")
        self.assertTrue(v.passed)
        roles = [m["role"] for m in svc.calls[0]["messages"]]
        contents = [m["content"] for m in svc.calls[0]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])  # question, reply, verdict ask
        self.assertIn("What is two plus two?", contents)
        self.assertIn("That's for", contents)

    async def test_streamed_segments_are_separate_messages(self):
        """Each reply segment becomes its own assistant message (no overlap)."""
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "ok"}'])
        judge = EvalJudge(svc)
        judge.add_assistant_message("Let me check on that.")
        judge.add_assistant_message("It's 72 and sunny.")
        await judge.evaluate("describes the weather")
        assistant = [m["content"] for m in svc.calls[0]["messages"] if m["role"] == "assistant"]
        self.assertEqual(assistant, ["Let me check on that.", "It's 72 and sunny."])

    async def test_caching_avoids_second_call(self):
        """Same (criterion, conversation) within one EvalJudge hits the cache."""
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "ok"}'])
        judge = EvalJudge(svc)
        judge.add_assistant_message("It rains.")
        v1 = await judge.evaluate("mentions weather")
        v2 = await judge.evaluate("mentions weather")
        self.assertTrue(v1.passed)
        self.assertTrue(v2.passed)
        self.assertEqual(len(svc.calls), 1, "second call should be cached")

    async def test_service_failure_reported_not_raised(self):
        """If the LLM call raises, the judge returns a failed verdict, not an exception."""

        class _BoomService:
            async def run_inference(self, **kwargs):
                raise RuntimeError("network down")

        judge = EvalJudge(_BoomService())
        judge.add_assistant_message("anything")
        v = await judge.evaluate("anything")
        self.assertFalse(v.passed)
        self.assertIn("RuntimeError", v.reason)

    async def test_empty_response_fails(self):
        svc = _FakeLLMService([""])
        judge = EvalJudge(svc)
        judge.add_assistant_message("anything")
        v = await judge.evaluate("anything")
        self.assertFalse(v.passed)


class TestJudgeVerdictDataclass(unittest.TestCase):
    def test_construction(self):
        v = JudgeVerdict(verdict="yes", reason="ok", raw_response="raw")
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "ok")
        self.assertEqual(v.raw_response, "raw")

    def test_passed_only_for_yes(self):
        self.assertTrue(JudgeVerdict(verdict="yes", reason="", raw_response="").passed)
        self.assertFalse(JudgeVerdict(verdict="no", reason="", raw_response="").passed)
        self.assertFalse(JudgeVerdict(verdict="continue", reason="", raw_response="").passed)


if __name__ == "__main__":
    unittest.main()
