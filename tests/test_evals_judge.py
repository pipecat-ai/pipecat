#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.evals.judge import (
    JUDGE_FINAL_SYSTEM_INSTRUCTION,
    JUDGE_SYSTEM_INSTRUCTION,
    EvalJudge,
    JudgeVerdict,
    _parse_run_verdicts,
    _parse_verdict,
)


class TestParseRunVerdicts(unittest.TestCase):
    def test_the_goal_and_a_verdict_per_turn_per_criterion(self):
        out = _parse_run_verdicts(
            '{"goal": {"verdict": "yes", "reason": "Berlin was named."}, '
            '"turns": {"politeness": ["yes", "no"], "brevity": ["yes", "yes"]}, '
            '"reasons": {"politeness": {"2": "curt"}}}',
            ["politeness", "brevity"],
            2,
        )
        self.assertEqual((out.goal.verdict, out.goal.reason), ("yes", "Berlin was named."))
        self.assertEqual([v.verdict for v in out.turns["politeness"]], ["yes", "no"])
        self.assertEqual(out.turns["politeness"][1].reason, "curt")
        self.assertEqual(out.turns["politeness"][0].reason, "")
        self.assertEqual([v.verdict for v in out.turns["brevity"]], ["yes", "yes"])

    def test_a_short_array_or_a_missing_criterion_fails_the_turns_it_lacks(self):
        out = _parse_run_verdicts(
            '```json\n{"goal": {"verdict": "no"}, "turns": {"Politeness": ["yes"]}}\n```',
            ["politeness", "brevity"],
            2,
        )
        self.assertEqual((out.goal.verdict, out.goal.reason), ("no", "(no reason given)"))
        self.assertEqual([v.verdict for v in out.turns["politeness"]], ["yes", "none"])
        self.assertEqual(out.turns["politeness"][1].reason, "(judge gave no verdict)")
        self.assertEqual([v.reason for v in out.turns["brevity"]], ["(judge gave no verdict)"] * 2)

    def test_a_failed_call_or_no_json_fails_everything(self):
        out = _parse_run_verdicts("\0judge call failed: Boom", ["politeness"], 1)
        self.assertEqual((out.goal.verdict, out.goal.reason), ("none", "judge call failed: Boom"))
        self.assertEqual(out.turns["politeness"][0].reason, "judge call failed: Boom")
        out = _parse_run_verdicts("no json here", ["politeness"], 1)
        self.assertEqual(out.goal.verdict, "none")
        self.assertEqual(out.turns["politeness"][0].verdict, "none")


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
        # Judge models sometimes append chatty text after the JSON object, and
        # "know" contains "no", so a substring-matching fallback would misread
        # the trailing sentence as a rejection.
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


class TestJudgeWithoutContinue(unittest.IsolatedAsyncioTestCase):
    async def test_a_reply_is_asked_as_yes_or_no(self):
        svc = _FakeLLMService(['{"verdict": "no", "reason": "never names Berlin"}'])
        judge = EvalJudge(svc, allow_continue=False)
        judge.add_assistant_message("No rush, take your time.")
        verdict = await judge.evaluate("says the capital of Germany is Berlin")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(svc.calls[0]["system_instruction"], JUDGE_FINAL_SYSTEM_INSTRUCTION)
        self.assertIn("Answer yes or no.", svc.calls[0]["messages"][-1]["content"])

    async def test_a_continue_answer_counts_as_no(self):
        svc = _FakeLLMService(['{"verdict": "continue", "reason": "still going"}'])
        judge = EvalJudge(svc, allow_continue=False)
        judge.add_assistant_message("Let me think.")
        verdict = await judge.evaluate("gives an answer")
        self.assertEqual(verdict.verdict, "no")

    async def test_continue_is_allowed_by_default(self):
        svc = _FakeLLMService(['{"verdict": "continue", "reason": "still going"}'])
        judge = EvalJudge(svc)
        judge.add_assistant_message("Let me check.")
        verdict = await judge.evaluate("gives the weather")
        self.assertEqual(verdict.verdict, "continue")
        self.assertEqual(svc.calls[0]["system_instruction"], JUDGE_SYSTEM_INSTRUCTION)

    def test_allow_continue_false_is_read_from_the_config(self):
        judge = EvalJudge.from_config({"service": "ollama", "allow_continue": False})
        self.assertFalse(judge._allow_continue)


class TestJudgeEvaluateRun(unittest.IsolatedAsyncioTestCase):
    async def test_the_run_is_judged_over_the_conversation_the_judge_kept(self):
        """Reply segments merge into one numbered bot turn; tool calls sit inline."""
        svc = _FakeLLMService(
            [
                '{"goal": {"verdict": "yes", "reason": "booked"}, "turns": {"polite": ["yes", "yes"]}}'
            ]
        )
        judge = EvalJudge(svc)
        judge.add_user_message("Book a table at six.")
        judge.add_tool_call('book({"time": "6pm"})')
        judge.add_assistant_message("Let me check.")
        judge.add_assistant_message("Done, six o'clock.")
        judge.add_user_message("Thanks.")
        judge.add_assistant_message("You're welcome.")

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")

        ask = svc.calls[0]["messages"][-1]["content"]
        self.assertIn("User: Book a table at six.", ask)
        self.assertIn('[tool call] book({"time": "6pm"})', ask)
        self.assertIn("Bot turn 1: Let me check. Done, six o'clock.", ask)
        self.assertIn("Bot turn 2: You're welcome.", ask)
        self.assertIn("there are 2 bot turns", ask)
        self.assertTrue(verdicts.goal.passed)
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["yes", "yes"])

    async def test_a_reply_is_judged_on_the_spoken_conversation_only(self):
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "ok"}'])
        judge = EvalJudge(svc)
        judge.add_tool_call("lookup()")
        judge.add_assistant_message("It's 72 and sunny.")
        await judge.evaluate("describes the weather")
        roles = [m["role"] for m in svc.calls[0]["messages"]]
        self.assertEqual(roles, ["assistant", "user"])

    async def test_a_transcript_passed_in_is_deprecated_and_judged_in_place_of_the_kept_one(self):
        svc = _FakeLLMService(
            ['{"goal": {"verdict": "yes", "reason": "ok"}, "turns": {"polite": ["yes"]}}'] * 2
        )
        judge = EvalJudge(svc)
        judge.add_assistant_message("kept")
        given = [{"role": "assistant", "content": "given"}]
        for call in (
            lambda: judge.evaluate_run(given, {"polite": "is polite"}, "done"),
            lambda: judge.evaluate_run({"polite": "is polite"}, "done", transcript=given),
        ):
            with self.assertWarns(DeprecationWarning):
                verdicts = await call()
            self.assertTrue(verdicts.goal.passed)
            ask = svc.calls[-1]["messages"][-1]["content"]
            self.assertIn("Bot turn 1: given", ask)
            self.assertNotIn("kept", ask)


class TestJudgeToolCalls(unittest.IsolatedAsyncioTestCase):
    """A verdict on one function call."""

    async def test_evaluate_call_asks_about_the_named_call(self):
        svc = _FakeLLMService(['{"verdict": "no", "reason": "wrong speaker"}'])
        judge = EvalJudge(svc)
        v = await judge.evaluate_call("submit", {"speaker": "Ann"}, "submitted for Bob")
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, "wrong speaker")
        ask = svc.calls[0]["messages"][-1]["content"]
        self.assertIn('called the function `submit` with arguments `{"speaker": "Ann"}`', ask)
        self.assertIn("Criterion: submitted for Bob", ask)

    async def test_evaluate_call_instructs_a_yes_or_no_verdict(self):
        """A call is judged under its own instructions: there is no reply to wait for."""
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "ok"}'])
        judge = EvalJudge(svc)
        await judge.evaluate_call("submit", {"speaker": "Ann"}, "submitted for Ann")
        instruction = svc.calls[0]["system_instruction"]
        self.assertIn("function call", instruction)
        self.assertNotIn("continue", instruction)

    async def test_evaluate_call_caches_per_call(self):
        """The same criterion on two different calls is two questions."""
        svc = _FakeLLMService(
            ['{"verdict": "yes", "reason": "a"}', '{"verdict": "no", "reason": "b"}']
        )
        judge = EvalJudge(svc)
        first = await judge.evaluate_call("submit", {"n": 1}, "n is one")
        again = await judge.evaluate_call("submit", {"n": 1}, "n is one")
        second = await judge.evaluate_call("submit", {"n": 2}, "n is one")
        self.assertTrue(first.passed)
        self.assertTrue(again.passed)
        self.assertFalse(second.passed)
        self.assertEqual(len(svc.calls), 2)


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
