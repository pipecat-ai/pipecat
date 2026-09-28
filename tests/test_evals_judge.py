#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import unittest
from types import SimpleNamespace

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceResult,
    ClassifierError,
    YesNoResult,
)
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.evals.judge import (
    JUDGE_CALL_SYSTEM_INSTRUCTION,
    JUDGE_FINAL_SYSTEM_INSTRUCTION,
    JUDGE_SYSTEM_INSTRUCTION,
    RUN_JUDGE_SYSTEM_INSTRUCTION,
    EvalJudge,
    JudgeVerdict,
    _Explainer,
)

REPLY_OPTIONS = ("yes", "no", "continue")
TURN_OPTIONS = ("meets", "fails", "not_applicable")


class _FakeClassifier(BaseClassifier):
    """Answers with queued results, or with a function of the state it is asked about.

    Records every question, so a test can check what the judge asked and what
    it sent as the state. A queued exception is raised instead of answered.
    """

    def __init__(self, answers):
        super().__init__()
        self._answers = answers
        self.asked: list[dict] = []

    async def _ask(self, state, questions):
        self.asked.append({"state": state, "questions": dict(questions)})
        answer = (
            self._answers(state, questions) if callable(self._answers) else self._answers.pop(0)
        )
        if isinstance(answer, Exception):
            raise answer
        return answer, None


class _FakeLLMService:
    """Returns a queued answer to every inference, and records the call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.settings = SimpleNamespace(model="fake-model")

    async def run_inference(
        self, context, max_tokens=None, system_instruction=None, response_schema=None
    ) -> str:
        self.calls.append(
            {
                "ask": context._messages[-1]["content"],
                "system_instruction": system_instruction,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("FakeLLMService: no more queued responses")
        return self._responses.pop(0)


def _choice(choice: str, confidence: float = 0.95, options=REPLY_OPTIONS) -> ChoiceResult:
    rest = (1 - confidence) / (len(options) - 1)
    probabilities = {o: (confidence if o == choice else rest) for o in options}
    return ChoiceResult(choice=choice, probabilities=probabilities, confidence=confidence)


def _turn(meets: float | dict) -> ChoiceResult:
    """A turn's answer: its probability of meeting the criterion, or the full distribution."""
    probabilities = (
        meets
        if isinstance(meets, dict)
        else {"meets": meets, "fails": 1 - meets, "not_applicable": 0.0}
    )
    choice = max(probabilities, key=probabilities.get)
    return ChoiceResult(
        choice=choice, probabilities=probabilities, confidence=probabilities[choice]
    )


def _judge(answers, explainer=None, **kwargs) -> tuple[EvalJudge, _FakeClassifier]:
    classifier = _FakeClassifier(answers)
    return EvalJudge(classifier, explainer=explainer, **kwargs), classifier


class TestParseRunVerdicts(unittest.TestCase):
    def test_the_goal_and_a_verdict_per_turn_per_criterion(self):
        out = _Explainer._parse_run_verdicts(
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
        out = _Explainer._parse_run_verdicts(
            '```json\n{"goal": {"verdict": "no"}, "turns": {"Politeness": ["yes"]}}\n```',
            ["politeness", "brevity"],
            2,
        )
        self.assertEqual((out.goal.verdict, out.goal.reason), ("no", "(no reason given)"))
        self.assertEqual([v.verdict for v in out.turns["politeness"]], ["yes", "none"])
        self.assertEqual(out.turns["politeness"][1].reason, "(judge gave no verdict)")
        self.assertEqual([v.reason for v in out.turns["brevity"]], ["(judge gave no verdict)"] * 2)

    def test_no_json_fails_everything(self):
        out = _Explainer._parse_run_verdicts("no json here", ["politeness"], 1)
        self.assertEqual(out.goal.verdict, "none")
        self.assertEqual(out.turns["politeness"][0].verdict, "none")


class TestParseVerdict(unittest.TestCase):
    def test_clean_json_yes(self):
        v = _Explainer._parse_verdict('{"verdict": "yes", "reason": "It mentions weather."}')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "It mentions weather.")

    def test_clean_json_no(self):
        v = _Explainer._parse_verdict('{"verdict": "no", "reason": "Does not mention it."}')
        self.assertFalse(v.passed)
        self.assertEqual(v.verdict, "no")
        self.assertEqual(v.reason, "Does not mention it.")

    def test_clean_json_continue(self):
        v = _Explainer._parse_verdict('{"verdict": "continue", "reason": "Just a filler so far."}')
        self.assertEqual(v.verdict, "continue")
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, "Just a filler so far.")

    def test_unknown_verdict_fails_closed(self):
        v = _Explainer._parse_verdict('{"verdict": "maybe", "reason": "x"}')
        self.assertEqual(v.verdict, "no")

    def test_unstructured_continue_fallback(self):
        v = _Explainer._parse_verdict("continue, more text is needed")
        self.assertEqual(v.verdict, "continue")

    def test_fenced_json(self):
        v = _Explainer._parse_verdict('```json\n{"verdict": "yes", "reason": "ok"}\n```')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "ok")

    def test_fenced_json_without_lang(self):
        v = _Explainer._parse_verdict('```\n{"verdict": "yes", "reason": "ok"}\n```')
        self.assertTrue(v.passed)

    def test_trailing_prose_after_json(self):
        # Models sometimes append chatty text after the JSON object, and "know"
        # contains "no", so a substring-matching fallback would misread the
        # trailing sentence as a rejection.
        v = _Explainer._parse_verdict(
            ' {"verdict": "yes", "reason": "The bot greets the user."}\n\n'
            "Let me know if you'd like to evaluate any further turns!"
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "The bot greets the user.")

    def test_leading_prose_before_json(self):
        v = _Explainer._parse_verdict(
            'Sure, here is my verdict: {"verdict": "no", "reason": "wrong"}'
        )
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, "wrong")

    def test_unstructured_yes_fallback(self):
        v = _Explainer._parse_verdict("yes, this satisfies the criterion")
        self.assertTrue(v.passed)

    def test_unstructured_no_fallback(self):
        v = _Explainer._parse_verdict("no, it does not")
        self.assertFalse(v.passed)

    def test_ambiguous_response_fails_closed(self):
        v = _Explainer._parse_verdict("the answer is yes or possibly no")
        self.assertFalse(v.passed)
        self.assertIn("could not parse", v.reason)

    def test_garbage_response(self):
        v = _Explainer._parse_verdict("???")
        self.assertFalse(v.passed)

    def test_extra_whitespace(self):
        v = _Explainer._parse_verdict('  \n {"verdict": "yes", "reason": "x"}  \n ')
        self.assertTrue(v.passed)

    def test_missing_reason(self):
        v = _Explainer._parse_verdict('{"verdict": "yes"}')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "(no reason given)")


class TestJudgeEvaluate(unittest.IsolatedAsyncioTestCase):
    async def test_the_latest_reply_is_judged_after_the_spoken_conversation(self):
        judge, classifier = _judge([{"verdict": _choice("yes")}])
        judge.add_user_message("What's the weather?")
        judge.add_tool_call("lookup()")
        judge.add_assistant_message("Let me check.")
        judge.add_assistant_message("It's 72 and sunny.")

        verdict = await judge.evaluate("describes the weather")

        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.confidence, 0.95)
        self.assertIn("P(yes)=0.95", verdict.reason)
        state = classifier.asked[0]["state"]
        self.assertEqual(state["latest_bot_reply"], "Let me check. It's 72 and sunny.")
        self.assertEqual(
            state["conversation"], [{"speaker": "user", "text": "What's the weather?"}]
        )
        question = classifier.asked[0]["questions"]["verdict"]
        self.assertEqual(set(question.options), set(REPLY_OPTIONS))
        self.assertIn("describes the weather", question.instructions)

    async def test_without_continue_a_reply_is_judged_yes_or_no(self):
        judge, classifier = _judge([{"verdict": _choice("no")}], allow_continue=False)
        judge.add_user_message("What is the capital of Germany?")
        judge.add_assistant_message("No rush, take your time.")
        verdict = await judge.evaluate("says the capital of Germany is Berlin")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(set(classifier.asked[0]["questions"]["verdict"].options), {"yes", "no"})

    async def test_a_verdict_is_cached_by_criterion_and_conversation(self):
        judge, classifier = _judge([{"verdict": _choice("yes")}])
        judge.add_assistant_message("It rains.")
        await judge.evaluate("mentions weather")
        await judge.evaluate("mentions weather")
        self.assertEqual(len(classifier.asked), 1)

    async def test_a_failed_question_is_asked_again_then_gives_a_no(self):
        judge, classifier = _judge([ClassifierError("boom"), {"verdict": _choice("yes")}])
        judge.add_assistant_message("It rains.")
        self.assertTrue((await judge.evaluate("mentions weather")).passed)
        self.assertEqual(len(classifier.asked), 2)

        judge, classifier = _judge([ClassifierError("boom")] * 2)
        judge.add_assistant_message("anything")
        verdict = await judge.evaluate("anything")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(verdict.reason, "judge call failed")
        self.assertEqual(len(classifier.asked), 2)

    async def test_an_answer_of_the_wrong_type_fails_the_question(self):
        judge, _ = _judge([{"verdict": YesNoResult(probability=0.9)}] * 2)
        judge.add_assistant_message("anything")
        verdict = await judge.evaluate("anything")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(verdict.reason, "judge call failed")


class TestJudgeExplainer(unittest.IsolatedAsyncioTestCase):
    async def test_a_no_takes_the_explainers_reason(self):
        llm = _FakeLLMService(['{"verdict": "no", "reason": "it never mentions rain"}'])
        judge, _ = _judge([{"verdict": _choice("no")}], explainer=llm)
        judge.add_assistant_message("Hello.")
        verdict = await judge.evaluate("mentions weather")
        self.assertEqual(verdict.verdict, "no")
        self.assertTrue(verdict.reason.startswith("it never mentions rain"))
        self.assertIn("P(no)=0.95", verdict.reason)

    async def test_a_disagreeing_explainer_is_noted_and_the_classifier_stands(self):
        llm = _FakeLLMService(['{"verdict": "yes", "reason": "it says it rains"}'])
        judge, _ = _judge([{"verdict": _choice("no")}], explainer=llm)
        judge.add_assistant_message("It rains.")
        verdict = await judge.evaluate("mentions weather")
        self.assertEqual(verdict.verdict, "no")
        self.assertIn("the explainer judged yes: it says it rains", verdict.reason)

    async def test_an_agreeing_explainer_without_a_reason_leaves_the_probabilities(self):
        llm = _FakeLLMService(['{"verdict": "no"}'])
        judge, _ = _judge([{"verdict": _choice("no")}], explainer=llm)
        judge.add_assistant_message("Hello.")
        verdict = await judge.evaluate("mentions weather")
        self.assertTrue(verdict.reason.startswith("P(yes)="))

    async def test_a_sure_yes_and_a_continue_are_not_explained(self):
        llm = _FakeLLMService([])
        judge, _ = _judge(
            [{"verdict": _choice("continue")}, {"verdict": _choice("yes")}],
            explainer=llm,
        )
        judge.add_assistant_message("Let me check.")
        self.assertEqual((await judge.evaluate("gives the weather")).verdict, "continue")
        judge.add_assistant_message("It's sunny.")
        self.assertTrue((await judge.evaluate("gives the weather")).passed)
        self.assertEqual(llm.calls, [])

    async def test_an_unsure_yes_is_explained(self):
        llm = _FakeLLMService(['{"verdict": "yes", "reason": "close enough"}'])
        judge, _ = _judge([{"verdict": _choice("yes", 0.6)}], explainer=llm)
        judge.add_assistant_message("Sunny-ish.")
        verdict = await judge.evaluate("gives the weather")
        self.assertTrue(verdict.passed)
        self.assertTrue(verdict.reason.startswith("close enough"))
        self.assertEqual(len(llm.calls), 1)


class TestJudgeEvaluateCall(unittest.IsolatedAsyncioTestCase):
    async def test_a_call_is_judged_by_name_and_arguments(self):
        judge, classifier = _judge([{"answer": YesNoResult(probability=0.2)}])
        judge.add_user_message("Book six o'clock.")
        verdict = await judge.evaluate_call("book", {"time": "7pm"}, "books six o'clock")
        self.assertEqual(verdict.verdict, "no")
        self.assertAlmostEqual(verdict.confidence, 0.8)
        state = classifier.asked[0]["state"]
        self.assertEqual(state["call"], {"name": "book", "arguments": {"time": "7pm"}})

    async def test_the_same_criterion_on_another_call_is_another_question(self):
        judge, classifier = _judge(
            [{"answer": YesNoResult(probability=0.9)}, {"answer": YesNoResult(probability=0.1)}]
        )
        self.assertTrue((await judge.evaluate_call("submit", {"n": 1}, "n is one")).passed)
        self.assertTrue((await judge.evaluate_call("submit", {"n": 1}, "n is one")).passed)
        self.assertFalse((await judge.evaluate_call("submit", {"n": 2}, "n is one")).passed)
        self.assertEqual(len(classifier.asked), 2)


class TestJudgeEvaluateRun(unittest.IsolatedAsyncioTestCase):
    def _converse(self, judge: EvalJudge) -> None:
        judge.add_user_message("Book a table at six.")
        judge.add_tool_call('book({"time": "6pm"})')
        judge.add_assistant_message("Let me check.")
        judge.add_assistant_message("Done, six o'clock.")
        judge.add_user_message("Thanks.")
        judge.add_assistant_message("You're welcome.")

    @staticmethod
    def _answer(goal: float, turns: dict[str, float | dict], failing: str | None = None):
        """Answers a run's questions: the goal's, and each turn's by its reply text.

        A turn's answer is its probability of meeting the criterion (the rest
        failing it), or the choice's full probabilities.
        """

        def respond(state, questions):
            reply = state.get("latest_bot_reply")
            if reply is None:
                if failing == "goal":
                    raise ClassifierError("goal question failed")
                return {"answer": YesNoResult(probability=goal)}
            if reply == failing:
                raise ClassifierError("turn question failed")
            return {name: _turn(turns[reply]) for name in questions}

        return respond

    async def test_the_goal_is_asked_over_the_whole_conversation(self):
        judge, classifier = _judge(
            self._answer(0.9, {"Let me check. Done, six o'clock.": 0.8, "You're welcome.": 0.3})
        )
        self._converse(judge)

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")

        self.assertTrue(verdicts.goal.passed)
        goal = next(a for a in classifier.asked if "latest_bot_reply" not in a["state"])
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
        judge, classifier = _judge(
            self._answer(0.9, {"Let me check. Done, six o'clock.": 0.8, "You're welcome.": 0.3})
        )
        self._converse(judge)

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")

        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["yes", "no"])
        turns = [a for a in classifier.asked if "latest_bot_reply" in a["state"]]
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
        self.assertEqual(set(by_reply["You're welcome."]["questions"]), {"polite"})

    async def test_a_turn_the_criterion_does_not_apply_to_passes(self):
        judge, classifier = _judge(
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
            a["questions"]["apology"] for a in classifier.asked if "latest_bot_reply" in a["state"]
        )
        self.assertEqual(set(question.options), set(TURN_OPTIONS))

    async def test_a_run_with_no_criteria_asks_only_the_goal(self):
        judge, classifier = _judge(self._answer(0.9, {}))
        self._converse(judge)
        verdicts = await judge.evaluate_run({}, "a table is booked")
        self.assertTrue(verdicts.goal.passed)
        self.assertEqual(len(classifier.asked), 1)

    async def test_a_failed_question_blanks_only_its_own_verdicts(self):
        judge, _ = _judge(
            self._answer(0.9, {"You're welcome.": 0.8}, failing="Let me check. Done, six o'clock.")
        )
        self._converse(judge)
        verdicts = await judge.evaluate_run({"polite": "is polite"}, "booked")
        self.assertTrue(verdicts.goal.passed)
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["none", "yes"])

    async def test_a_failed_goal_question_gives_no_goal_verdict(self):
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
            explainer=llm,
        )
        self._converse(judge)

        verdicts = await judge.evaluate_run({"polite": "is polite"}, "booked")

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(verdicts.goal.reason, "P(yes)=0.95")
        first, second = verdicts.turns["polite"]
        self.assertEqual(first.reason, "P(meets)=0.90, P(fails)=0.10, P(not_applicable)=0.00")
        self.assertTrue(second.reason.startswith("curt"))

    async def test_a_transcript_passed_in_is_deprecated_and_judged_in_place_of_the_kept_one(self):
        judge, classifier = _judge(self._answer(0.9, {"given": 0.9}))
        judge.add_assistant_message("kept")
        given = [{"role": "assistant", "content": "given"}]
        for call in (
            lambda: judge.evaluate_run(given, {"polite": "is polite"}, "done"),
            lambda: judge.evaluate_run({"polite": "is polite"}, "done", transcript=given),
        ):
            with self.assertWarns(DeprecationWarning):
                verdicts = await call()
            self.assertTrue(verdicts.goal.passed)
            replies = [a["state"].get("latest_bot_reply") for a in classifier.asked]
            self.assertIn("given", replies)
            self.assertNotIn("kept", replies)


class TestJudgeOverAnLLM(unittest.IsolatedAsyncioTestCase):
    """The judge over an LLM classifier, with the same LLM explaining."""

    async def test_a_reply_is_classified_in_one_call(self):
        llm = _FakeLLMService(
            [json.dumps({"verdict": {"choice": "yes", "probabilities": {"yes": 0.95}}})]
        )
        judge = EvalJudge(LLMClassifier(llm=llm))
        judge.add_assistant_message("It's 72 and sunny.")
        verdict = await judge.evaluate("describes the weather")
        self.assertTrue(verdict.passed)
        self.assertEqual(len(llm.calls), 1)
        ask = llm.calls[0]["ask"]
        self.assertIn("describes the weather", ask)
        self.assertIn("It's 72 and sunny.", ask)
        for option in REPLY_OPTIONS:
            self.assertIn(f"- {option}:", ask)

    async def test_a_no_is_explained_by_the_explainers_llm(self):
        llm = _FakeLLMService(
            [
                json.dumps({"verdict": {"choice": "no", "probabilities": {"no": 0.9}}}),
                '{"verdict": "no", "reason": "it never mentions the weather"}',
            ]
        )
        judge = EvalJudge(LLMClassifier(llm=llm), explainer=llm)
        judge.add_assistant_message("Hello there.")
        verdict = await judge.evaluate("describes the weather")
        self.assertEqual(verdict.verdict, "no")
        self.assertTrue(verdict.reason.startswith("it never mentions the weather"))
        self.assertEqual(len(llm.calls), 2)

    async def test_without_an_explainer_the_probabilities_are_the_reason(self):
        llm = _FakeLLMService(
            [json.dumps({"verdict": {"choice": "no", "probabilities": {"no": 0.9}}})]
        )
        judge = EvalJudge(LLMClassifier(llm=llm))
        judge.add_assistant_message("Hello there.")
        verdict = await judge.evaluate("describes the weather")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(verdict.reason, "P(yes)=0.00, P(no)=0.90, P(continue)=0.00")
        self.assertEqual(len(llm.calls), 1)


class TestJudgeOverADeprecatedService(unittest.IsolatedAsyncioTestCase):
    """An LLM service in place of a classifier still classifies and explains."""

    async def test_a_service_warns_and_classifies_and_explains_with_itself(self):
        llm = _FakeLLMService(
            [
                json.dumps({"verdict": {"choice": "no", "probabilities": {"no": 0.9}}}),
                '{"verdict": "no", "reason": "it never mentions the weather"}',
            ]
        )
        with self.assertWarns(DeprecationWarning):
            judge = EvalJudge(llm)
        self.assertIsInstance(judge.classifier, LLMClassifier)
        judge.add_assistant_message("Hello there.")
        verdict = await judge.evaluate("describes the weather")
        self.assertTrue(verdict.reason.startswith("it never mentions the weather"))
        self.assertEqual(len(llm.calls), 2)

    async def test_the_service_keyword_warns_and_works(self):
        llm = _FakeLLMService(
            [json.dumps({"verdict": {"choice": "yes", "probabilities": {"yes": 0.95}}})]
        )
        with self.assertWarns(DeprecationWarning):
            judge = EvalJudge(service=llm)
        judge.add_assistant_message("It's sunny.")
        self.assertTrue((await judge.evaluate("describes the weather")).passed)

    def test_a_judge_needs_a_classifier(self):
        with self.assertRaises(ValueError):
            EvalJudge()


class TestJudgeExplainerPrompts(unittest.IsolatedAsyncioTestCase):
    """What the explainer is asked when a verdict needs a reason."""

    async def test_a_reply_is_explained_over_the_conversation(self):
        llm = _FakeLLMService(['{"verdict": "no", "reason": "it never mentions rain"}'])
        judge, _ = _judge([{"verdict": _choice("no")}], explainer=llm)
        judge.add_user_message("What's the weather?")
        judge.add_assistant_message("Hello there.")
        await judge.evaluate("describes the weather")
        ask = llm.calls[0]["ask"]
        self.assertIn("Criterion: describes the weather", ask)
        self.assertIn("Answer yes, no, or continue.", ask)
        self.assertEqual(llm.calls[0]["system_instruction"], JUDGE_SYSTEM_INSTRUCTION)

    async def test_without_continue_the_explainer_is_asked_yes_or_no(self):
        llm = _FakeLLMService(['{"verdict": "no", "reason": "never names Berlin"}'])
        judge, _ = _judge(
            [{"verdict": _choice("no", options=("yes", "no"))}],
            explainer=llm,
            allow_continue=False,
        )
        judge.add_assistant_message("Take your time.")
        await judge.evaluate("says the capital of Germany is Berlin")
        self.assertEqual(llm.calls[0]["system_instruction"], JUDGE_FINAL_SYSTEM_INSTRUCTION)
        self.assertIn("Answer yes or no.", llm.calls[0]["ask"])

    async def test_a_call_is_explained_by_name_and_arguments(self):
        llm = _FakeLLMService(['{"verdict": "no", "reason": "wrong speaker"}'])
        judge, _ = _judge([{"answer": YesNoResult(probability=0.1)}], explainer=llm)
        verdict = await judge.evaluate_call("submit", {"speaker": "Ann"}, "submitted for Bob")
        self.assertTrue(verdict.reason.startswith("wrong speaker"))
        ask = llm.calls[0]["ask"]
        self.assertIn('called the function `submit` with arguments `{"speaker": "Ann"}`', ask)
        self.assertEqual(llm.calls[0]["system_instruction"], JUDGE_CALL_SYSTEM_INSTRUCTION)

    async def test_a_run_is_explained_from_the_whole_transcript(self):
        llm = _FakeLLMService(
            ['{"goal": {"verdict": "no", "reason": "never booked"}, "turns": {"polite": ["yes"]}}']
        )
        judge, _ = _judge(
            lambda state, questions: (
                {"answer": YesNoResult(probability=0.1)}
                if "latest_bot_reply" not in state
                else {name: _turn(0.9) for name in questions}
            ),
            explainer=llm,
        )
        judge.add_user_message("Book a table at six.")
        judge.add_tool_call('book({"time": "6pm"})')
        judge.add_assistant_message("Sure.")
        verdicts = await judge.evaluate_run({"polite": "is polite"}, "a table is booked")
        self.assertEqual(verdicts.goal.verdict, "no")
        self.assertTrue(verdicts.goal.reason.startswith("never booked"))
        ask = llm.calls[0]["ask"]
        self.assertIn("User: Book a table at six.", ask)
        self.assertIn('[tool call] book({"time": "6pm"})', ask)
        self.assertIn("Bot turn 1: Sure.", ask)
        self.assertEqual(llm.calls[0]["system_instruction"], RUN_JUDGE_SYSTEM_INSTRUCTION)

    async def test_max_tokens_caps_the_explainer(self):
        llm = _FakeLLMService(['{"verdict": "no", "reason": "no rain"}'])
        judge, _ = _judge([{"verdict": _choice("no")}], explainer=llm, max_tokens=64)
        judge.add_assistant_message("Hello.")
        await judge.evaluate("describes the weather")
        self.assertEqual(llm.calls[0]["max_tokens"], 64)


class TestJudgeConfig(unittest.IsolatedAsyncioTestCase):
    def test_the_default_judge_classifies_and_explains_with_one_llm(self):
        judge = EvalJudge.from_config(None)
        self.assertIsInstance(judge.classifier, LLMClassifier)
        self.assertIs(judge._explainer.service, judge.classifier.llm)

    def test_an_explainer_block_names_the_llm_that_gives_the_reasons(self):
        judge = EvalJudge.from_config(
            {"service": "ollama", "model": "a", "explainer": {"service": "ollama", "model": "b"}}
        )
        self.assertEqual(judge.classifier.llm.settings.model, "a")
        self.assertEqual(judge._explainer.service.settings.model, "b")

    def test_explainer_false_builds_no_explainer(self):
        judge = EvalJudge.from_config({"service": "ollama", "explainer": False})
        self.assertIsNone(judge._explainer)

    def test_allow_continue_false_is_read_from_the_config(self):
        judge = EvalJudge.from_config({"service": "ollama", "allow_continue": False})
        self.assertFalse(judge._allow_continue)
        self.assertNotIn("continue", judge._reply_outcomes)
        self.assertFalse(judge._explainer._allow_continue)


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
