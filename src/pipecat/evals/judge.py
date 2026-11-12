#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The judge: decides whether the bot did what a scenario asks.

A judge answers a natural-language criterion, an ``eval:`` on a scripted
turn or a simulation's ``success:`` and metrics, about the conversation it
was given. It keeps that conversation, fed as it happens: the user's turns,
the bot's replies, and the tool calls the bot makes. A terse reply ("That's
four") is judged in context, and a whole run is judged over the same record.

How it decides:

Every question goes to a :class:`~pipecat.classifiers.base_classifier.BaseClassifier`,
which answers with an option and a probability for each of the options it
was offered. Two classifiers come with Pipecat:

- :class:`~pipecat.classifiers.llm.classifier.LLMClassifier`, over any
  Pipecat LLM service with ``run_inference()``: OpenAI, Ollama, Together,
  and others. This is the default.
- :class:`~pipecat.classifiers.typesafe.jev.classifier.JevClassifier`, over
  TypeSafe's Jev, a hosted classification model. Jev answers in a few hundred
  milliseconds, costs little, and its probabilities are calibrated, so a
  scenario can tell a sure verdict from a close call.

A scenario picks one in its ``judge.eval:`` block. All the options::

    judge:
      eval:
        service: ollama          # the LLM that classifies (the default), or
        model: gemma4:12b        #   factory: a dotted path to a callable that
                                 #   takes this block and returns a classifier
        explainer:               # optional; the LLM that gives reasons
          service: ollama        #   (the judging LLM if omitted, the default
          model: gemma4:12b      #   LLM for a factory's classifier; none if
                                 #   set to false)
        explain_below: 0.75      # optional; see "The explainer" below
        allow_continue: true     # optional; false judges a reply yes or no only

A factory may also return an LLM service, which the judge then classifies
with.

What the judge asks:

The harness asks three kinds of question, and each is one classifier call:

- A reply, for an ``eval:`` on a scripted turn. The state holds the
  conversation so far and, separately, the bot's latest reply. The answers
  are ``yes`` (the reply meets the criterion), ``no`` (it's a real answer
  that doesn't, including a reply that waits for the user instead), and
  ``continue`` (the bot is still working toward its answer: it only
  greeted, said it's checking, or the reply is still arriving). On
  ``continue`` the harness waits for more of the reply and asks again. A
  suite whose judged replies are all final answers, with nothing for the bot
  to fetch first, sets ``allow_continue: false``, and a reply is then yes or
  no; a reply that is still arriving is still judged again as more of it
  comes.
- A function call, for an ``eval:`` on a ``function_call``. The state holds
  the call's name and arguments and the conversation as context. The
  answer is yes or no.
- A whole simulation, at the end of the run. The goal is one yes/no
  question over the whole conversation, tool calls included. Each bot turn
  is asked about on its own, once per criterion, with only the conversation
  before that turn as context. The answers are ``meets``, ``fails``, and
  ``not_applicable`` (the criterion only covers some situation, like "when
  the time is taken, apologise", and this turn isn't in it); only ``fails``
  fails the turn. All of a run's questions go out at once.

How each question is written:

- The reply or function call being judged is sent separately from the
  conversation before it (``latest_bot_reply`` or ``call``), so the
  classifier knows which part to judge.
- Every possible answer is one of the options. "The bot hasn't answered
  yet" is the option ``continue``, and "this criterion doesn't apply to this
  turn" is the option ``not_applicable``. Don't write these as rules in the
  instructions instead: a classifier follows an instruction for every
  criterion, even where it doesn't fit. A rule like "a criterion about some
  situation passes when that situation doesn't come up" also passes turns
  that never state a price against "the reply states a price".
- The instructions hold only the question and the criterion.

How answers become verdicts:

Each verdict carries a confidence from 0 to 1, and its reason starts out as
the probabilities the classifier gave (for example ``P(yes)=0.97``):

- A reply's verdict is the option chosen, with the confidence in it.
- A yes/no question (a function call, a goal) is ``yes`` when the
  probability of yes is at least 0.5; its confidence is the probability of
  the answer given.
- A turn's confidence is the probability of the verdict given: for a pass,
  the probability of ``meets`` and ``not_applicable`` together.

Each question's verdict is cached by its criterion and state, so asking the
same question about the same conversation twice costs one call.

The explainer:

A classifier gives no reasons, so the judge asks an LLM of its own, the
explainer, for them, in a one-shot inference outside the pipeline with the
prompts in this module. It is asked for:

- every ``no``, on a reply, a function call, a turn or a goal;
- every verdict the classifier is unsure of, meaning a confidence below
  ``explain_below`` (0.75 by default). When a verdict is close to 50/50, the
  next run can go the other way, so these are worth a second look. An LLM
  classifier's probabilities are not calibrated and rarely fall this low, so
  in practice it is the ``no`` verdicts that are explained;
- never a ``continue``, which is only a request to wait for more text.

``explain_below: 0`` limits it to the ``no`` verdicts, and a value above 1
has every verdict explained.

The explainer never decides a verdict. It is told the classifier's and asked
what in the conversation supports it, so the reason points to evidence for the
verdict that counts, not to how the classifier reached it. Its reason takes the
place of the probabilities as the verdict's ``reason``, and the probabilities
stay in ``probabilities``. For a simulation, every verdict of the run that needs
a reason is explained in one call. Each question names the reasons it wants in
a response schema, so an LLM that can enforce one answers in exactly that
shape, and a question whose call fails or times out, or whose answer leaves a
reason out, is asked once more. A verdict still without a reason keeps the
probabilities.

When a call fails:

A question that times out, can't connect, or comes back unusable is asked
once more. A second failure fails the question: a reply or a function call
gets a ``no`` with the reason "judge call failed", and each verdict of a
failed simulation question is a ``none`` (the verdict the harness reports as
not given).

Lifecycle:

The judge sets its classifier up before its first question and cleans it up
when the session closes it at the end of the run, so a classifier that keeps
a connection opens it once and a simulation's questions share it.

Example::

    from pipecat.classifiers.llm.classifier import LLMClassifier
    from pipecat.services.ollama.llm import OLLamaLLMService

    service = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="gemma4:12b"))
    judge = EvalJudge(LLMClassifier(llm=service), explainer=service)
    judge.add_user_message("What can you help me with?")
    judge.add_assistant_message("I can answer questions, set reminders, and look things up.")
    verdict = await judge.evaluate("describes the bot's capabilities")
    if not verdict.passed:
        print(f"judge said no: {verdict.reason}")
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar, cast

from loguru import logger

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.evals.services import classifier_from_config, llm_service_from_config
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.deprecation import warn_deprecated

_R = TypeVar("_R")

# The explainer's instructions, for every question it is asked. The classifier
# has decided: the explainer is told the verdict and says what in the
# conversation supports it, so a reason always explains the verdict that counts.
EXPLAIN_SYSTEM_INSTRUCTION = (
    "You explain the verdicts a judge gave on a conversation between a user and a bot "
    "under test. The 'user' messages are the user; the 'assistant' messages are the bot's "
    "replies. The judge's verdicts are final. For each verdict you are asked about, say in "
    "one short sentence what the bot said or did that makes it right, naming it; never "
    "just restate the verdict, and never argue for a different verdict. When a criterion "
    "does not apply to a reply, say what the reply does instead and why that is outside "
    "what the criterion covers. "
    "When the bot spoke its replies, its text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, "
    "and missing punctuation; read it by its intended spoken meaning. "
    "Respond ONLY with the JSON object the question describes, on a single line, with no "
    "other text or markdown."
)

# The ask for a reply's verdict, after the conversation so far.
EXPLAIN_REPLY_ASK_TEMPLATE = (
    "The judge's verdict: the bot's most recent reply {outcome} this criterion.\n\n"
    "Criterion: {criterion}\n\n"
    'Answer with {{"reason": "<one short sentence>"}}.'
)

# The ask for a function call's verdict; the conversation so far is context.
EXPLAIN_CALL_ASK_TEMPLATE = (
    "The bot called the function `{name}` with arguments `{args}`. The judge's verdict: "
    "this call {outcome} this criterion.\n\n"
    "Criterion: {criterion}\n\n"
    'Answer with {{"reason": "<one short sentence>"}}.'
)

# The ask for a simulation run's verdicts: the transcript, with the bot's turns
# numbered and its tool calls inline, what the run was judged on, and the
# verdicts to explain, one per line.
EXPLAIN_RUN_ASK_TEMPLATE = (
    "Transcript:\n{transcript}\n\n"
    "A line marked '[tool call]' is a function the bot called at that point.\n\n"
    "Criteria every bot reply was judged on:\n{criteria}\n\n"
    "Goal the conversation as a whole was judged on: {success}\n\n"
    "The judge's verdicts to explain:\n{verdicts}\n\n"
    'Answer with {{"goal": "<reason>", "turns": {{"<criterion name>": '
    '{{"<bot turn number>": "<reason>"}}}}}}, a reason for each verdict listed and '
    "nothing else."
)


@dataclass
class JudgeVerdict:
    """Outcome of a single judge call.

    Parameters:
        verdict: ``"yes"`` (satisfies), ``"no"`` (substantive answer that fails),
            or ``"continue"`` (interim/filler/incomplete — re-judge once more text
            arrives).
        reason: The explainer's one-sentence reason for the verdict, or the
            classifier's probabilities when nothing explained it.
        raw_response: The judge's raw answer, for diagnostics.
        confidence: How sure the judge is of the verdict, from 0 to 1.
        probabilities: The probability of each outcome the classifier was
            offered; empty when the question failed.
    """

    verdict: str
    reason: str
    raw_response: str
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True only when the verdict is a definite ``"yes"``."""
        return self.verdict == "yes"


@dataclass
class RunVerdicts:
    """A whole simulation run's verdicts.

    Parameters:
        goal: The verdict on the goal, over the whole conversation.
        turns: Per criterion name, a verdict per bot turn, in order.
    """

    goal: JudgeVerdict
    turns: dict[str, list[JudgeVerdict]]


# The questions the judge asks, each followed by the criterion and the notes
# that go with every question; see ``EvalJudge._instructions``.
_REPLY_QUESTION = (
    "Does `latest_bot_reply`, the bot's most recent reply, following `conversation`, satisfy "
    "this criterion?"
)
_CALL_QUESTION = (
    "Does the bot's function `call`, judged by its name and arguments, satisfy this criterion?"
)
_CALL_NOTE = "`conversation` is context only."
_GOAL_QUESTION = "Does the conversation as a whole achieve this goal?"
_GOAL_NOTE = (
    "A `tool` entry is a function the bot called at that point; a completed call is stronger "
    "evidence of an action than the bot saying it did it."
)
_TURN_QUESTION = (
    "`latest_bot_reply` is the bot's reply following `conversation`. How does it stand against "
    "this criterion?"
)
_TRANSCRIPTION_NOTE = (
    "The bot's text may be an automatic speech-to-text transcription: judge its intended "
    "spoken meaning, never its spelling ('for' may mean 'four', 'to' may mean 'two')."
)

_REPLY_OUTCOMES: dict[str, Any] = {
    "yes": "The bot has given its answer, and the answer satisfies the criterion.",
    "no": (
        "The bot has given its answer, and the answer does not satisfy the criterion. A reply "
        "that waits for the user (asking them to take their time or to go on) instead of "
        "giving what the criterion asks for is a no."
    ),
    "continue": (
        "The bot is still working toward its answer: it only greets, says it is checking or "
        "looking something up and will report back, or the reply is an obviously incomplete "
        "fragment."
    ),
}

# A run's per-turn outcomes. A classifier can't tell by itself whether a
# criterion only applies in some situation ("when the time is taken,
# apologises"), so "doesn't apply" is an outcome of its own rather than a rule
# in the instructions.
_TURN_OUTCOMES: dict[str, Any] = {
    "meets": "The reply does what the criterion asks.",
    "fails": "The criterion applies to this reply, and the reply does not do what it asks.",
    "not_applicable": (
        "The criterion only asks something of replies in a particular situation (it says "
        "'when', 'if', or similar), and that situation does not arise in this reply."
    ),
}


class EvalJudge:
    """Judges a conversation with a classifier, and asks an LLM for the reasons.

    Args:
        classifier: The classifier that decides the verdicts. The judge sets
            it up and cleans it up with itself.
        explainer: The LLM asked for the reason behind a ``no`` or an unsure
            verdict; ``None`` reports the probabilities alone.
        explain_below: A ``yes`` less sure than this is explained too.
        allow_continue: Whether a reply may be judged ``continue`` (the bot is
            still working toward its answer, so the harness waits for more of
            it). A suite whose judged replies are all final answers turns it
            off, and a reply is then ``yes`` or ``no``.
        max_tokens: Cap on the explainer's response length for one reason.
            Default 200, enough for a short sentence.
    """

    def __init__(
        self,
        classifier: BaseClassifier | LLMService[Any] | None = None,
        *,
        explainer: LLMService[Any] | None = None,
        explain_below: float = 0.75,
        allow_continue: bool = True,
        max_tokens: int = 200,
        service: LLMService[Any] | None = None,
    ):
        """Initialize the judge with an empty conversation.

        Args:
            classifier: The classifier that decides the verdicts. An LLM
                service is still accepted here, deprecated like ``service``:
                it classifies through an
                :class:`~pipecat.classifiers.llm.classifier.LLMClassifier` and
                explains with itself.
            explainer: The LLM asked for the reason behind a verdict.
            explain_below: A ``yes`` less sure than this is explained too.
            allow_continue: Whether a reply may be judged ``continue``.
            max_tokens: Cap on the explainer's response length for one reason.
            service: The LLM service that classifies and explains, in place of
                ``classifier``.

                .. deprecated:: 1.12.0
                    Use :class:`~pipecat.classifiers.llm.classifier.LLMClassifier`
                    over the service as the ``classifier``, passing the service
                    as the ``explainer`` for the reasons. Will be removed in
                    2.0.0.

        Raises:
            ValueError: If no classifier is given.
        """
        classifier = classifier if classifier is not None else service
        if classifier is None:
            raise ValueError("EvalJudge needs a classifier to decide the verdicts")
        if not isinstance(classifier, BaseClassifier):
            warn_deprecated(
                "`EvalJudge(llm_service)` is deprecated since 1.12.0 and will be "
                "removed in 2.0.0. Use `EvalJudge(LLMClassifier(llm=service))` instead. Pass "
                "the service itself as the `explainer` for the reasons.",
                stacklevel=2,
            )
            service = cast("LLMService[Any]", classifier)
            classifier = LLMClassifier(llm=service)
            explainer = explainer if explainer is not None else service
        self._classifier = classifier
        self._explainer = (
            _Explainer(explainer, max_tokens=max_tokens) if explainer is not None else None
        )
        self._explain_below = explain_below
        self._reply_outcomes = (
            _REPLY_OUTCOMES
            if allow_continue
            else {k: v for k, v in _REPLY_OUTCOMES.items() if k != "continue"}
        )
        # The conversation the judge evaluates against, grown by the harness over
        # the scenario: dicts with a ``role`` of ``user``, ``assistant`` (a
        # segment of a reply), or ``tool`` (a call the bot made, one line), and
        # the ``content``.
        self._transcript: list[dict] = []
        self._cache: dict[str, JudgeVerdict] = {}
        self._run_cache: dict[str, RunVerdicts] = {}
        self._setup_task: asyncio.Task | None = None

    @classmethod
    def from_config(cls, judge_config: dict | None) -> "EvalJudge":
        """Build a judge from a scenario's ``judge.eval:`` block.

        The block names an LLM, ``ollama`` by default, and the judge classifies
        and explains with it. A ``factory`` (a dotted path to a callable taking
        the config) builds the classifier instead, or an LLM service to
        classify with; the explainer is then the ``explainer:`` block's LLM, or
        the default LLM when the block names none. For a fully custom judge,
        construct ``EvalJudge`` directly and pass it to the session.

        Args:
            judge_config: The ``judge.eval:`` block, with the keys the module
                docstring lists. ``None`` uses all defaults.

        Returns:
            A configured EvalJudge.

        Raises:
            ValueError: If the block names an unknown service, or its factory
                returned neither a classifier nor an LLM service.

        Example::

            # In the scenario: judge.eval.factory: "my_pkg.make_judge"
            def make_judge(config):
                return JevClassifier(api_key=...)  # or any LLM service with run_inference()
        """
        config = judge_config or {}
        classifier = classifier_from_config(config, where="judge.eval")
        explainer_config = config.get("explainer")
        if explainer_config is False:
            explainer = None
        elif explainer_config:
            explainer = llm_service_from_config(explainer_config, where="judge.eval.explainer")
        elif isinstance(classifier, LLMClassifier):
            # Without a block of its own, the explainer is the judging LLM.
            explainer = classifier.llm
        else:
            explainer = llm_service_from_config(None, where="judge.eval.explainer")
        return cls(
            classifier,
            explainer=explainer,
            explain_below=float(config.get("explain_below", 0.75)),
            allow_continue=config.get("allow_continue", True) is not False,
        )

    @property
    def classifier(self) -> BaseClassifier:
        """The classifier that decides the verdicts."""
        return self._classifier

    def add_user_message(self, text: str | None) -> None:
        """Record a user turn, so a later reply is judged in context.

        Args:
            text: The user's utterance, or ``None`` for a bot-first turn (ignored).
        """
        if text and text.strip():
            self._transcript.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str | None) -> None:
        """Add a segment of the bot's current reply to the conversation the judge sees.

        Consecutive segments are one reply: a judged run counts them as one
        bot turn.

        Args:
            text: The new reply segment; empty or ``None`` is ignored.
        """
        if text and text.strip():
            self._transcript.append({"role": "assistant", "content": text})

    def add_tool_call(self, text: str | None) -> None:
        """Record a tool call the bot made, as evidence for a judged run.

        Args:
            text: The call on one line, e.g. ``book({"time": "6pm"})`` or
                ``book was cancelled``; empty or ``None`` is ignored.
        """
        if text and text.strip():
            self._transcript.append({"role": "tool", "content": text})

    async def evaluate(self, criterion: str) -> JudgeVerdict:
        """Judge whether the bot's latest reply satisfies ``criterion``, in the conversation so far.

        Args:
            criterion: Natural-language description of what the reply should express.

        Returns:
            A ``yes``, a ``no``, or, when ``allow_continue`` is on, a
            ``continue`` when the bot is still working toward its answer,
            cached by criterion and conversation. A final verdict that needs a
            reason is explained, and a failed call is a ``no`` with the failure
            as its reason.
        """
        entries = self._numbered_turns(e for e in self._transcript if e["role"] != "tool")
        latest = entries.pop()["content"] if entries and entries[-1]["role"] == "bot" else ""
        state = {"conversation": self._conversation(entries), "latest_bot_reply": latest}
        key = _cache_key("reply", criterion, state)
        if key not in self._cache:
            question = ChoiceQuestion(
                instructions=self._instructions(_REPLY_QUESTION, "Criterion", criterion),
                options=self._reply_outcomes,
            )
            answers = await self._choices(state, {"verdict": question})
            if answers is None:
                verdict = self._failed("no")
            else:
                verdict = self._reply_verdict(answers["verdict"])
                if (
                    verdict.verdict != "continue"
                    and self._explainer
                    and self._needs_reason(verdict)
                ):
                    reason = await self._explainer.explain(self._transcript, criterion, verdict)
                    verdict = self._with_reason(verdict, reason)
            self._cache[key] = verdict
        return self._cache[key]

    async def evaluate_call(self, name: str, args: dict | None, criterion: str) -> JudgeVerdict:
        """Judge whether a function call the bot made satisfies ``criterion``, in the conversation so far.

        Args:
            name: The function's name.
            args: The call's arguments.
            criterion: Natural-language description of what the call should be.

        Returns:
            A ``yes`` or a ``no``, cached by call, criterion and conversation,
            and explained when it needs a reason.
        """
        entries = self._numbered_turns(e for e in self._transcript if e["role"] != "tool")
        state = {
            "conversation": self._conversation(entries),
            "call": {"name": name, "arguments": args or {}},
        }
        key = _cache_key("call", criterion, state)
        if key not in self._cache:
            answer = await self._yes_no(
                state, self._instructions(_CALL_QUESTION, "Criterion", criterion, _CALL_NOTE)
            )
            if answer is None:
                verdict = self._failed("no")
            else:
                verdict = self._yes_no_verdict(answer)
                if self._explainer and self._needs_reason(verdict):
                    reason = await self._explainer.explain_call(
                        self._transcript, name, args, criterion, verdict
                    )
                    verdict = self._with_reason(verdict, reason)
            self._cache[key] = verdict
        return self._cache[key]

    async def evaluate_run(
        self,
        criteria: dict[str, str],
        success: str,
        transcript: Sequence[dict] | None = None,
    ) -> RunVerdicts:
        """Judge the whole conversation: every bot turn on every criterion, and the goal.

        The goal is asked over the whole conversation. Each bot turn is its
        own call, judged in the light of the conversation before it, with the
        turn itself as ``latest_bot_reply``; the calls run concurrently. When
        any verdict needs a reason, the explainer judges the run once and its
        reasons go with the classifier's verdicts.

        Args:
            criteria: The per-turn criteria to decide, by name.
            success: The goal criterion, decided over the whole conversation.
            transcript: A conversation to judge in place of the one the judge
                kept: dicts with a ``role`` of ``user``, ``assistant`` or
                ``tool`` and the ``content``. Also accepted first, before
                ``criteria`` and ``success``.

                .. deprecated:: 1.11.0
                    Feed the judge with :meth:`add_user_message`,
                    :meth:`add_assistant_message` and :meth:`add_tool_call`
                    instead. Will be removed in 2.0.0.

        Returns:
            The goal's verdict and, per criterion, a verdict per bot turn in
            order. A failed call is a ``none`` for each verdict it asked for.
        """
        if not isinstance(criteria, dict):
            # The (transcript, criteria, success) order of the deprecated form.
            transcript, criteria, success = (
                cast("Sequence[dict]", criteria),
                cast("dict[str, str]", success),
                cast("str", transcript),
            )
        if transcript is not None:
            warn_deprecated(
                "`EvalJudge.evaluate_run(transcript=...)` is deprecated since 1.11.0 and will "
                "be removed in 2.0.0. Use `add_user_message`, `add_assistant_message` and "
                "`add_tool_call` instead.",
                stacklevel=2,
            )
        conversation = list(self._transcript if transcript is None else transcript)
        entries = self._numbered_turns(conversation)
        names = list(criteria)
        key = _cache_key("run", criteria, success, entries)
        if key in self._run_cache:
            return self._run_cache[key]

        goal_instructions = self._instructions(_GOAL_QUESTION, "Goal", success, _GOAL_NOTE)
        turn_questions = {
            name: ChoiceQuestion(
                instructions=self._instructions(_TURN_QUESTION, "Criterion", criteria[name]),
                options=_TURN_OUTCOMES,
            )
            for name in names
        }
        turn_states = [
            {"conversation": self._conversation(entries[:i]), "latest_bot_reply": entry["content"]}
            for i, entry in enumerate(entries)
            if entry["role"] == "bot"
        ]

        # The goal, and every criterion on each turn: one call each, all at once.
        goal_answer, turn_answers = await asyncio.gather(
            self._yes_no({"conversation": self._conversation(entries)}, goal_instructions),
            asyncio.gather(
                *(
                    self._choices(state, turn_questions)
                    for state in (turn_states if turn_questions else [])
                )
            ),
        )

        failed = self._failed("none")
        verdicts = RunVerdicts(
            goal=self._yes_no_verdict(goal_answer) if goal_answer else failed,
            turns={
                name: [
                    self._turn_verdict(answers[name]) if answers else failed
                    for answers in turn_answers
                ]
                for name in names
            },
        )
        verdicts = await self._explain_run(verdicts, conversation, criteria, success)
        self._run_cache[key] = verdicts
        return verdicts

    async def close(self) -> None:
        """Release what the judge holds open; the session calls it when the run ends."""
        if self._setup_task is not None:
            await self._setup_task
        await self._classifier.cleanup()

    async def _setup(self) -> None:
        """Wire the classifier up, before the first question.

        The judge runs outside any pipeline, so it gives the classifier a task
        manager of its own. A classifier that cannot be set up is only a
        warning: the question goes out regardless.
        """
        try:
            await self._classifier.setup(TaskManager())
        except Exception as e:
            logger.warning(f"Judge couldn't set its classifier up: {e}")

    def _needs_reason(self, verdict: JudgeVerdict) -> bool:
        """Whether a verdict is worth explaining: a ``no``, or an unsure ``yes``."""
        unsure = verdict.confidence is not None and verdict.confidence < self._explain_below
        return verdict.verdict == "no" or unsure

    async def _explain_run(
        self,
        verdicts: RunVerdicts,
        transcript: Sequence[dict],
        criteria: dict[str, str],
        success: str,
    ) -> RunVerdicts:
        """The run's verdicts, those that need a reason given the explainer's; it explains them in one call."""
        goal = verdicts.goal if self._needs_reason(verdicts.goal) else None
        turns = {
            name: {turn: v for turn, v in enumerate(turn_verdicts, 1) if self._needs_reason(v)}
            for name, turn_verdicts in verdicts.turns.items()
        }
        if self._explainer is None or (goal is None and not any(turns.values())):
            return verdicts
        reasons = await self._explainer.explain_run(transcript, criteria, success, goal, turns)
        return RunVerdicts(
            goal=self._with_reason(verdicts.goal, reasons.goal),
            turns={
                name: [
                    self._with_reason(v, reasons.turns.get(name, {}).get(turn, ""))
                    for turn, v in enumerate(turn_verdicts, 1)
                ]
                for name, turn_verdicts in verdicts.turns.items()
            },
        )

    async def _yes_no(self, state, instructions: str) -> YesNoResult | None:
        """The probability of yes, or ``None`` when the question failed."""
        answers = await self._ask(
            state,
            lambda: self._classifier.yes_no(
                state, {"answer": YesNoQuestion(instructions=instructions)}
            ),
        )
        return answers["answer"] if answers else None

    async def _choices(
        self, state, questions: Mapping[str, ChoiceQuestion]
    ) -> dict[str, ChoiceResult] | None:
        """The option chosen for each question, or ``None`` when they failed.

        The questions share the state, so they go in one call.
        """
        return await self._ask(state, lambda: self._classifier.choice(state, questions))

    async def _ask(self, state, ask: Callable[[], Awaitable[_R]]) -> "_R | None":
        """The classifier's answer, asked once more if it failed, or ``None``."""
        # The first question sets the classifier up; the others wait for it.
        if self._setup_task is None:
            self._setup_task = asyncio.get_running_loop().create_task(self._setup())
        await asyncio.shield(self._setup_task)
        logger.debug(f"Judge asking over state:\n{json.dumps(state)}")
        for attempt in (1, 2):
            try:
                answers = await ask()
                logger.debug(f"Judge answered: {answers!r}")
                return answers
            except ClassifierError as e:
                if attempt == 1:
                    logger.warning(f"Judge question failed, asking again: {e}")
                else:
                    logger.error(f"Judge question failed: {e}")
        return None

    @staticmethod
    def _numbered_turns(transcript: Iterable[dict]) -> list[dict]:
        """The conversation with each bot reply joined into one numbered turn.

        A bot turn is a run of reply segments with nothing else between them.

        Args:
            transcript: The judge's conversation entries.

        Returns:
            The entries, each a dict with a ``role`` of ``user``, ``bot`` or
            ``tool`` and the ``content``, a bot entry also carrying its ``turn``
            (from 1).
        """
        entries: list[dict] = []
        turn = 0
        for entry in transcript:
            if entry["role"] == "assistant":
                if entries and entries[-1]["role"] == "bot":
                    entries[-1]["content"] += f" {entry['content']}"
                    continue
                turn += 1
                entries.append({"role": "bot", "turn": turn, "content": entry["content"]})
            else:
                entries.append({"role": entry["role"], "content": entry["content"]})
        return entries

    @staticmethod
    def _conversation(entries: list[dict]) -> list[dict]:
        """The conversation as the state carries it: a speaker, the text, and a bot turn's number."""
        return [
            {
                "speaker": e["role"],
                **({"turn": e["turn"]} if "turn" in e else {}),
                "text": e["content"],
            }
            for e in entries
        ]

    @staticmethod
    def _instructions(question: str, label: str, criterion: str, note: str = "") -> str:
        """A question's instructions: the question, the labelled criterion, and the notes.

        The criterion gets its final punctuation so the note after it reads as a
        new sentence, and every question carries the transcription note.
        """
        criterion = criterion.strip()
        if not criterion.endswith((".", "!", "?")):
            criterion += "."
        parts = [question, f"{label}: {criterion}", note, _TRANSCRIPTION_NOTE]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _reply_verdict(answer: ChoiceResult) -> JudgeVerdict:
        """A reply's verdict: the choice of ``yes``, ``no`` or ``continue``."""
        return JudgeVerdict(
            verdict=answer.choice,
            reason=EvalJudge._probabilities(answer.probabilities),
            raw_response=answer.model_dump_json(),
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
        )

    @staticmethod
    def _yes_no_verdict(answer: YesNoResult) -> JudgeVerdict:
        """A yes/no verdict from the probability of yes."""
        probability = answer.probability
        return JudgeVerdict(
            verdict="yes" if answer.is_yes else "no",
            reason=f"P(yes)={probability:.2f}",
            raw_response=answer.model_dump_json(),
            confidence=max(probability, 1 - probability),
            probabilities={"yes": probability, "no": 1 - probability},
        )

    @staticmethod
    def _turn_verdict(answer: ChoiceResult) -> JudgeVerdict:
        """A turn's verdict: a ``no`` only when the criterion applies to the turn and fails.

        Its confidence is the probability of the verdict given: for a pass, the
        probability of either outcome that passes.
        """
        fails = answer.probabilities.get("fails", 0.0)
        passed = answer.choice != "fails"
        return JudgeVerdict(
            verdict="yes" if passed else "no",
            reason=EvalJudge._probabilities(answer.probabilities),
            raw_response=answer.model_dump_json(),
            confidence=1 - fails if passed else fails,
            probabilities=dict(answer.probabilities),
        )

    @staticmethod
    def _probabilities(probabilities: dict[str, float]) -> str:
        """The probability of each outcome, as a verdict's reason."""
        return ", ".join(f"P({k})={v:.2f}" for k, v in probabilities.items())

    @staticmethod
    def _failed(verdict: str) -> JudgeVerdict:
        """The verdict a failed call gives."""
        return JudgeVerdict(verdict=verdict, reason="judge call failed", raw_response="")

    @staticmethod
    def _with_reason(verdict: JudgeVerdict, reason: str) -> JudgeVerdict:
        """The verdict with the explainer's reason in place of the probabilities, when it gave one."""
        return replace(verdict, reason=reason) if reason else verdict


# How long the explainer has to answer a question before it is asked again.
_EXPLAINER_TIMEOUT_S = 60.0


@dataclass
class _RunReasons:
    """The explainer's reasons for a run's verdicts.

    Parameters:
        goal: The goal verdict's reason; empty when it was not asked or not given.
        turns: Per criterion name, the reason for each turn asked about, by turn number.
    """

    goal: str
    turns: dict[str, dict[int, str]]


class _Explainer:
    """Asks an LLM why the classifier gave its verdicts, telling it each verdict.

    Each question names the reasons it wants in a response schema, so a
    service that can enforce one answers in exactly that shape. A question
    whose call fails, or whose answer lacks a reason it asked for, is asked
    once more. Its answers are cached by question and conversation, so
    explaining the same verdict twice costs one call. A reason is empty when
    it was still not given.

    Args:
        service: A pipecat LLM service with a ``run_inference()`` method
            (i.e. ``BaseOpenAILLMService`` or any subclass: OpenAI, Ollama, etc.).
        max_tokens: Cap on the explainer's response length for one reason.
            Default 200, enough for a short sentence in JSON.
    """

    def __init__(self, service: LLMService[Any], *, max_tokens: int = 200):
        """Initialize the explainer with a configured pipecat LLM service.

        Args:
            service: A pipecat LLM service exposing ``run_inference()``.
            max_tokens: Cap on the explainer's response length for one reason.
        """
        self._service = service
        self._max_tokens = max_tokens
        self._cache: dict[str, str] = {}
        self._run_cache: dict[str, _RunReasons] = {}

    @property
    def service(self) -> LLMService[Any]:
        """The LLM service that answers."""
        return self._service

    async def explain(
        self, transcript: Sequence[dict], criterion: str, verdict: JudgeVerdict
    ) -> str:
        """Why the bot's latest reply got ``verdict`` on ``criterion``.

        Args:
            transcript: The conversation so far, as the judge kept it.
            criterion: Natural-language description of what the reply should express.
            verdict: The classifier's ``yes`` or ``no``.

        Returns:
            A one-sentence reason, or ``""`` when the call failed.
        """
        ask = EXPLAIN_REPLY_ASK_TEMPLATE.format(
            outcome=self._satisfies(verdict), criterion=criterion
        )
        return await self._reason(transcript, criterion, ask)

    async def explain_call(
        self,
        transcript: Sequence[dict],
        name: str,
        args: dict | None,
        criterion: str,
        verdict: JudgeVerdict,
    ) -> str:
        """Why a function call the bot made got ``verdict`` on ``criterion``.

        Args:
            transcript: The conversation so far, as the judge kept it.
            name: The function's name.
            args: The call's arguments, shown to the explainer as JSON.
            criterion: Natural-language description of what the call should be.
            verdict: The classifier's ``yes`` or ``no``.

        Returns:
            A one-sentence reason, or ``""`` when the call failed.
        """
        ask = EXPLAIN_CALL_ASK_TEMPLATE.format(
            name=name,
            args=json.dumps(args or {}, ensure_ascii=False),
            outcome=self._satisfies(verdict),
            criterion=criterion,
        )
        return await self._reason(transcript, criterion, ask)

    async def explain_run(
        self,
        transcript: Sequence[dict],
        criteria: dict[str, str],
        success: str,
        goal: JudgeVerdict | None,
        turns: dict[str, dict[int, JudgeVerdict]],
    ) -> _RunReasons:
        """Why a whole conversation's verdicts were given, in one call.

        The conversation goes in the question, bot turns numbered and tool
        calls inline. A bot turn is a run of reply segments with nothing else
        between them.

        Args:
            transcript: The conversation, as the judge kept it.
            criteria: The per-turn criteria the turns were judged on, by name.
            success: The goal criterion the whole conversation was judged on.
            goal: The goal's verdict to explain, or ``None``.
            turns: Per criterion name, the turn verdicts to explain, by turn number.

        Returns:
            The reasons for the verdicts asked about; a reason not given is empty.
        """
        lines: list[str] = []
        turn = 0
        for entry in transcript:
            if entry["role"] == "assistant":
                if lines and lines[-1].startswith(f"Bot turn {turn}:"):
                    lines[-1] += f" {entry['content']}"
                    continue
                turn += 1
                lines.append(f"Bot turn {turn}: {entry['content']}")
            elif entry["role"] == "tool":
                lines.append(f"[tool call] {entry['content']}")
            else:
                lines.append(f"User: {entry['content']}")
        verdicts = []
        if goal is not None:
            verdicts.append(f"- goal: {'achieved' if goal.passed else 'not achieved'}")
        for name, by_turn in turns.items():
            for number, verdict in by_turn.items():
                verdicts.append(f"- {name}, bot turn {number}: {self._turn_outcome(verdict)}")
        listed = "\n".join(f"- {name}: {criterion}" for name, criterion in criteria.items())
        ask = EXPLAIN_RUN_ASK_TEMPLATE.format(
            transcript="\n".join(lines) or "(nothing was said)",
            criteria=listed or "(none)",
            success=success,
            verdicts="\n".join(verdicts),
        )
        key = _cache_key(ask, [])
        if key not in self._run_cache:
            # Room for a reason per verdict; a budget sized for one cuts it short.
            budget = max(300, 50 * len(verdicts) + 80)
            schema = self._run_schema(goal, turns)
            reasons = _RunReasons(goal="", turns={})
            for attempt in (1, 2):
                response = await self._ask(success, [], ask, schema, max_tokens=budget)
                reasons = self._filled(reasons, self._parse_run_reasons(response or "", turns))
                given = [reasons.goal] if goal is not None else []
                given += [
                    reason for by_turn in reasons.turns.values() for reason in by_turn.values()
                ]
                if all(given):
                    break
                if attempt == 1:
                    logger.warning("Explainer left reasons out, asking again")
            self._run_cache[key] = reasons
        return self._run_cache[key]

    async def _reason(self, transcript: Sequence[dict], criterion: str, ask: str) -> str:
        """The reason the explainer gives to ``ask`` after the spoken conversation."""
        messages = [e for e in transcript if e["role"] != "tool"]
        key = _cache_key(ask, messages)
        if key not in self._cache:
            schema = self._object({"reason": {"type": "string"}})
            reason = ""
            for _ in (1, 2):
                response = await self._ask(criterion, messages, ask, schema)
                reason = self._parse_reason(response) if response else ""
                if reason:
                    break
            self._cache[key] = reason
        return self._cache[key]

    async def _ask(
        self,
        criterion: str,
        messages: list,
        ask: str,
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> str | None:
        """The explainer's raw answer to ``ask``, in ``schema``'s shape, or ``None`` when the call failed or said nothing."""
        # Copy the conversation and append the transient ask, so neither the ask
        # nor the answer ever lands in the persistent context.
        context = LLMContext(messages=list(messages))
        context.add_message({"role": "user", "content": ask})

        # Log the conversation the explainer is about to read, before its answer,
        # so the debug log shows exactly what it saw. A run-level ask carries the
        # transcript itself, so that is what to show.
        transcript = "\n".join(f"  [{m.get('role')}] {m.get('content')}" for m in messages)
        logger.debug(
            "Explainer explaining {!r} over conversation:\n{}",
            criterion,
            transcript or "\n".join(f"  {line}" for line in ask.splitlines()),
        )

        try:
            response = await asyncio.wait_for(
                self._service.run_inference(
                    context=context,
                    max_tokens=self._max_tokens if max_tokens is None else max_tokens,
                    system_instruction=EXPLAIN_SYSTEM_INSTRUCTION,
                    response_schema=schema,
                ),
                _EXPLAINER_TIMEOUT_S,
            )
        except Exception as e:
            logger.error(f"Explainer call failed: {e.__class__.__name__} ({e})")
            return None
        if not response:
            logger.error("Explainer returned an empty response")
            return None
        logger.debug(f"Explainer answered: {response}")
        return response

    @staticmethod
    def _filled(reasons: _RunReasons, answer: _RunReasons) -> _RunReasons:
        """``reasons``, with the ones it lacks taken from ``answer``."""
        return _RunReasons(
            goal=reasons.goal or answer.goal,
            turns={
                name: {
                    number: reasons.turns.get(name, {}).get(number) or reason
                    for number, reason in by_turn.items()
                }
                for name, by_turn in answer.turns.items()
            },
        )

    @staticmethod
    def _object(properties: dict[str, Any]) -> dict[str, Any]:
        """A schema for an object with exactly these properties, all required."""
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    @staticmethod
    def _run_schema(
        goal: JudgeVerdict | None, turns: dict[str, dict[int, JudgeVerdict]]
    ) -> dict[str, Any]:
        """The run answer's schema: the goal's reason when it is asked, and each asked turn's by its number."""
        reason = {"type": "string"}
        properties: dict[str, Any] = {"goal": reason} if goal is not None else {}
        asked = {name: by_turn for name, by_turn in turns.items() if by_turn}
        if asked:
            properties["turns"] = _Explainer._object(
                {
                    name: _Explainer._object({str(number): reason for number in by_turn})
                    for name, by_turn in asked.items()
                }
            )
        return _Explainer._object(properties)

    @staticmethod
    def _satisfies(verdict: JudgeVerdict) -> str:
        """A yes or a no as the ask words it."""
        return "satisfies" if verdict.passed else "does not satisfy"

    @staticmethod
    def _turn_outcome(verdict: JudgeVerdict) -> str:
        """A turn's verdict as the ask words it: which outcome the classifier chose."""
        if not verdict.passed:
            return "fails the criterion"
        probabilities = verdict.probabilities
        if probabilities.get("not_applicable", 0.0) > probabilities.get("meets", 0.0):
            return "passes because the criterion does not apply to it"
        return "meets the criterion"

    @staticmethod
    def _parse_reason(response: str) -> str:
        """The reason in the answer: its JSON ``reason``, or the answer itself when it is plain text."""
        reason = _Explainer._json_object(response).get("reason")
        return str(reason).strip() if reason else response.strip()

    @staticmethod
    def _parse_run_reasons(response: str, turns: dict[str, dict[int, JudgeVerdict]]) -> _RunReasons:
        """The reasons in the run answer, for the turns asked about; one not given is empty.

        Criterion names match case-insensitively, and a turn is the number in
        its key, so ``"bot turn 2"`` reads as turn 2. An answer that is not the
        expected JSON gives no reasons, and is logged.
        """
        obj = _Explainer._json_object(response)
        if not obj:
            logger.warning(f"Explainer answer was not the expected JSON: {response!r}")
        goal = obj.get("goal")
        if isinstance(goal, dict):
            goal = goal.get("reason")
        given: dict[str, dict[int, str]] = {}
        for name, by_turn in (obj.get("turns") or {}).items():
            if isinstance(by_turn, dict):
                numbered = ((re.search(r"\d+", str(k)), v) for k, v in by_turn.items())
                given[str(name).lower()] = {int(m.group()): str(v) for m, v in numbered if m}
        reasons = {
            name: {n: given.get(name.lower(), {}).get(n, "").strip() for n in by_turn}
            for name, by_turn in turns.items()
        }
        return _RunReasons(goal=str(goal or "").strip(), turns=reasons)

    @staticmethod
    def _json_object(response: str) -> dict:
        """The JSON object in the answer, or ``{}`` when there is none; a fenced or prefaced answer still parses."""
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        start = cleaned.find("{")
        if start != -1:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}


def _cache_key(*parts) -> str:
    """Hash what a question was about — its kind, criterion and conversation — for the cache."""
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
