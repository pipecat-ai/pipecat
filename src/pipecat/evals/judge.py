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
- :class:`~pipecat.classifiers.jev.classifier.JevClassifier`, over TypeSafe's
  Jev, a hosted classification model. Jev answers in a few hundred
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
explainer, for them. It judges the same conversation with the prompts in
this module, in a one-shot inference outside the pipeline, and is asked for:

- every ``no``, on a reply, a function call, a turn or a goal;
- every verdict the classifier is unsure of, meaning a confidence below
  ``explain_below`` (0.75 by default). When a verdict is close to 50/50, the
  next run can go the other way, so these are worth a second look. An LLM
  classifier's probabilities are not calibrated and rarely fall this low, so
  in practice it is the ``no`` verdicts that are explained;
- never a ``continue``, which is only a request to wait for more text.

``explain_below: 0`` limits it to the ``no`` verdicts, and a value above 1
sends it every verdict (useful for comparing a classifier against the LLM).

The explainer is asked the same question and makes its own judgement, which
costs a full LLM call. For a simulation, it judges the whole run in one call
if any verdict needs a reason, and its reasons go only to the verdicts that
need one. Either way, the explainer never changes a verdict: the
classifier's stands. If the explainer agrees, its reason is used, followed
by the probabilities. If it disagrees, the reason says so ("the explainer
judged yes: ...").

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
import warnings
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
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

_R = TypeVar("_R")

JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a conversation between a user and a "
    "bot under test. The 'user' messages are the user; the 'assistant' messages are "
    "the bot's replies. Judge only the bot's most recent reply — which may have "
    "arrived as several consecutive 'assistant' messages — against the given "
    "criterion, using the earlier turns only as context. The reply may still be "
    "streaming in. "
    "When the bot spoke its reply, the 'assistant' text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, and "
    "missing punctuation. Always judge it by the intended spoken meaning, never by its exact "
    "spelling. In particular, treat a number as the same value whether it is spelled out, "
    "written as a digit, or transcribed as a homophone: 'for' and 'fore' mean 'four' (4), and "
    "'to' and 'too' mean 'two' (2). Never answer 'no' solely because of a transcription error "
    "when the intended spoken meaning satisfies the criterion. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no" | "continue", "reason": "<one short sentence>"}. '
    'Use "yes" if the reply satisfies the criterion. '
    'Use "continue" if the bot has not given its answer yet: it says it is checking, '
    "looking something up, fetching, working on something, or that it will report back. "
    "The answer is still coming, so there is nothing to judge yet. This holds however "
    'long and however fluent the reply is: "The system is checking the current '
    'conditions for you right now." is waiting, not answering. A greeting or an '
    'obviously incomplete fragment is also "continue". '
    'Use "no" only when the bot has given its answer and that answer fails the '
    'criterion. If the bot has not answered yet, always use "continue", never "no". '
    "Do not include any other text, explanation, or markdown."
)

# Transient final user message appended for the explainer's call. The
# conversation it refers to ("the bot's most recent reply") is the LLMContext
# built from the judge's transcript; this just poses the question.
JUDGE_ASK_TEMPLATE = (
    "Does the bot's most recent reply satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes, no, or continue."
)

# The instructions for a reply when ``continue`` isn't allowed: every judged
# reply is a final answer, so the verdict is yes or no.
JUDGE_FINAL_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a conversation between a user and a "
    "bot under test. The 'user' messages are the user; the 'assistant' messages are "
    "the bot's replies. Judge only the bot's most recent reply — which may have "
    "arrived as several consecutive 'assistant' messages — against the given "
    "criterion, using the earlier turns only as context. The reply is the bot's final "
    "answer. "
    "When the bot spoke its reply, the 'assistant' text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, and "
    "missing punctuation. Always judge it by the intended spoken meaning, never by its exact "
    "spelling. In particular, treat a number as the same value whether it is spelled out, "
    "written as a digit, or transcribed as a homophone: 'for' and 'fore' mean 'four' (4), and "
    "'to' and 'too' mean 'two' (2). Never answer 'no' solely because of a transcription error "
    "when the intended spoken meaning satisfies the criterion. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no", "reason": "<one short sentence>"}. '
    'Use "yes" if the reply satisfies the criterion and "no" if it does not. '
    "Do not include any other text, explanation, or markdown."
)

JUDGE_FINAL_ASK_TEMPLATE = (
    "Does the bot's most recent reply satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes or no."
)

# The instructions for an ``eval:`` on a function call. The call is the
# subject, and the conversation is context for it, so the verdict is yes or
# no: a call is not a partial reply, and there is nothing to wait for.
JUDGE_CALL_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a function call made by a bot under "
    "test in a conversation with a user. The 'user' messages are the user; the "
    "'assistant' messages are the bot's replies so far, given only as context for the "
    "call. Judge only the call you are asked about, by its name and its arguments, "
    "against the given criterion. "
    "When the bot spoke its replies, the 'assistant' text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, and "
    "missing punctuation; judge it by the intended spoken meaning. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no", "reason": "<one short sentence>"}. '
    'Use "yes" if the call satisfies the criterion and "no" if it does not. '
    "Do not include any other text, explanation, or markdown."
)

# The ask for an ``eval:`` on a function call. It names the call and gives its
# arguments as JSON, so the verdict is about that call rather than about what
# the bot said around it.
JUDGE_CALL_ASK_TEMPLATE = (
    "The bot called the function `{name}` with arguments `{args}`. "
    "Does this call satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes or no."
)


RUN_JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a complete conversation between a user "
    "and a bot under test, given as a transcript. The bot's replies are numbered 'Bot "
    "turn 1', 'Bot turn 2', and so on; the user's lines are marked 'User'; a line "
    "marked '[tool call]' is a function the bot called at that point, and a completed "
    "call is stronger evidence of an action (a booking, a lookup) than the bot saying "
    "it did it. "
    "You are given criteria, each with a name, that every bot reply is judged against "
    "on its own, in the light of the conversation before it, and a goal that the "
    "conversation as a whole is judged against. A criterion that forbids something "
    "('never ...', 'does not ...') or that applies only in a situation ('when ...', "
    "'if ...') is satisfied by a reply that does not do the forbidden thing or is not "
    "in that situation; do not fault a reply for something the criterion does not ask "
    "of it. "
    "When the bot spoke its replies, its text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, "
    "and missing punctuation. Always judge it by the intended spoken meaning, never by "
    "its exact spelling. "
    "Respond ONLY with a JSON object on a single line of the form "
    '{"goal": {"verdict": "yes" | "no", "reason": "<one short sentence>"}, '
    '"turns": {"<criterion name>": ["yes" | "no", ...]}, '
    '"reasons": {"<criterion name>": {"<bot turn number>": "<one short sentence>"}}}. '
    'Under "turns", give every criterion an array with exactly one entry per bot '
    'turn, in order. Under "reasons", give a reason only for the turns you '
    'answered "no". Do not include any other text, explanation, or markdown.'
)

RUN_JUDGE_ASK_TEMPLATE = (
    "Transcript:\n{transcript}\n\n"
    "Criteria for every bot reply:\n{criteria}\n\n"
    "Goal for the conversation as a whole: {success}\n\n"
    "Answer with the JSON described, one array entry per bot turn: there are "
    "{turn_count} bot turns."
)


@dataclass
class JudgeVerdict:
    """Outcome of a single judge call.

    Parameters:
        verdict: ``"yes"`` (satisfies), ``"no"`` (substantive answer that fails),
            or ``"continue"`` (interim/filler/incomplete — re-judge once more text
            arrives).
        reason: One-sentence justification.
        raw_response: The judge's raw answer, for diagnostics.
        confidence: How sure the judge is of the verdict, from 0 to 1.
    """

    verdict: str
    reason: str
    raw_response: str
    confidence: float | None = None

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


_TRANSCRIPTION_NOTE = (
    "The bot's text may be an automatic speech-to-text transcription: judge its intended "
    "spoken meaning, never its spelling ('for' may mean 'four', 'to' may mean 'two')."
)

_REPLY_OUTCOMES = {
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
_TURN_OUTCOMES = {
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
        max_tokens: Cap on the explainer's response length. Default 200 —
            enough for a verdict and a short reason.
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
            max_tokens: Cap on the explainer's response length.
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
            warnings.warn(
                "Passing an LLM service to `EvalJudge` is deprecated since 1.12.0 and will be "
                "removed in 2.0.0. Pass `LLMClassifier(llm=service)`, with the service itself "
                "as the `explainer` for the reasons.",
                DeprecationWarning,
                stacklevel=2,
            )
            service = cast("LLMService[Any]", classifier)
            classifier = LLMClassifier(llm=service)
            explainer = explainer if explainer is not None else service
        self._classifier = classifier
        self._explainer = (
            _Explainer(explainer, max_tokens=max_tokens, allow_continue=allow_continue)
            if explainer is not None
            else None
        )
        self._explain_below = explain_below
        self._allow_continue = allow_continue
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
            judge_config: Mapping with keys ``service`` (default ``"ollama"``),
                ``model`` (default ``"gemma4:12b"``), optional ``endpoint``
                (service-specific default if omitted), an optional ``extra``
                mapping forwarded to the model as top-level request parameters,
                or a ``factory``; plus an optional ``explainer`` block
                (``false`` for verdicts without reasons), an optional
                ``explain_below``, and an optional ``allow_continue`` (``false``
                judges a reply yes or no only). ``None`` uses all defaults.

        Returns:
            A configured EvalJudge.

        Raises:
            ValueError: If ``service`` is unknown (matching
                :func:`pipecat.evals.services.tts_service_from_config` and
                :func:`pipecat.evals.services.stt_service_from_config`).

        Example::

            # In the scenario: judge.eval.factory: "my_pkg.make_judge_llm"
            def make_judge_llm(config):
                return TogetherLLMService(...)  # any service exposing run_inference()
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
        entries = _numbered_turns(e for e in self._transcript if e["role"] != "tool")
        latest = entries.pop()["content"] if entries and entries[-1]["role"] == "bot" else ""
        state = {"conversation": _conversation(entries), "latest_bot_reply": latest}
        key = _cache_key("reply", criterion, state)
        if key not in self._cache:
            answers = await self._choices(
                state,
                {
                    "verdict": (
                        "Does `latest_bot_reply`, the bot's most recent reply, following "
                        f"`conversation`, satisfy this criterion? Criterion: "
                        f"{_sentence(criterion)} {_TRANSCRIPTION_NOTE}",
                        self._reply_outcomes,
                    )
                },
            )
            if answers is None:
                verdict = _failed("no")
            else:
                verdict = _reply_verdict(answers["verdict"])
                if verdict.verdict != "continue":
                    verdict = await self._explain(
                        verdict, lambda e: e.explain(self._transcript, criterion)
                    )
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
        entries = _numbered_turns(e for e in self._transcript if e["role"] != "tool")
        state = {
            "conversation": _conversation(entries),
            "call": {"name": name, "arguments": args or {}},
        }
        key = _cache_key("call", criterion, state)
        if key not in self._cache:
            answer = await self._yes_no(
                state,
                "Does the bot's function `call`, judged by its name and arguments, satisfy "
                f"this criterion? Criterion: {_sentence(criterion)} `conversation` is context "
                f"only. {_TRANSCRIPTION_NOTE}",
            )
            if answer is None:
                verdict = _failed("no")
            else:
                verdict = await self._explain(
                    _yes_no_verdict(answer),
                    lambda e: e.explain_call(self._transcript, name, args, criterion),
                )
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
            warnings.warn(
                "`transcript` parameter of `EvalJudge.evaluate_run` is deprecated since 1.11.0 "
                "and will be removed in 2.0.0. Feed the judge with `add_user_message`, "
                "`add_assistant_message` and `add_tool_call` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        conversation = list(self._transcript if transcript is None else transcript)
        entries = _numbered_turns(conversation)
        names = list(criteria)
        key = _cache_key("run", criteria, success, entries)
        if key in self._run_cache:
            return self._run_cache[key]

        goal_instructions = (
            f"Does the conversation as a whole achieve this goal? Goal: {_sentence(success)} "
            "A `tool` entry is a function the bot called at that point; a completed "
            "call is stronger evidence of an action than the bot saying it did it. "
            f"{_TRANSCRIPTION_NOTE}"
        )
        turn_questions = {
            name: (
                "`latest_bot_reply` is the bot's reply following `conversation`. How does it "
                f"stand against this criterion? Criterion: {_sentence(criteria[name])} "
                f"{_TRANSCRIPTION_NOTE}",
                _TURN_OUTCOMES,
            )
            for name in names
        }
        turn_states = [
            {"conversation": _conversation(entries[:i]), "latest_bot_reply": entry["content"]}
            for i, entry in enumerate(entries)
            if entry["role"] == "bot"
        ]

        # The goal, and every criterion on each turn: one call each, all at once.
        goal_answer, turn_answers = await asyncio.gather(
            self._yes_no({"conversation": _conversation(entries)}, goal_instructions),
            asyncio.gather(
                *(
                    self._choices(state, turn_questions)
                    for state in (turn_states if turn_questions else [])
                )
            ),
        )

        failed = _failed("none")
        verdicts = RunVerdicts(
            goal=_yes_no_verdict(goal_answer) if goal_answer else failed,
            turns={
                name: [
                    _turn_verdict(answers[name]) if answers else failed for answers in turn_answers
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

    async def _explain(
        self,
        verdict: JudgeVerdict,
        ask: Callable[["_Explainer"], Awaitable[JudgeVerdict]],
    ) -> JudgeVerdict:
        """The verdict with the explainer's reason for it, when it needs one and there is an explainer."""
        if self._explainer is None or not self._needs_reason(verdict):
            return verdict
        return _explained(verdict, await ask(self._explainer))

    async def _explain_run(
        self,
        verdicts: RunVerdicts,
        transcript: Sequence[dict],
        criteria: dict[str, str],
        success: str,
    ) -> RunVerdicts:
        """The run's verdicts, those that need a reason given the explainer's; it judges the run once."""
        everything = [verdicts.goal, *(v for vs in verdicts.turns.values() for v in vs)]
        if self._explainer is None or not any(self._needs_reason(v) for v in everything):
            return verdicts
        explanation = await self._explainer.explain_run(transcript, criteria, success)

        def explain(verdict: JudgeVerdict, reasoned: JudgeVerdict | None) -> JudgeVerdict:
            if reasoned is None or reasoned.verdict == "none" or not self._needs_reason(verdict):
                return verdict
            return _explained(verdict, reasoned)

        turns: dict[str, list[JudgeVerdict]] = {}
        for name, turn_verdicts in verdicts.turns.items():
            reasoned = explanation.turns.get(name, [])
            turns[name] = [
                explain(verdict, reasoned[i] if i < len(reasoned) else None)
                for i, verdict in enumerate(turn_verdicts)
            ]
        return RunVerdicts(goal=explain(verdicts.goal, explanation.goal), turns=turns)

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
        self, state, questions: dict[str, tuple[str, dict[str, str]]]
    ) -> dict[str, ChoiceResult] | None:
        """The option chosen for each question, or ``None`` when they failed.

        The questions share the state, so they go in one call.
        """
        return await self._ask(
            state,
            lambda: self._classifier.choice(
                state,
                {
                    name: ChoiceQuestion(instructions=instructions, options=dict(options))
                    for name, (instructions, options) in questions.items()
                },
            ),
        )

    async def _ask(self, state, ask: Callable[[], Awaitable[_R]]) -> "_R | None":
        """The classifier's answer, asked once more if it failed, or ``None``."""
        # The first question sets the classifier up; the others wait for it.
        if self._setup_task is None:
            self._setup_task = asyncio.get_running_loop().create_task(self._setup())
        await asyncio.shield(self._setup_task)
        logger.debug(f"Judge asking over state:\n{json.dumps(state)}")
        for attempt in (1, 2):
            try:
                return await ask()
            except ClassifierError as e:
                if attempt == 1:
                    logger.warning(f"Judge question failed, asking again: {e}")
                else:
                    logger.error(f"Judge question failed: {e}")
        return None


class _Explainer:
    """Asks an LLM the question a classifier answered, for the reason behind it.

    Its answers are cached by question and conversation, so explaining the
    same verdict twice costs one call.

    Args:
        service: A pipecat LLM service with a ``run_inference()`` method
            (i.e. ``BaseOpenAILLMService`` or any subclass: OpenAI, Ollama, etc.).
        max_tokens: Cap on the explainer's response length. Default 200 —
            enough for a JSON verdict + short reason.
        allow_continue: Whether a reply may be judged ``continue``; when
            ``False``, a reply is ``yes`` or ``no``. It follows the judge's
            own setting, so both are asked the same question.
    """

    def __init__(
        self, service: LLMService[Any], *, max_tokens: int = 200, allow_continue: bool = True
    ):
        """Initialize the explainer with a configured pipecat LLM service.

        Args:
            service: A pipecat LLM service exposing ``run_inference()``.
            max_tokens: Cap on the explainer's response length.
            allow_continue: Whether a reply may be judged ``continue``.
        """
        self._service = service
        self._max_tokens = max_tokens
        self._allow_continue = allow_continue
        self._cache: dict[str, JudgeVerdict] = {}
        self._run_cache: dict[str, RunVerdicts] = {}

    @property
    def service(self) -> LLMService[Any]:
        """The LLM service that answers."""
        return self._service

    async def explain(self, transcript: Sequence[dict], criterion: str) -> JudgeVerdict:
        """Judge whether the bot's latest reply satisfies ``criterion``, and say why.

        Args:
            transcript: The conversation so far, as the judge kept it.
            criterion: Natural-language description of what the reply should express.

        Returns:
            The explainer's own verdict, with a one-sentence reason.
        """
        if self._allow_continue:
            ask = JUDGE_ASK_TEMPLATE.format(criterion=criterion)
            return await self._evaluate(transcript, criterion, JUDGE_SYSTEM_INSTRUCTION, ask)
        ask = JUDGE_FINAL_ASK_TEMPLATE.format(criterion=criterion)
        verdict = await self._evaluate(transcript, criterion, JUDGE_FINAL_SYSTEM_INSTRUCTION, ask)
        if verdict.verdict == "continue":
            # An answer that ignored the yes/no instructions counts as a no.
            return JudgeVerdict(
                verdict="no", reason=verdict.reason, raw_response=verdict.raw_response
            )
        return verdict

    async def explain_call(
        self, transcript: Sequence[dict], name: str, args: dict | None, criterion: str
    ) -> JudgeVerdict:
        """Judge whether a function call the bot made satisfies ``criterion``, and say why.

        The ask names the call and its arguments, the conversation so far is
        context, and the verdict is yes or no: a call is not a partial reply,
        so there is nothing to wait for.

        Args:
            transcript: The conversation so far, as the judge kept it.
            name: The function's name.
            args: The call's arguments, shown to the explainer as JSON.
            criterion: Natural-language description of what the call should be.

        Returns:
            The explainer's own verdict, with a one-sentence reason.
        """
        ask = JUDGE_CALL_ASK_TEMPLATE.format(
            name=name, args=json.dumps(args or {}, ensure_ascii=False), criterion=criterion
        )
        return await self._evaluate(transcript, criterion, JUDGE_CALL_SYSTEM_INSTRUCTION, ask)

    async def explain_run(
        self, transcript: Sequence[dict], criteria: dict[str, str], success: str
    ) -> RunVerdicts:
        """Judge the whole conversation in one call: every bot turn on every criterion, and the goal.

        The conversation goes in the question, bot turns numbered and tool
        calls inline. A bot turn is a run of reply segments with nothing else
        between them.

        Args:
            transcript: The conversation, as the judge kept it.
            criteria: The per-turn criteria to decide, by name.
            success: The goal criterion, decided over the whole conversation.

        Returns:
            The goal's verdict and, per criterion, a verdict per bot turn in
            order, each with a reason for a ``no``. A verdict of ``none`` is
            one the explainer did not give: a turn it left out, a goal it did
            not answer, or a call that failed.
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
        listed = "\n".join(f"- {name}: {criterion}" for name, criterion in criteria.items())
        ask = RUN_JUDGE_ASK_TEMPLATE.format(
            transcript="\n".join(lines) or "(nothing was said)",
            criteria=listed or "(none)",
            success=success,
            turn_count=turn,
        )
        key = _cache_key(ask, [])
        if key not in self._run_cache:
            # Room for a verdict per turn per criterion, a reason per "no", and
            # the goal's verdict; a budget sized for one verdict cuts it short.
            budget = max(300, 4 * turn * len(criteria) + 60 * len(criteria) + 80)
            response = await self._ask(
                success, [], RUN_JUDGE_SYSTEM_INSTRUCTION, ask, max_tokens=budget
            )
            self._run_cache[key] = _parse_run_verdicts(response, list(criteria), turn)
        return self._run_cache[key]

    async def _evaluate(
        self, transcript: Sequence[dict], criterion: str, instruction: str, ask: str
    ) -> JudgeVerdict:
        # The spoken conversation only: a reply is judged on what was said.
        messages = [e for e in transcript if e["role"] != "tool"]
        key = _cache_key(ask, messages)
        if key not in self._cache:
            response = await self._ask(criterion, messages, instruction, ask)
            if response.startswith("\0"):
                self._cache[key] = JudgeVerdict(verdict="no", reason=response[1:], raw_response="")
            else:
                self._cache[key] = _parse_verdict(response)
        return self._cache[key]

    async def _ask(
        self,
        criterion: str,
        messages: list,
        instruction: str,
        ask: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """The explainer's raw answer to ``ask``.

        A failed or empty call comes back as a NUL-prefixed reason, which no
        answer starts with, so callers can report it as a ``no``.
        """
        # Copy the conversation and append the transient ask, so neither the ask
        # nor the answer ever lands in the persistent context.
        context = LLMContext(messages=list(messages))
        context.add_message({"role": "user", "content": ask})

        # Log the conversation the explainer is about to read, before its answer,
        # so the debug log shows exactly what it saw (handy when a terse or
        # mis-transcribed reply gets an unexpected verdict). A run-level ask
        # carries the transcript itself, so that is what to show.
        transcript = "\n".join(f"  [{m.get('role')}] {m.get('content')}" for m in messages)
        logger.debug(
            "Explainer evaluating {!r} over conversation:\n{}",
            criterion,
            transcript or "\n".join(f"  {line}" for line in ask.splitlines()),
        )

        try:
            response = await self._service.run_inference(
                context=context,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
                system_instruction=instruction,
            )
        except Exception as e:
            logger.error(f"Explainer call failed: {e.__class__.__name__} ({e})")
            return f"\0explainer call failed: {e.__class__.__name__}"

        if not response:
            return "\0explainer returned empty response"

        return response


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


def _conversation(entries: list[dict]) -> list[dict]:
    """The conversation as the state carries it: a speaker, the text, and a bot turn's number."""
    return [
        {"speaker": e["role"], **({"turn": e["turn"]} if "turn" in e else {}), "text": e["content"]}
        for e in entries
    ]


def _reply_verdict(answer: ChoiceResult) -> JudgeVerdict:
    """A reply's verdict: the choice of ``yes``, ``no`` or ``continue``."""
    return JudgeVerdict(
        verdict=answer.choice,
        reason=_probabilities(answer.probabilities),
        raw_response=answer.model_dump_json(),
        confidence=answer.confidence,
    )


def _yes_no_verdict(answer: YesNoResult) -> JudgeVerdict:
    """A yes/no verdict from the probability of yes."""
    probability = answer.probability
    return JudgeVerdict(
        verdict="yes" if answer.is_yes else "no",
        reason=f"P(yes)={probability:.2f}",
        raw_response=answer.model_dump_json(),
        confidence=max(probability, 1 - probability),
    )


def _turn_verdict(answer: ChoiceResult) -> JudgeVerdict:
    """A turn's verdict: a ``no`` only when the criterion applies to the turn and fails.

    Its confidence is the probability of the verdict given: for a pass, the
    probability of either outcome that passes.
    """
    fails = answer.probabilities.get("fails", 0.0)
    passed = answer.choice != "fails"
    return JudgeVerdict(
        verdict="yes" if passed else "no",
        reason=_probabilities(answer.probabilities),
        raw_response=answer.model_dump_json(),
        confidence=1 - fails if passed else fails,
    )


def _sentence(text: str) -> str:
    """``text`` ending in punctuation, so the instruction after it reads as a new sentence."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _probabilities(probabilities: dict[str, float]) -> str:
    """The probability of each outcome, as a verdict's reason."""
    return ", ".join(f"P({k})={v:.2f}" for k, v in probabilities.items())


def _failed(verdict: str) -> JudgeVerdict:
    """The verdict a failed call gives."""
    return JudgeVerdict(verdict=verdict, reason="judge call failed", raw_response="")


def _explained(verdict: JudgeVerdict, explanation: JudgeVerdict) -> JudgeVerdict:
    """The verdict with the explainer's reason, noting when the explainer disagreed."""
    if explanation.verdict == verdict.verdict:
        given = explanation.reason and explanation.reason != NO_REASON
        reason = f"{explanation.reason} ({verdict.reason})" if given else verdict.reason
    else:
        reason = (
            f"{verdict.reason}; the explainer judged {explanation.verdict}: {explanation.reason}"
        )
    return JudgeVerdict(
        verdict=verdict.verdict,
        reason=reason,
        raw_response=verdict.raw_response,
        confidence=verdict.confidence,
    )


# The reason a verdict carries when the judge gave none.
_NO_VERDICT = "(judge gave no verdict)"
# The reason a verdict carries when the judge gave the verdict without one.
NO_REASON = "(no reason given)"


def _parse_run_verdicts(response: str, names: list[str], turn_count: int) -> RunVerdicts:
    """Parse the run answer into the goal's verdict and one per turn per criterion.

    Anything missing or malformed is a ``none`` with a reason, and the raw
    answer is logged, so a bad answer never passes a turn silently.
    """
    if response.startswith("\0"):
        failed = JudgeVerdict(verdict="none", reason=response[1:], raw_response="")
        return RunVerdicts(goal=failed, turns={n: [failed] * turn_count for n in names})
    obj = _judge_json(response)
    goal = obj.get("goal")
    if not isinstance(goal, dict):
        goal = {}
    answer = str(goal.get("verdict", "")).strip().lower()
    goal_verdict = answer if answer in ("yes", "no") else "none"
    goal_reason = str(goal.get("reason", "")).strip()
    if goal_verdict == "none":
        goal_reason = _NO_VERDICT
    elif goal_verdict == "no" and not goal_reason:
        goal_reason = NO_REASON
    return RunVerdicts(
        goal=JudgeVerdict(verdict=goal_verdict, reason=goal_reason, raw_response=response),
        turns={name: _turn_verdicts(obj, name, turn_count, response) for name in names},
    )


def _judge_json(response: str) -> dict:
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
        except (json.JSONDecodeError, AttributeError):
            pass
    logger.warning(f"Explainer answer was not the expected JSON: {response!r}")
    return {}


def _turn_verdicts(obj: dict, name: str, turn_count: int, response: str) -> list[JudgeVerdict]:
    """One verdict per bot turn for criterion ``name``; a turn left out is a ``none``.

    Criterion names match case-insensitively; a ``reasons`` entry, keyed by
    the turn number, gives a ``no`` its reason.
    """
    turns_by_name = {
        str(k).lower(): v for k, v in (obj.get("turns") or {}).items() if isinstance(v, list)
    }
    reasons_by_name = {
        str(k).lower(): v for k, v in (obj.get("reasons") or {}).items() if isinstance(v, dict)
    }
    answers = turns_by_name.get(name.lower(), [])
    reasons = reasons_by_name.get(name.lower(), {})
    if len(answers) != turn_count:
        logger.warning(
            f"Explainer gave {len(answers)} verdict(s) for {name!r} over {turn_count} bot "
            f"turn(s); its answer was: {response!r}"
        )
    verdicts = []
    for index in range(turn_count):
        answer = answers[index] if index < len(answers) else None
        if isinstance(answer, dict):
            answer = answer.get("verdict")
        if answer is None:
            verdicts.append(JudgeVerdict(verdict="none", reason=_NO_VERDICT, raw_response=response))
            continue
        verdict = "yes" if str(answer).strip().lower() == "yes" else "no"
        reason = str(reasons.get(str(index + 1), "")).strip()
        if verdict == "no" and not reason:
            reason = NO_REASON
        verdicts.append(JudgeVerdict(verdict=verdict, reason=reason, raw_response=response))
    return verdicts


def _cache_key(*parts) -> str:
    """Hash what a question was about — its kind, criterion and conversation — for the cache."""
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _parse_verdict(response: str) -> JudgeVerdict:
    """Parse the explainer's response. Tolerant of extra whitespace and code fences."""
    cleaned = response.strip()
    # Strip markdown code fences if the model ignored instructions
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    # Parse the first JSON object and ignore anything around it. Some models
    # ignore "respond ONLY with JSON" and wrap the verdict in prose (e.g. a trailing
    # "Let me know if you'd like to evaluate further turns!"); raw_decode from the
    # first '{' parses the object and stops, leaving the trailing text out.
    start = cleaned.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict not in ("yes", "no", "continue"):
                verdict = "no"
            reason = str(obj.get("reason", "")).strip()
            return JudgeVerdict(
                verdict=verdict,
                reason=reason or NO_REASON,
                raw_response=response,
            )
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: scan for a verdict keyword in the raw text.
    lowered = cleaned.lower()
    if "continue" in lowered:
        return JudgeVerdict(
            verdict="continue", reason="(unstructured continue)", raw_response=response
        )
    if "yes" in lowered and "no" not in lowered:
        return JudgeVerdict(verdict="yes", reason="(unstructured yes)", raw_response=response)
    if "no" in lowered and "yes" not in lowered:
        return JudgeVerdict(verdict="no", reason="(unstructured no)", raw_response=response)
    return JudgeVerdict(
        verdict="no",
        reason=f"could not parse explainer response: {response!r}",
        raw_response=response,
    )
