#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The judge interface: what the harness asks a judge, and the verdicts it gets back.

A judge decides whether the bot did what a scenario asks: an ``eval:`` on a
scripted turn, or a simulation's ``success:`` and metrics.
:class:`BaseEvalJudge` keeps the conversation, fed as it happens (the user's
turns, the bot's replies, and the tool calls the bot makes), and its
subclasses answer questions about it: :class:`~pipecat.evals.judge.EvalJudge`
with an LLM, :class:`~pipecat.evals.jev_judge.JevEvalJudge` with TypeSafe's
Jev. :func:`judge_from_config` builds the one a scenario's ``judge.eval:``
block names.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgeVerdict:
    """Outcome of a single judge call.

    Parameters:
        verdict: ``"yes"`` (satisfies), ``"no"`` (substantive answer that fails),
            or ``"continue"`` (interim/filler/incomplete — re-judge once more text
            arrives).
        reason: One-sentence justification.
        raw_response: The judge's raw answer, for diagnostics.
        confidence: How sure the judge is of the verdict, from 0 to 1, when it
            reports one; an LLM judge does not.
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


class BaseEvalJudge(ABC):
    """The judge interface the harness drives: a conversation in, verdicts out.

    The base keeps the conversation; subclasses decide how a question about
    it is answered. A judge serves one scenario, so its conversation starts
    empty and the session closes it when the run ends.
    """

    def __init__(self):
        """Initialize the judge with an empty conversation."""
        # The conversation the judge evaluates against, grown by the harness over
        # the scenario: dicts with a ``role`` of ``user``, ``assistant`` (a
        # segment of a reply), or ``tool`` (a call the bot made, one line), and
        # the ``content``.
        self._transcript: list[dict] = []

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

    @abstractmethod
    async def evaluate(self, criterion: str) -> JudgeVerdict:
        """Judge whether the bot's latest reply satisfies ``criterion``, in the conversation so far.

        Args:
            criterion: Natural-language description of what the reply should express.

        Returns:
            A ``yes``, a ``no``, or a ``continue`` when the bot has not
            answered yet. A failed call is a ``no`` with the failure as its
            reason.
        """

    @abstractmethod
    async def evaluate_call(self, name: str, args: dict | None, criterion: str) -> JudgeVerdict:
        """Judge whether a function call the bot made satisfies ``criterion``, in the conversation so far.

        Args:
            name: The function's name.
            args: The call's arguments.
            criterion: Natural-language description of what the call should be.

        Returns:
            A ``yes`` or a ``no``.
        """

    @abstractmethod
    async def evaluate_run(self, criteria: dict[str, str], success: str) -> RunVerdicts:
        """Judge the whole conversation: every bot turn on every criterion, and the goal.

        Args:
            criteria: The per-turn criteria to decide, by name.
            success: The goal criterion, decided over the whole conversation.

        Returns:
            The goal's verdict and, per criterion, a verdict per bot turn in
            order. A verdict of ``none`` is one the judge could not give.
        """

    async def close(self) -> None:
        """Release what the judge holds open; the session calls it when the run ends."""


def judge_from_config(judge_config: dict | None) -> BaseEvalJudge:
    """Build the judge a scenario's ``judge.eval:`` block names.

    ``service: typesafe`` builds a
    :class:`~pipecat.evals.jev_judge.JevEvalJudge`; anything else builds an
    :class:`~pipecat.evals.judge.EvalJudge` over the LLM service the block
    names. Each is imported only when chosen.

    Args:
        judge_config: The ``judge.eval:`` mapping, or ``None`` for the default
            LLM judge.

    Returns:
        A configured judge.
    """
    if str((judge_config or {}).get("service", "")).lower() == "typesafe":
        from pipecat.evals.jev_judge import JevEvalJudge

        return JevEvalJudge.from_config(judge_config)
    from pipecat.evals.judge import EvalJudge

    return EvalJudge.from_config(judge_config)
