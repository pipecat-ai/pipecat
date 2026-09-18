#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An eval judge that decides with TypeSafe judgments instead of an LLM.

:class:`TypeSafeEvalJudge` answers the same questions as
:class:`~pipecat.evals.judge.EvalJudge`, from the same conversation record,
but as typed yes/no questions to TypeSafe's Jev. A scripted ``eval:``
expectation is two ``Noul`` questions asked together, whether the bot has
finished answering and whether the reply satisfies the criterion, and code
combines them into yes, no or continue. A simulation run is one request
holding a ``Noul`` for the goal plus one per bot turn per metric. Each
verdict carries the probabilities behind it as its reason, in place of the
sentence an LLM judge writes.

Select it in a scenario with::

    judge:
      eval:
        service: typesafe
        model: jev-latest      # optional
        timeout: 10            # optional, seconds
        threshold: 0.5         # optional: the yes probability a Noul needs
        uncertain_band: 0.05   # optional: run verdicts this close to the
                               # threshold are "none" rather than a coin flip

Requires the ``typesafe`` extra and ``TYPESAFE_API_KEY``.
"""

import json
from collections.abc import Mapping
from typing import Any

from loguru import logger

from pipecat.evals.judge import EvalJudge, JudgeVerdict, RunVerdicts
from pipecat.services.typesafe.judge import Noul, NoulCriteria, TypeSafeJudge

FINISHED_QUESTION_ID = "finished"
"""Question id of the ``Noul`` asking whether the bot has finished its answer."""

SATISFIES_QUESTION_ID = "satisfies"
"""Question id of the ``Noul`` asking whether the reply satisfies the criterion."""

GOAL_QUESTION_ID = "goal"
"""Question id of the ``Noul`` deciding a simulation run's goal."""

REPLY_CONTEXT = (
    "A bot under test is in a conversation with a user. `conversation` is what was said "
    "before the bot's most recent reply, `bot_reply` is that reply so far (it may still be "
    "streaming in), and `criterion` is what the reply should express. When the bot spoke, "
    "`bot_reply` is a speech-to-text transcript, so judge it by the intended spoken "
    "meaning, never by spelling: 'for' can mean 'four' and 'to' can mean 'two'."
)

FINISHED_INSTRUCTIONS = (
    REPLY_CONTEXT + " Is `bot_reply` far enough along to be judged against `criterion`, "
    "or is the bot still on its way to the thing `criterion` is about?"
)

FINISHED_CRITERIA: NoulCriteria = {
    "true": {
        "what": (
            "The reply already contains what `criterion` asks about, right or wrong, or "
            "the bot has plainly given its answer. A greeting counts when `criterion` asks "
            "for a greeting or an opening"
        ),
        "examples": ["The capital of Germany is Berlin.", "Sorry, I can't help with that."],
    },
    "false": {
        "what": (
            "The bot has not reached what `criterion` asks about: it says it is checking, "
            "looking something up, or will report back; or it has so far only greeted or "
            "said a fragment while `criterion` asks for something more"
        ),
        "examples": ["Let me check on that for you.", "The capital of"],
    },
}

SATISFIES_INSTRUCTIONS = (
    REPLY_CONTEXT + " Judging only `bot_reply` and using `conversation` as context, does "
    "`bot_reply` satisfy `criterion`?"
)

SATISFIES_CRITERIA: NoulCriteria = {
    "true": "The reply does what the criterion asks, or is not in the situation the criterion is about",
    "false": "The reply does something the criterion forbids, or fails to do what it asks",
}

RUN_RULES = (
    "The transcript is a complete conversation between a user and a bot under test. The "
    "bot's replies are numbered 'Bot turn 1', 'Bot turn 2', and so on; lines marked 'User' "
    "are the user; a line marked '[tool call]' is a function the bot called at that point, "
    "and a completed call is stronger evidence of an action than the bot saying it did it. "
    "When the bot spoke, its text is a speech-to-text transcript: judge it by the intended "
    "spoken meaning, never by spelling."
)

GOAL_CRITERIA: NoulCriteria = {
    "true": (
        "The transcript shows the goal was reached, by what the bot said or by a completed "
        "tool call"
    ),
    "false": (
        "The conversation ended before the goal was reached, or the bot did something other "
        "than what the goal asks"
    ),
}

TURN_CRITERIA: NoulCriteria = {
    "true": {
        "what": (
            "The turn does what the criterion asks; or the criterion forbids something and "
            "the turn does not do it; or the criterion applies only in a situation the turn "
            "is not in"
        ),
    },
    "false": {
        "what": (
            "The turn does something the criterion forbids, or is in the situation the "
            "criterion is about and fails to do what it asks"
        ),
    },
}


def _turn_question_id(name: str, turn: int) -> str:
    return f"{name}:{turn}"


class TypeSafeEvalJudge(EvalJudge):
    """Decides scenario verdicts with TypeSafe judgments.

    Keeps the conversation exactly as :class:`EvalJudge` does and is fed the
    same way; only the deciding differs. A scripted expectation is two
    ``Noul`` questions in one request: is the answer finished, and does it
    satisfy the criterion. The verdict is ``continue`` while the first is
    below ``threshold``, otherwise yes or no from the second. A simulation
    run is one request: a ``Noul`` for the goal and one per bot turn per
    criterion. Each is yes at or above ``threshold`` and no below it, except
    that a probability within ``uncertain_band`` of the threshold is ``none``:
    the judge could not tell, which the harness counts as a failure with that
    reason rather than a verdict. Reasons carry the probabilities, for example
    ``finished 0.97, satisfies 0.08``.
    """

    def __init__(
        self, judge: TypeSafeJudge, *, threshold: float = 0.5, uncertain_band: float = 0.05
    ):
        """Initialize the judge.

        Args:
            judge: The TypeSafe client wrapper.
            threshold: The probability a ``Noul`` needs to count as yes.
            uncertain_band: How close to ``threshold`` a run verdict's
                probability may be before it is ``none`` instead of yes or no.
        """
        super().__init__(None)
        self._judge = judge
        self._threshold = threshold
        self._uncertain_band = uncertain_band

    @classmethod
    def from_config(cls, judge_config: Mapping[str, Any]) -> "TypeSafeEvalJudge":
        """Build the judge from a ``judge.eval:`` block with ``service: typesafe``.

        Args:
            judge_config: The block. Keys: ``model`` (default ``jev-latest``),
                ``timeout`` in seconds (default 10), ``threshold`` (default
                0.5), ``uncertain_band`` (default 0.05).

        Returns:
            A configured judge.
        """
        return cls(
            TypeSafeJudge(
                model=str(judge_config.get("model") or "jev-latest"),
                timeout=float(judge_config.get("timeout", 10.0)),
            ),
            threshold=float(judge_config.get("threshold", 0.5)),
            uncertain_band=float(judge_config.get("uncertain_band", 0.05)),
        )

    async def _call_judge(
        self, criterion: str, messages: list, instruction: str, ask: str
    ) -> JudgeVerdict:
        # The reply under judgment is the run of assistant segments at the end.
        reply: list[str] = []
        while messages and messages[-1]["role"] == "assistant":
            reply.insert(0, str(messages[-1]["content"]))
            messages = messages[:-1]
        state = {
            "conversation": [
                {"speaker": "bot" if m["role"] == "assistant" else "user", "text": m["content"]}
                for m in messages
            ],
            "bot_reply": " ".join(reply),
            "criterion": criterion,
        }
        logger.debug("Judge evaluating {!r} over conversation:\n{}", criterion, _show(state))
        try:
            result = await self._judge.ask(
                state,
                {
                    FINISHED_QUESTION_ID: Noul(
                        instructions=FINISHED_INSTRUCTIONS, criteria=FINISHED_CRITERIA
                    ),
                    SATISFIES_QUESTION_ID: Noul(
                        instructions=SATISFIES_INSTRUCTIONS, criteria=SATISFIES_CRITERIA
                    ),
                },
            )
        except Exception as e:
            logger.error(f"TypeSafeEvalJudge call failed: {e.__class__.__name__} ({e})")
            return JudgeVerdict(
                verdict="no", reason=f"judge call failed: {e.__class__.__name__}", raw_response=""
            )
        finished = result.nouls.get(FINISHED_QUESTION_ID)
        satisfies = result.nouls.get(SATISFIES_QUESTION_ID)
        raw = json.dumps({k: v.probability for k, v in result.nouls.items()}, sort_keys=True)
        if finished is None or satisfies is None:
            return JudgeVerdict(verdict="no", reason="judge gave no verdict", raw_response=raw)
        reason = f"finished {finished.probability:.2f}, satisfies {satisfies.probability:.2f}"
        if finished.probability < self._threshold:
            verdict = "continue"
        elif satisfies.probability >= self._threshold:
            verdict = "yes"
        else:
            verdict = "no"
        return JudgeVerdict(verdict=verdict, reason=reason, raw_response=raw)

    async def _judge_run(
        self, lines: list[str], turn_count: int, criteria: dict[str, str], success: str
    ) -> RunVerdicts:
        state = {"rules": RUN_RULES, "transcript": lines or ["(nothing was said)"]}
        questions: dict[str, Noul] = {
            GOAL_QUESTION_ID: Noul(
                instructions=(
                    "Following `rules`, and considering the whole `transcript`, is this goal "
                    f"for the conversation met: {success}"
                ),
                criteria=GOAL_CRITERIA,
            )
        }
        for name, criterion in criteria.items():
            for turn in range(1, turn_count + 1):
                questions[_turn_question_id(name, turn)] = Noul(
                    instructions=(
                        f"Following `rules`, does 'Bot turn {turn}' in `transcript`, judged on "
                        "its own in the light of the conversation before it, satisfy this "
                        f"criterion: {criterion}"
                    ),
                    criteria=TURN_CRITERIA,
                )
        logger.debug(
            "Judge evaluating {!r} and {} criteria over {} bot turn(s):\n{}",
            success,
            len(criteria),
            turn_count,
            "\n".join(f"  {line}" for line in lines),
        )
        try:
            result = await self._judge.ask(state, questions)
        except Exception as e:
            logger.error(f"TypeSafeEvalJudge call failed: {e.__class__.__name__} ({e})")
            failed = JudgeVerdict(
                verdict="none", reason=f"judge call failed: {e.__class__.__name__}", raw_response=""
            )
            return RunVerdicts(goal=failed, turns={n: [failed] * turn_count for n in criteria})

        raw = json.dumps({k: v.probability for k, v in result.nouls.items()}, sort_keys=True)

        def verdict(question_id: str) -> JudgeVerdict:
            answer = result.nouls.get(question_id)
            if answer is None:
                return JudgeVerdict(
                    verdict="none", reason="(judge gave no verdict)", raw_response=raw
                )
            p = answer.probability
            reason = f"yes {p:.2f}"
            if abs(p - self._threshold) < self._uncertain_band:
                return JudgeVerdict(
                    verdict="none", reason=f"{reason}: too close to call", raw_response=raw
                )
            return JudgeVerdict(
                verdict="yes" if p >= self._threshold else "no", reason=reason, raw_response=raw
            )

        return RunVerdicts(
            goal=verdict(GOAL_QUESTION_ID),
            turns={
                name: [verdict(_turn_question_id(name, turn)) for turn in range(1, turn_count + 1)]
                for name in criteria
            },
        )


def _show(state: Mapping[str, Any]) -> str:
    lines = [f"  [{m['speaker']}] {m['text']}" for m in state["conversation"]]
    lines.append(f"  [bot, judged] {state['bot_reply']}")
    return "\n".join(lines)
