#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""An eval judge that decides with TypeSafe judgments instead of an LLM.

:class:`TypeSafeEvalJudge` answers the same questions as
:class:`~pipecat.evals.judge.EvalJudge`, from the same conversation record,
but as typed questions to TypeSafe's Jev: a scripted ``eval:`` expectation is
one ``Choice`` between yes, no and continue, and a simulation run is one
request holding a ``Noul`` for the goal plus a ``Noul`` per bot turn per
metric. Each verdict carries the probability behind it as its reason, in
place of the sentence an LLM judge writes.

Select it in a scenario with::

    judge:
      eval:
        service: typesafe
        model: jev-latest      # optional
        timeout: 10            # optional, seconds
        threshold: 0.5         # optional: the yes probability a Noul needs

Requires the ``typesafe`` extra and ``TYPESAFE_API_KEY``.
"""

import json
from collections.abc import Mapping
from typing import Any

from loguru import logger

from pipecat.evals.judge import EvalJudge, JudgeVerdict, RunVerdicts
from pipecat.services.typesafe.judge import Choice, Noul, TypeSafeJudge

VERDICT_QUESTION_ID = "verdict"
"""Question id of the yes/no/continue ``Choice`` for a scripted expectation."""

GOAL_QUESTION_ID = "goal"
"""Question id of the ``Noul`` deciding a simulation run's goal."""

REPLY_INSTRUCTIONS = (
    "A bot under test is in a conversation with a user. `conversation` is what was said "
    "before the bot's most recent reply, `bot_reply` is that reply so far (it may still be "
    "streaming in), and `criterion` is what the reply should express. When the bot spoke, "
    "`bot_reply` is a speech-to-text transcript, so judge it by the intended spoken "
    "meaning, never by spelling: 'for' can mean 'four' and 'to' can mean 'two'. Judge only "
    "`bot_reply`, using `conversation` as context. Does `bot_reply` satisfy `criterion`?"
)

REPLY_CRITERIA = {
    "yes": "The bot has given its answer and it satisfies the criterion",
    "no": "The bot has given its answer and it fails the criterion",
    "continue": (
        "The bot has not given its answer yet: it says it is checking, looking something "
        "up, or will report back; or the reply is a greeting or an obviously unfinished "
        "fragment. There is nothing to judge yet, however fluent the words are"
    ),
}

RUN_RULES = (
    "The transcript is a complete conversation between a user and a bot under test. The "
    "bot's replies are numbered 'Bot turn 1', 'Bot turn 2', and so on; lines marked 'User' "
    "are the user; a line marked '[tool call]' is a function the bot called at that point, "
    "and a completed call is stronger evidence of an action than the bot saying it did it. "
    "A criterion that forbids something ('never ...', 'does not ...') or applies only in a "
    "situation ('when ...', 'if ...') is satisfied by a reply that does not do the "
    "forbidden thing or is not in that situation. When the bot spoke, its text is a "
    "speech-to-text transcript: judge it by the intended spoken meaning, never by spelling."
)


def _turn_question_id(name: str, turn: int) -> str:
    return f"{name}:{turn}"


class TypeSafeEvalJudge(EvalJudge):
    """Decides scenario verdicts with TypeSafe judgments.

    Keeps the conversation exactly as :class:`EvalJudge` does and is fed the
    same way; only the deciding differs. A scripted expectation becomes one
    ``Choice`` over the reply so far, answered as whichever of yes, no or
    continue is most probable. A simulation run becomes one request: a
    ``Noul`` for the goal and one per bot turn per criterion, each answered
    yes when its probability reaches ``threshold``. Reasons carry the
    probabilities, for example ``yes 0.91, no 0.07, continue 0.02``.
    """

    def __init__(self, judge: TypeSafeJudge, *, threshold: float = 0.5):
        """Initialize the judge.

        Args:
            judge: The TypeSafe client wrapper.
            threshold: The probability a ``Noul`` needs to count as yes.
        """
        super().__init__(None)
        self._judge = judge
        self._threshold = threshold

    @classmethod
    def from_config(cls, judge_config: Mapping[str, Any]) -> "TypeSafeEvalJudge":
        """Build the judge from a ``judge.eval:`` block with ``service: typesafe``.

        Args:
            judge_config: The block. Keys: ``model`` (default ``jev-latest``),
                ``timeout`` in seconds (default 10), ``threshold`` (default 0.5).

        Returns:
            A configured judge.
        """
        return cls(
            TypeSafeJudge(
                model=str(judge_config.get("model") or "jev-latest"),
                timeout=float(judge_config.get("timeout", 10.0)),
            ),
            threshold=float(judge_config.get("threshold", 0.5)),
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
                    VERDICT_QUESTION_ID: Choice(
                        instructions=REPLY_INSTRUCTIONS, criteria=REPLY_CRITERIA
                    )
                },
            )
        except Exception as e:
            logger.error(f"TypeSafeEvalJudge call failed: {e.__class__.__name__} ({e})")
            return JudgeVerdict(
                verdict="no", reason=f"judge call failed: {e.__class__.__name__}", raw_response=""
            )
        decision = result.choices.get(VERDICT_QUESTION_ID)
        raw = json.dumps(decision.probabilities if decision else {}, sort_keys=True)
        if decision is None:
            return JudgeVerdict(verdict="no", reason="judge gave no verdict", raw_response=raw)
        return JudgeVerdict(
            verdict=decision.choice, reason=_probabilities(decision.probabilities), raw_response=raw
        )

    async def _judge_run(
        self, lines: list[str], turn_count: int, criteria: dict[str, str], success: str
    ) -> RunVerdicts:
        state = {"rules": RUN_RULES, "transcript": lines or ["(nothing was said)"]}
        questions: dict[str, Noul] = {
            GOAL_QUESTION_ID: Noul(
                instructions=(
                    "Following `rules`, and considering the whole `transcript`, is this goal "
                    f"for the conversation met: {success}"
                )
            )
        }
        for name, criterion in criteria.items():
            for turn in range(1, turn_count + 1):
                questions[_turn_question_id(name, turn)] = Noul(
                    instructions=(
                        f"Following `rules`, does 'Bot turn {turn}' in `transcript`, judged on "
                        "its own in the light of the conversation before it, satisfy this "
                        f"criterion: {criterion}"
                    )
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
            passed = answer.probability >= self._threshold
            return JudgeVerdict(
                verdict="yes" if passed else "no",
                reason=f"yes {answer.probability:.2f}",
                raw_response=raw,
            )

        return RunVerdicts(
            goal=verdict(GOAL_QUESTION_ID),
            turns={
                name: [verdict(_turn_question_id(name, turn)) for turn in range(1, turn_count + 1)]
                for name in criteria
            },
        )


def _probabilities(probabilities: Mapping[str, float]) -> str:
    return ", ".join(
        f"{k} {v:.2f}" for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1])
    )


def _show(state: Mapping[str, Any]) -> str:
    lines = [f"  [{m['speaker']}] {m['text']}" for m in state["conversation"]]
    lines.append(f"  [bot, judged] {state['bot_reply']}")
    return "\n".join(lines)
