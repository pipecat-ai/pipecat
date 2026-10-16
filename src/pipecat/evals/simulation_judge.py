#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Judges a conversation against a simulation's ``success`` and its criterion metrics.

The simulation driver judges a run with it when the run ends, and ``pipecat
eval judge`` judges saved conversation files with it, so a conversation gets
the same verdicts either way. Measured metrics are not its part: they come
from a live run, not from what was said.
"""

import json
from dataclasses import dataclass

from pipecat.evals.judge import EvalJudge, JudgeVerdict
from pipecat.evals.results import EvalSimulationMetricScore, EvalSimulationTurnVerdict, EvalTrace
from pipecat.evals.simulation import EvalSimulationMetric, EvalSimulationScenario


@dataclass
class EvalSimulationJudgement:
    """What the judge decided about one conversation.

    Parameters:
        goal: The verdict on the simulation's ``success``: ``yes``, ``no``, or
            ``none`` when the judge gave none.
        metrics: The criterion metrics' outcomes, in the simulation's order.
    """

    goal: JudgeVerdict
    metrics: list[EvalSimulationMetricScore]


class EvalSimulationJudge:
    """Judges conversations against a simulation.

    Each conversation is judged by a judge built from the simulation's
    ``judge`` config, closed once the conversation is judged. A judge passed
    in is used instead and left open for its owner to close; it keeps the
    conversation it is told, so it judges one conversation.

    Example::

        scenario = EvalScenarioFile.load("book_table.yaml")["book_table/available"]
        messages = json.loads(Path("run.conversation.json").read_text())
        judgement = await EvalSimulationJudge(scenario).judge(messages)
    """

    def __init__(
        self,
        simulation: EvalSimulationScenario,
        *,
        judge: EvalJudge | None = None,
        explain_below: float | None = None,
        trace: EvalTrace | None = None,
    ):
        """Initialize the simulation judge.

        Args:
            simulation: The simulation whose ``success`` and metrics to judge.
            judge: The judge to ask, which should not have seen any
                conversation yet; ``None`` builds one for each conversation.
            explain_below: For the judges it builds, a ``yes`` less sure than
                this is explained too (above 1, every verdict is); ``None``
                keeps the simulation's ``judge`` config.
            trace: The run's trace, to log each verdict in, or ``None``.
        """
        self._simulation = simulation
        self._judge = judge
        self._explain_below = explain_below
        self._trace = trace

    @staticmethod
    def call_lines(message: dict) -> list[str]:
        """A ``tool_call`` message as the judge reads it: the call, and a line more when it was cancelled."""
        name, arguments = message["name"], message["arguments"]
        lines = [f"{name}({json.dumps(arguments) if arguments else ''})"]
        if message["cancelled"]:
            lines.append(f"{name} was cancelled")
        return lines

    async def judge(self, messages: list[dict]) -> EvalSimulationJudgement:
        """Judge a conversation, its messages as a conversation file holds them.

        Args:
            messages: The conversation (see :mod:`pipecat.evals.conversation`).

        Returns:
            The verdict on the goal and the criterion metrics' outcomes.
        """
        if self._judge is not None:
            return await self._ask(self._judge, messages)
        judge = self._build_judge()
        try:
            return await self._ask(judge, messages)
        finally:
            await judge.close()

    def _build_judge(self) -> EvalJudge:
        """A judge from the simulation's ``judge`` config."""
        config = dict(self._simulation.judge)
        if self._explain_below is not None:
            config["explain_below"] = self._explain_below
        return EvalJudge.from_config(config)

    async def _ask(self, judge: EvalJudge, messages: list[dict]) -> EvalSimulationJudgement:
        """Tell the judge the conversation and ask it about the goal and the criteria."""
        self._tell_judge(judge, messages)
        criteria = {m.name: m.criterion for m in self._simulation.metrics if m.criterion}
        judged = await judge.evaluate_run(criteria, self._simulation.success)
        metrics = [
            self._score(metric, judged.turns.get(metric.name, []))
            for metric in self._simulation.metrics
            if metric.criterion
        ]
        return EvalSimulationJudgement(goal=judged.goal, metrics=metrics)

    def _log(self, line: str) -> None:
        """Log a line in the run's trace, when there is one."""
        if self._trace is not None:
            self._trace.log(line)

    def _tell_judge(self, judge: EvalJudge, messages: list[dict]) -> None:
        """Give the judge the conversation, each tool call where it was made."""
        calls: list[str] = []
        for message in messages:
            if message["role"] == "user":
                judge.add_user_message(message["content"])
            elif message["role"] == "assistant":
                judge.add_assistant_message(message["content"])
            elif message["role"] == "tool_call":
                for line in self.call_lines(message):
                    judge.add_tool_call(line)
                    calls.append(line)
        if calls:
            self._log(f"judge: the bot's tool calls: {'; '.join(calls)}")

    def _score(
        self, metric: EvalSimulationMetric, judged: list[JudgeVerdict]
    ) -> EvalSimulationMetricScore:
        """A metric's score over its turn verdicts: the share of turns that passed.

        A metric that fell short only on turns the judge left unanswered is
        judge trouble, and its failure kind says so.
        """
        verdicts = [
            EvalSimulationTurnVerdict(
                turn=index,
                passed=verdict.verdict == "yes",
                reason=verdict.reason,
                verdict=verdict.verdict,
                confidence=verdict.confidence,
                probabilities=verdict.probabilities,
            )
            for index, verdict in enumerate(judged, 1)
        ]
        for verdict in verdicts:
            self._log(
                f"judge: turn {verdict.turn} {metric.name} {verdict.verdict}: {verdict.reason}"
            )
        failure_kind = None
        if not verdicts:
            score, reason, passed = None, "no bot turn to judge", True
        else:
            score = sum(1 for v in verdicts if v.passed) / len(verdicts)
            failed = [v for v in verdicts if not v.passed]
            reason = (
                f"all {len(verdicts)} turn(s)"
                if not failed
                else "; ".join(f"turn {v.turn}: {v.reason}" for v in failed)
            )
            passed = metric.min_score is None or score >= metric.min_score
            if not passed:
                failure_kind = (
                    "judge_no" if any(v.verdict == "no" for v in failed) else "judge_no_verdict"
                )
        self._log(
            f"judge: {metric.name} = {'unscored' if score is None else f'{score:.2f}'}"
            f"{'' if passed else f' (below {metric.min_score:.2f})'}: {reason}"
        )
        return EvalSimulationMetricScore(
            name=metric.name,
            score=score,
            passed=passed,
            reason=reason,
            min_score=metric.min_score,
            verdicts=verdicts,
            failure_kind=failure_kind,
            criterion=metric.criterion,
        )
