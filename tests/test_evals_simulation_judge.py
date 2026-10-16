#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import patch

from pipecat.evals.judge import JudgeVerdict, RunVerdicts
from pipecat.evals.results import EvalTrace
from pipecat.evals.simulation import EvalSimulationMetric, EvalSimulationScenario
from pipecat.evals.simulation_judge import EvalSimulationJudge


class _FakeJudge:
    """Records the conversation it is told, and passes every turn but the ones in ``failing``."""

    def __init__(self, goal: str = "yes", failing: tuple[int, ...] = ()):
        self.goal = goal
        self.failing = failing
        self.transcript: list[dict] = []
        self.criteria: dict[str, str] = {}
        self.closed = False

    def add_user_message(self, text):
        self.transcript.append({"role": "user", "content": text})

    def add_assistant_message(self, text):
        self.transcript.append({"role": "assistant", "content": text})

    def add_tool_call(self, text):
        self.transcript.append({"role": "tool", "content": text})

    async def evaluate_run(self, criteria, success):
        self.criteria = dict(criteria)
        turns = sum(1 for e in self.transcript if e["role"] == "assistant")
        return RunVerdicts(
            goal=JudgeVerdict(verdict=self.goal, reason=f"because {success}", raw_response=""),
            turns={
                name: [
                    JudgeVerdict(
                        verdict="no" if turn in self.failing else "yes",
                        reason=f"turn {turn} {name}",
                        raw_response="",
                    )
                    for turn in range(1, turns + 1)
                ]
                for name in criteria
            },
        )

    async def close(self):
        self.closed = True


def _simulation(metrics: list[EvalSimulationMetric]) -> EvalSimulationScenario:
    return EvalSimulationScenario(
        name="book/available",
        persona="A caller.",
        goal="Book a table.",
        success="the bot booked a table",
        metrics=metrics,
    )


MESSAGES = [
    {"role": "user", "content": "A table at six."},
    {"role": "tool_call", "name": "check", "arguments": {"time": "6"}, "cancelled": False},
    {"role": "tool_call", "name": "book", "arguments": {}, "cancelled": True},
    {"role": "assistant", "content": "Six is free."},
    {"role": "user", "content": "Thanks."},
    {"role": "assistant", "content": "Bye."},
]


class TestSimulationJudge(unittest.IsolatedAsyncioTestCase):
    async def test_the_judge_reads_the_conversation_with_each_call_in_place(self):
        judge = _FakeJudge()
        await EvalSimulationJudge(_simulation([]), judge=judge).judge(MESSAGES)  # type: ignore[arg-type]
        self.assertEqual(
            judge.transcript,
            [
                {"role": "user", "content": "A table at six."},
                {"role": "tool", "content": 'check({"time": "6"})'},
                {"role": "tool", "content": "book()"},
                {"role": "tool", "content": "book was cancelled"},
                {"role": "assistant", "content": "Six is free."},
                {"role": "user", "content": "Thanks."},
                {"role": "assistant", "content": "Bye."},
            ],
        )
        # A judge passed in is its owner's to close.
        self.assertFalse(judge.closed)

    async def test_without_a_judge_each_conversation_gets_one_built_and_closed(self):
        configs: list[dict] = []
        built: list[_FakeJudge] = []

        def from_config(config):
            configs.append(config)
            built.append(_FakeJudge())
            return built[-1]

        simulation = _simulation([EvalSimulationMetric("politeness", "stays polite")])
        simulation_judge = EvalSimulationJudge(simulation, explain_below=2)
        with patch("pipecat.evals.simulation_judge.EvalJudge.from_config", from_config):
            await simulation_judge.judge(MESSAGES)
            judgement = await simulation_judge.judge(MESSAGES)
        self.assertEqual(configs, [{**simulation.judge, "explain_below": 2}] * 2)
        self.assertEqual([judge.closed for judge in built], [True, True])
        self.assertEqual(len(built[1].transcript), 7)
        self.assertEqual(judgement.metrics[0].score, 1.0)

    async def test_the_goal_and_the_criterion_metrics_are_judged_and_the_measures_left_out(self):
        metrics = [
            EvalSimulationMetric("politeness", "stays polite", min_score=1.0),
            EvalSimulationMetric("turns", measure="turns", max_value=10),
            EvalSimulationMetric("brevity", "keeps it short"),
        ]
        judge = _FakeJudge(failing=(2,))
        trace = EvalTrace()
        judgement = await EvalSimulationJudge(_simulation(metrics), judge=judge, trace=trace).judge(  # type: ignore[arg-type]
            MESSAGES
        )
        self.assertEqual(
            judge.criteria, {"politeness": "stays polite", "brevity": "keeps it short"}
        )
        self.assertEqual(
            (judgement.goal.verdict, judgement.goal.reason),
            ("yes", "because the bot booked a table"),
        )
        politeness, brevity = judgement.metrics
        self.assertEqual(
            (politeness.name, politeness.score, politeness.passed), ("politeness", 0.5, False)
        )
        self.assertEqual(politeness.failure_kind, "judge_no")
        self.assertEqual(politeness.reason, "turn 2: turn 2 politeness")
        self.assertEqual(politeness.criterion, "stays polite")
        # Without a min_score, a metric reports and fails nothing.
        self.assertEqual((brevity.score, brevity.passed), (0.5, True))
        self.assertIn(
            "judge: politeness = 0.50 (below 1.00): turn 2: turn 2 politeness",
            "\n".join(trace.lines),
        )


if __name__ == "__main__":
    unittest.main()
