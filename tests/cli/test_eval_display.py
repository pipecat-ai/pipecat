#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how `pipecat eval` renders a run's outcome."""

import unittest
from pathlib import Path

from pipecat.cli.commands.eval import _turn_tally
from pipecat.evals.harness import EvalResult, EvalTurnResult
from pipecat.evals.suite import EvalRun


def _run(statuses: list[str] | None) -> EvalRun:
    """An EvalRun whose result has turns in the given statuses (None for no result)."""
    result = None
    if statuses is not None:
        turns = [EvalTurnResult(turn_index=i, status=s) for i, s in enumerate(statuses)]
        result = EvalResult(
            scenario_name="s",
            passed=all(t.status == "passed" for t in turns),
            turns=turns,
        )
    return EvalRun(bot="bot", scenario="s", scenario_path=Path("s.yaml"), result=result)


class TestTurnTally(unittest.TestCase):
    def test_fully_driven_run_is_counted(self):
        self.assertEqual(_turn_tally(_run(["passed", "failed", "passed"])), "2/3 turns")
        # Nothing passing is still a rate worth printing: every turn was scored.
        self.assertEqual(_turn_tally(_run(["failed", "failed"])), "0/2 turns")

    def test_stopped_run_has_no_rate(self):
        # The undriven turns were never attempted, so any fraction over them would
        # read as turns that failed. Where it stopped is in the failure listing.
        for statuses in (["passed", "failed", "not_run"], ["failed", "not_run"]):
            with self.subTest(statuses=statuses):
                self.assertEqual(_turn_tally(_run(statuses)), "")

    def test_passing_run_says_nothing(self):
        # The ✓ already says it; a tally beside it would only add noise.
        self.assertEqual(_turn_tally(_run(["passed", "passed"])), "")

    def test_run_without_turns(self):
        self.assertEqual(_turn_tally(_run(None)), "")
        self.assertEqual(_turn_tally(_run([])), "")
