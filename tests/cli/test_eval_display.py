#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how `pipecat eval` renders a run's outcome."""

import tempfile
import unittest
from pathlib import Path

from pipecat.cli.commands.eval import _expand_scenario_paths, _turn_tally
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


class TestScenarioPathExpansion(unittest.TestCase):
    def test_directory_arguments_expand_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "zeta.yaml").write_text("name: zeta\n")
            (directory / "alpha.yaml").write_text("name: alpha\n")
            (directory / "notes.txt").write_text("not a scenario\n")

            paths = _expand_scenario_paths([directory])

        self.assertEqual(paths, [directory / "alpha.yaml", directory / "zeta.yaml"])

    def test_both_yaml_suffixes_are_taken(self):
        """A manifest resolves either suffix, so a directory does too."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "beta.yml").write_text("name: beta\n")
            (directory / "alpha.yaml").write_text("name: alpha\n")

            paths = _expand_scenario_paths([directory])

        self.assertEqual(paths, [directory / "alpha.yaml", directory / "beta.yml"])

    def test_file_arguments_are_preserved(self):
        scenario = Path("scenario.yaml")
        self.assertEqual(_expand_scenario_paths([scenario]), [scenario])

    def test_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Exception, "No \.yaml or \.yml scenario files found"):
                _expand_scenario_paths([Path(tmp)])
