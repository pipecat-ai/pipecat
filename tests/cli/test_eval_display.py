#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how `pipecat eval` renders a run's outcome."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from pipecat.cli.commands.eval import (
    _eval_verdict,
    _expand_scenario_paths,
    _finalize_evals,
    _group_below_threshold,
    _print_progress,
    _turn_tally,
)
from pipecat.evals.results import (
    EvalScriptResult,
    EvalScriptTurnResult,
    EvalSimulationProgress,
    EvalSimulationResult,
)
from pipecat.evals.suite import EvalRun


def _run(statuses: list[str] | None) -> EvalRun:
    """An EvalRun whose result has turns in the given statuses (None for no result)."""
    result = None
    if statuses is not None:
        turns = [EvalScriptTurnResult(turn_index=i, status=s) for i, s in enumerate(statuses)]
        result = EvalScriptResult(
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

    def test_include_fragments_are_left_out(self):
        """The judge, user, and simulator blocks a directory's scenarios include have no name."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "alpha.yaml").write_text("name: alpha\nturns: []\n")
            (directory / "judge_text.yaml").write_text("modality: text\n")
            (directory / "simulator.yaml").write_text("service: openai\n")
            # Not valid YAML: still taken, so the run reports it instead of hiding it.
            (directory / "broken.yaml").write_text("name: [broken\n")

            paths = _expand_scenario_paths([directory])

        self.assertEqual(paths, [directory / "alpha.yaml", directory / "broken.yaml"])

    def test_file_arguments_are_preserved(self):
        scenario = Path("scenario.yaml")
        self.assertEqual(_expand_scenario_paths([scenario]), [scenario])

    def test_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Exception, "No \.yaml or \.yml scenario files found"):
                _expand_scenario_paths([Path(tmp)])


def _simulation_run(
    succeeded: bool | None,
    *,
    attempt: int = 1,
    attempts: int = 3,
    threshold: float | None = 0.67,
    quality: float | None = 1.0,
) -> EvalRun:
    """A finished simulation run; ``succeeded=None`` is one that errored out."""
    if succeeded is None:
        result = EvalSimulationResult(
            simulation_name="book", succeeded=False, error="bot never answered"
        )
    else:
        result = EvalSimulationResult(
            simulation_name="book",
            succeeded=succeeded,
            reason="judged",
            quality=quality,
            ended_by="end_call" if succeeded else "max_turns",
        )
    return EvalRun(
        bot="bot",
        scenario="book",
        scenario_path=Path("book.yaml"),
        kind="simulation",
        attempt=attempt,
        attempts=attempts,
        pass_threshold=threshold,
        status="done",
        result=result,
    )


class TestSimulationProgress(unittest.TestCase):
    def test_lines_print_as_spoken_and_the_end_says_how(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_progress(None, EvalSimulationProgress("bot", "Hi! How can I help?", 0))  # type: ignore[arg-type]
            _print_progress(None, EvalSimulationProgress("user", "A table for two.", 1))  # type: ignore[arg-type]
            _print_progress(None, EvalSimulationProgress("ended", "end_call", 1))  # type: ignore[arg-type]
        self.assertEqual(
            out.getvalue().splitlines(),
            [
                "      bot: Hi! How can I help?",
                "      user: A table for two.",
                "      ended by end_call after 1 persona turn(s)",
            ],
        )


class TestSimulationVerdicts(unittest.TestCase):
    def test_a_simulation_run_reads_its_own_outcome(self):
        self.assertEqual(_eval_verdict(_simulation_run(True)), "passed")
        self.assertEqual(_eval_verdict(_simulation_run(False)), "failed")
        self.assertEqual(_eval_verdict(_simulation_run(None)), "error")

    def test_detail_is_quality_and_how_it_ended(self):
        self.assertEqual(_turn_tally(_simulation_run(True)), "quality 1.00 · end_call")
        self.assertEqual(_turn_tally(_simulation_run(False, quality=None)), "max_turns")

    def test_rate_is_measured_against_the_threshold(self):
        two_of_three = [_simulation_run(s, attempt=i) for i, s in enumerate((True, True, False), 1)]
        self.assertFalse(_group_below_threshold(two_of_three))
        one_of_three = [
            _simulation_run(s, attempt=i) for i, s in enumerate((True, False, False), 1)
        ]
        self.assertTrue(_group_below_threshold(one_of_three))

    def test_errored_runs_stay_out_of_the_rate(self):
        # One success out of one completed run meets any threshold; the errors are
        # reported on their own rather than counted as the bot failing.
        group = [_simulation_run(s, attempt=i) for i, s in enumerate((True, None, None), 1)]
        self.assertFalse(_group_below_threshold(group))
        self.assertTrue(_group_below_threshold([_simulation_run(None)]))

    def test_scenario_groups_have_no_threshold(self):
        self.assertFalse(_group_below_threshold([_run(["failed"]), _run(["passed"])]))

    def test_exit_code_follows_the_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                passing = [
                    _simulation_run(s, attempt=i) for i, s in enumerate((True, True, False), 1)
                ]
                self.assertEqual(_finalize_evals(passing, Path(tmp), 1.0, False), 0)
                failing = [
                    _simulation_run(s, attempt=i) for i, s in enumerate((True, False, False), 1)
                ]
                self.assertEqual(_finalize_evals(failing, Path(tmp), 1.0, False), 1)
                # A single-run simulation is judged on that run alone.
                self.assertEqual(
                    _finalize_evals([_simulation_run(False, attempts=1)], Path(tmp), 1.0, False),
                    1,
                )
            self.assertIn("below 67%", out.getvalue())
            self.assertIn("goal not achieved", out.getvalue())
