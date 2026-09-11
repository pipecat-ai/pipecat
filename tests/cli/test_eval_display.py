#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how `pipecat eval` renders a run's outcome."""

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path

from rich.console import Console

from pipecat.cli.commands.eval import (
    _eval_verdict,
    _EvalDashboard,
    _expand_scenario_paths,
    _finalize_evals,
    _fmt_duration,
    _group_outcome,
    _print_progress,
    _rate_level,
    _turn_tally,
)
from pipecat.evals.results import (
    EvalScriptResult,
    EvalScriptTurnResult,
    EvalSimulationProgress,
    EvalSimulationResult,
)
from pipecat.evals.scenario import EvalKind
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
    sweep: bool = False,
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
            ended_by="end_call" if succeeded else "max_turns",
        )
    return EvalRun(
        bot="bot",
        scenario="book",
        scenario_path=Path("book.yaml"),
        kind=EvalKind.SIMULATION,
        attempt=attempt,
        attempts=attempts,
        sweep=sweep,
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
                "",
                "      ended by end_call after 1 persona turn(s)",
            ],
        )


class TestSimulationVerdicts(unittest.TestCase):
    def test_a_simulation_run_reads_its_own_outcome(self):
        self.assertEqual(_eval_verdict(_simulation_run(True)), "passed")
        self.assertEqual(_eval_verdict(_simulation_run(False)), "failed")
        self.assertEqual(_eval_verdict(_simulation_run(None)), "error")

    def test_detail_is_how_it_ended(self):
        self.assertEqual(_turn_tally(_simulation_run(True)), "end_call")
        self.assertEqual(_turn_tally(_simulation_run(False)), "max_turns")

    def test_errored_runs_stay_out_of_the_rate(self):
        # The errors are reported on their own rather than counted as the bot failing.
        group = [_simulation_run(s, attempt=i) for i, s in enumerate((True, None, None), 1)]
        self.assertEqual(_group_outcome(group)[:3], (1, 1, 2))

    def test_every_required_run_must_pass_and_a_sweep_only_measures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                required = [
                    _simulation_run(s, attempt=i) for i, s in enumerate((True, True, True), 1)
                ]
                self.assertEqual(_finalize_evals(required, Path(tmp), 1.0, False), 0)
                one_failed = [
                    _simulation_run(s, attempt=i) for i, s in enumerate((True, True, False), 1)
                ]
                self.assertEqual(_finalize_evals(one_failed, Path(tmp), 1.0, False), 1)
                # The same runs as a --repeat sweep are data, not a break.
                sweep = [
                    _simulation_run(s, attempt=i, sweep=True)
                    for i, s in enumerate((True, True, False), 1)
                ]
                self.assertEqual(_finalize_evals(sweep, Path(tmp), 1.0, False), 0)
                # A single-run simulation is judged on that run alone.
                self.assertEqual(
                    _finalize_evals([_simulation_run(False, attempts=1)], Path(tmp), 1.0, False),
                    1,
                )
            self.assertIn("goal not met", out.getvalue())


class TestDurations(unittest.TestCase):
    def test_tenths_then_minutes_with_no_sixty_seconds_in_between(self):
        self.assertEqual(_fmt_duration(5.04), "5.0s")
        self.assertEqual(_fmt_duration(59.94), "59.9s")
        self.assertEqual(_fmt_duration(59.96), "1m 00s")
        self.assertEqual(_fmt_duration(124.4), "2m 04s")


class TestDashboard(unittest.TestCase):
    """A plain run's dashboard: one row per run, the tally last, sized to the terminal."""

    def test_rows_and_the_tally_render(self):
        runs = [_simulation_run(True, attempt=1, attempts=1) for _ in range(3)]
        for i, run in enumerate(runs):
            run.scenario = f"scenario_{i}"
            run.duration_ms = 1234
        console = Console(width=100, height=24, record=True, force_terminal=False)
        console.print(_EvalDashboard(runs, 0.0))
        lines = console.export_text().rstrip("\n").splitlines()
        self.assertEqual(len([line for line in lines if "scenario_" in line]), 3)
        self.assertIn("1234ms", lines[0])
        self.assertIn("3/3 passed", lines[-1])

    def test_a_long_run_is_windowed_to_the_terminal(self):
        runs = []
        for i in range(30):
            run = _simulation_run(True, attempt=1, attempts=1)
            run.scenario = f"scenario_{i:02d}"
            if i > 20:
                run.status = "pending"
                run.result = None
            runs.append(run)
        runs[21].status = "running"
        console = Console(width=100, height=12, record=True, force_terminal=False)
        console.print(_EvalDashboard(runs, 0.0))
        lines = console.export_text().rstrip("\n").splitlines()
        self.assertLessEqual(len(lines), 12)
        self.assertIn("↑", lines[0])
        self.assertIn("scenario_21", "".join(lines))
        self.assertIn("passed", lines[-1])


class TestRateLevel(unittest.TestCase):
    """A rate's color is a verdict on the attempts that finished."""

    def test_levels(self):
        self.assertEqual(_rate_level(0, 0), "dim")
        self.assertEqual(_rate_level(3, 3), "green")
        self.assertEqual(_rate_level(2, 3), "red")
        self.assertEqual(_rate_level(0, 3), "red")


class TestDashboardSpinner(unittest.TestCase):
    """A running row's spinner advances from frame to frame."""

    @staticmethod
    def _status_glyph(dashboard: _EvalDashboard, at: float) -> str:
        console = Console(
            width=100, height=24, record=True, force_terminal=False, get_time=lambda: at
        )
        console.print(dashboard)
        row = next(line for line in console.export_text().splitlines() if "scenario_1" in line)
        return row[0]

    def test_the_spinner_animates_across_frames(self):
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                runs = [_simulation_run(True, attempt=1, attempts=1) for _ in range(2)]
                for i, run in enumerate(runs):
                    run.scenario = f"scenario_{i}"
                runs[1].status = "running"
                runs[1].result = None
                dashboard = _EvalDashboard(runs, 0.0, grouped=grouped)
                first = self._status_glyph(dashboard, 0.0)
                later = self._status_glyph(dashboard, 0.3)
                self.assertNotEqual(first, later)

    def test_a_row_keeps_spinning_while_a_finished_attempts_bot_is_stopped(self):
        # The finished attempt holds the row's slot until its bot has stopped,
        # so the row is still busy rather than waiting for a slot.
        stopping = _simulation_run(True, attempt=1, attempts=2)
        pending = _simulation_run(True, attempt=2, attempts=2)
        pending.status = "pending"
        pending.result = None
        for run in (stopping, pending):
            run.scenario = "scenario_1"
        dashboard = _EvalDashboard([stopping, pending], 0.0, grouped=True)
        self.assertEqual(self._status_glyph(dashboard, 0.0), "·")
        stopping.stopping = True
        first = self._status_glyph(dashboard, 0.0)
        later = self._status_glyph(dashboard, 0.3)
        self.assertNotIn("·", (first, later))
        self.assertNotEqual(first, later)


class TestGroupedDashboard(unittest.TestCase):
    """A repeated row reads what is left beside passed over its total, and only the rate once every attempt is in."""

    @staticmethod
    def _render(
        runs: list[EvalRun], *, width: int = 120, height: int = 24, styles: bool = False
    ) -> str:
        console = Console(width=width, height=height, record=True, force_terminal=styles)
        console.print(_EvalDashboard(runs, 0.0, grouped=True))
        return console.export_text(styles=styles)

    def test_a_row_still_running_shows_what_is_left_and_its_rate_so_far(self):
        pending = _simulation_run(True, attempt=3, attempts=3)
        pending.status = "pending"
        pending.result = None
        text = self._render(
            [_simulation_run(True, attempt=1), _simulation_run(True, attempt=2), pending]
        )
        self.assertRegex(text, r"1 left\s+2/3 \(66%\)")
        self.assertNotIn("—", text)

    def test_a_running_rows_rate_is_a_verdict_on_the_finished_attempts(self):
        # Green while every finished attempt passed, red once one has not, and
        # dim before the first is in.
        for succeeded, code in ((True, "32"), (False, "31")):
            with self.subTest(succeeded=succeeded):
                pending = _simulation_run(True, attempt=2, attempts=2)
                pending.status = "pending"
                pending.result = None
                text = self._render(
                    [_simulation_run(succeeded, attempt=1, attempts=2), pending], styles=True
                )
                self.assertRegex(text, rf"\x1b\[{code}m\s*{int(succeeded)}/2 \({50 * succeeded}%\)")
        waiting = [_simulation_run(True, attempt=n, attempts=2) for n in (1, 2)]
        for run in waiting:
            run.status = "pending"
            run.result = None
        self.assertRegex(self._render(waiting, styles=True), r"\x1b\[2m\s*0/2 \(0%\)")

    def test_a_running_row_shows_the_attempts_clock(self):
        running = _simulation_run(True, attempt=3, attempts=3)
        running.status = "running"
        running.result = None
        running.started_at = time.monotonic() - 5
        text = self._render(
            [_simulation_run(True, attempt=1), _simulation_run(True, attempt=2), running]
        )
        # The row's clock runs from its first attempt's start; the finished
        # attempts here carry no start time, so it is the running one's.
        self.assertRegex(text, r"1 left\s+2/3 \(66%\)\s+5\.\ds")

    def test_what_is_left_and_the_clock_keep_their_columns(self):
        # A waiting row's "left" lines up with a running row's, not with its clock.
        running = _simulation_run(True, attempt=3, attempts=3)
        running.status = "running"
        running.result = None
        running.started_at = time.monotonic() - 5
        waiting = _simulation_run(True, attempt=1, attempts=3)
        waiting.status = "pending"
        waiting.result = None
        waiting.scenario = "other"
        text = self._render(
            [_simulation_run(True, attempt=1), _simulation_run(True, attempt=2), running, waiting]
        )
        lines = [line for line in text.splitlines() if "left" in line]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].index("left"), lines[1].index("left"))

    def test_the_progress_column_is_as_wide_as_a_rate_from_the_start(self):
        # Every row still running: the column already leaves room for
        # "3/3 (100%)", so the rows do not shift when the first one finishes.
        rows = []
        for i in range(2):
            run = _simulation_run(True, attempt=1, attempts=3)
            run.status = "running"
            run.result = None
            run.scenario = f"scenario_{i}"
            run.started_at = time.monotonic()
            rows.append(run)
        before = self._render(rows)
        rows[0].status = "done"
        rows[0].duration_ms = 1000
        for attempt in (2, 3):
            more = _simulation_run(True, attempt=attempt, attempts=3)
            more.scenario = "scenario_0"
            more.duration_ms = 1000
            rows.append(more)
        after = self._render(rows)
        line_before = next(l for l in before.splitlines() if "scenario_1" in l)
        line_after = next(l for l in after.splitlines() if "scenario_1" in l)
        self.assertEqual(line_before.index("left"), line_after.index("left"))

    def test_a_finished_row_drops_what_was_left_and_keeps_its_rate_in_place(self):
        finished = [_simulation_run(True, attempt=n) for n in (1, 2, 3)]
        running = _simulation_run(True, attempt=1, attempts=3)
        running.status = "running"
        running.result = None
        running.scenario = "other"
        running.started_at = time.monotonic()
        text = self._render([*finished, running])
        rate_line = next(line for line in text.splitlines() if "100%" in line)
        left_line = next(line for line in text.splitlines() if "left" in line)
        self.assertNotIn("left", rate_line)
        self.assertEqual(
            rate_line.index("100%)") + len("100%)"), left_line.index("(0%)") + len("(0%)")
        )

    def test_a_finished_row_shows_the_rate_and_how_long_it_took(self):
        # Three attempts, each 30 s, run one after another: the row took 90 s
        # from the first start to the last end.
        runs = [_simulation_run(True, attempt=n) for n in (1, 2, 3)]
        now = time.monotonic()
        for i, run in enumerate(runs):
            run.duration_ms = 30000
            run.started_at = now - 90 + 30 * i
        text = self._render(runs)
        self.assertIn("3/3 (100%)", text)
        self.assertIn("1m 30s", text)
        self.assertNotIn("left", text)

    def test_a_row_that_ran_once_shows_its_duration(self):
        run = _simulation_run(True, attempt=1, attempts=1)
        run.duration_ms = 12300
        text = self._render([run])
        self.assertIn("12.3s", text)

    def test_a_long_sweep_is_windowed_with_the_tally_kept_on_screen(self):
        # More rows than a terminal shows: the window follows the active row,
        # the rest is counted on scroll markers, and the tally stays below.
        runs = []
        for i in range(40):
            for attempt in (1, 2, 3):
                run = _simulation_run(True, attempt=attempt)
                run.scenario = f"scenario_{i:02d}"
                if i > 25:
                    run.status = "pending"
                    run.result = None
                runs.append(run)
        runs[26 * 3].status = "running"
        text = self._render(runs)
        lines = [line for line in text.splitlines() if line.strip()]
        self.assertLess(len(lines), 30)
        self.assertIn("↑", text)
        self.assertIn("↓", text)
        self.assertIn("scenario_26", text)
        self.assertNotIn("scenario_00", text)
        self.assertNotIn("scenario_39", text)
        self.assertIn("passed", lines[-1])

    def test_the_window_follows_the_terminal_height_and_rows_never_wrap(self):
        # A short, narrow pane: fewer rows fit, and a bot path too long for the
        # width is cut rather than wrapped, so the tally is the last line and
        # Rich never has to crop the dashboard with an ellipsis.
        runs = []
        for i in range(12):
            run = _simulation_run(True, attempt=1, attempts=1)
            run.bot = f"function-calling/function-calling-openai-responses-async-{i:02d}.py"
            run.scenario = f"async_tool_deferred_delivery_audio_{i:02d}"
            runs.append(run)
        text = self._render(runs, width=60, height=12)
        lines = text.rstrip("\n").splitlines()
        self.assertLessEqual(len(lines), 12)
        self.assertIn("passed", lines[-1])
        self.assertTrue(all(len(line) <= 60 for line in lines))
        self.assertIn("↑", text)
        # The path is what gets shortened; the glyph and the rate survive.
        row = next(line for line in lines if "function-calling" in line)
        self.assertTrue(row.startswith("✓"))
        self.assertIn("…", row)
        self.assertIn("100%", row)
