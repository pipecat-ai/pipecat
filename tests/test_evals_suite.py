#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval suite's manifest parsing, per-run log capture, and run updates."""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

from loguru import logger

from pipecat.evals.suite import (
    DEFAULT_CONCURRENCY,
    DEFAULT_SPAWN,
    EvalManifest,
    EvalRun,
    EvalSuite,
    capture_pipeline_logs,
)

MANIFEST = """
bots_dir: bots
scenarios_dir: my-scenarios
concurrency: 2
runs_dir: out
record: true
suite:
  - bot: voice/voice-a.py
    scenarios: [simple_math, multi_turn]
  - bot: vision/vision-b.py
    runner_body: bodies/cat.json
    scenarios: [other/special.yaml]
"""


class TestEvalManifestLoad(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.manifest_path = self.base / "manifest.yaml"
        self.manifest_path.write_text(MANIFEST)

    def tearDown(self):
        self._tmp.cleanup()

    def test_paths_resolve_relative_to_manifest(self):
        m = EvalManifest.load(self.manifest_path)
        self.assertEqual(m.concurrency, 2)
        self.assertTrue(m.record)
        self.assertEqual(m.runs_dir, self.base / "out")
        self.assertEqual(len(m.runs), 3)  # 2 + 1 scenarios

        first = m.runs[0]
        self.assertEqual(first.bot, "voice/voice-a.py")
        self.assertEqual(first.bot_path, self.base / "bots" / "voice" / "voice-a.py")
        # Bare scenario names resolve under scenarios_dir, with .yaml appended.
        self.assertEqual(first.scenario, "simple_math")
        self.assertEqual(first.scenario_path, self.base / "my-scenarios" / "simple_math.yaml")

    def test_scenario_paths_resolve_relative_to_manifest(self):
        m = EvalManifest.load(self.manifest_path)
        special = m.runs[2]
        # A path-like scenario bypasses scenarios_dir and resolves to the manifest.
        self.assertEqual(special.scenario, "special")
        self.assertEqual(special.scenario_path, self.base / "other" / "special.yaml")
        self.assertEqual(special.runner_body_path, self.base / "bodies" / "cat.json")

    def test_defaults(self):
        (self.base / "minimal.yaml").write_text("suite: []\n")
        m = EvalManifest.load(self.base / "minimal.yaml")
        self.assertEqual(m.concurrency, DEFAULT_CONCURRENCY)
        self.assertEqual(m.spawn, DEFAULT_SPAWN)
        self.assertFalse(m.record)
        self.assertIsNone(m.runs_dir)
        self.assertEqual(m.runs, [])

    def test_overrides_win(self):
        m = EvalManifest.load(self.manifest_path, concurrency=8, record=False, spawn="x {bot}")
        self.assertEqual(m.concurrency, 8)
        self.assertFalse(m.record)
        self.assertEqual(m.spawn, "x {bot}")


class TestCapturePipelineLogs(unittest.TestCase):
    def test_writes_sections_per_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            with capture_pipeline_logs(logs_dir, "run1", name="simple_math", enabled=True):
                with logger.contextualize(eval_pipeline="judge"):
                    logger.debug("judge line")
                logger.debug("harness line")

            content = (logs_dir / "run1.debug.log").read_text()
            self.assertIn("===== judge logs: simple_math =====", content)
            self.assertIn("judge line", content)
            self.assertIn("===== harness logs: simple_math =====", content)
            self.assertIn("harness line", content)

    def test_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            with capture_pipeline_logs(logs_dir, "run1", name="x", enabled=False):
                logger.debug("dropped")
            self.assertEqual(list(logs_dir.iterdir()), [])

    def test_concurrent_runs_do_not_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            # Logs emitted under a different eval_run id must not land in run1's file.
            with capture_pipeline_logs(logs_dir, "run1", name="a", enabled=True):
                logger.debug("mine")
                with logger.contextualize(eval_run="run2"):
                    logger.debug("theirs")
            content = (logs_dir / "run1.debug.log").read_text()
            self.assertIn("mine", content)
            self.assertNotIn("theirs", content)


class TestSuiteUpdateEvent(unittest.IsolatedAsyncioTestCase):
    """``on_update`` handlers see each run enter ``running`` and reach ``done``."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self._tmp.name)
        # A bot path that doesn't exist: the run errors out before spawning
        # anything, which is enough to drive both status changes.
        run = EvalRun(
            bot="missing.py",
            scenario="none",
            scenario_path=self.logs_dir / "none.yaml",
            bot_path=self.logs_dir / "missing.py",
        )
        self.suite = EvalSuite(
            EvalManifest(
                runs=[run],
                spawn=DEFAULT_SPAWN,
                python=sys.executable,
                concurrency=1,
                repeat=1,
                base_port=7900,
                runs_dir=self.logs_dir,
                record=False,
                cache_dir=None,
            )
        )

    def tearDown(self):
        self._tmp.cleanup()
        # EvalSuite.run() drops every log sink to keep stdout clean for its caller;
        # put loguru's default back so the rest of the session still logs.
        logger.remove()
        logger.add(sys.stderr)

    async def test_event_handler_receives_runs(self):
        seen = []

        @self.suite.event_handler("on_update")
        async def on_update(source, run):
            seen.append((source, run.status))

        await self.suite.run(self.logs_dir)

        self.assertTrue(all(source is self.suite for source, _ in seen))
        self.assertEqual([status for _, status in seen], ["running", "done"])

    async def test_callback_is_deprecated_and_still_called(self):
        seen = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await self.suite.run(self.logs_dir, on_update=lambda run: seen.append(run.status))
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)
        # The callback takes only the run, not the suite an event handler gets.
        self.assertEqual(seen, ["running", "done"])

    async def test_deprecated_knobs_still_work(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await self.suite.run(self.logs_dir, use_cache=False, default_timeout_ms=1234)
        self.assertEqual([w.category for w in caught], [DeprecationWarning])
        self.assertIn("`EvalSuite.run`", str(caught[0].message))

    async def test_callback_stays_scoped_to_the_call_it_was_passed_to(self):
        """The callback is a per-call parameter, so a reused suite doesn't accumulate it."""
        seen = []
        callback = lambda run: seen.append(run.status)  # noqa: E731

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            await self.suite.run(self.logs_dir, on_update=callback)
            self.assertEqual(seen, ["running", "done"])

            # Passing it again reports each change once more, not twice.
            await self.suite.run(self.logs_dir, on_update=callback)
            self.assertEqual(seen, ["running", "done"] * 2)

        # Omitting it stops the reporting.
        await self.suite.run(self.logs_dir)
        self.assertEqual(seen, ["running", "done"] * 2)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Simulations in a manifest's scenarios: list, and their results.jsonl records.
# ---------------------------------------------------------------------------

import json  # noqa: E402

from pipecat.evals.results import (  # noqa: E402
    EvalSimulationMetricScore,
    EvalSimulationResult,
    EvalSimulationTurnVerdict,
)
from pipecat.evals.scenario import EvalKind  # noqa: E402
from pipecat.evals.suite import _append_result, _simulation_result_from_dict  # noqa: E402

SIMULATION = """
name: {name}
persona: "A caller."
goal: "Get it done."
simulator: {{service: openai}}
success: "it got done"
runs: {runs}
"""


class TestManifestSimulations(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        (self.base / "scenarios").mkdir()
        (self.base / "scenarios" / "book.yaml").write_text(SIMULATION.format(name="book", runs=3))
        (self.base / "scenarios" / "once.yaml").write_text(SIMULATION.format(name="once", runs=1))
        (self.base / "scenarios" / "greet.yaml").write_text("name: greet\nturns: []\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _manifest(self, text: str, **overrides) -> EvalManifest:
        path = self.base / "manifest.yaml"
        path.write_text(text)
        return EvalManifest.load(path, **overrides)

    def test_the_file_says_which_kind_a_scenario_is_and_how_often_it_runs(self):
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [greet, book, once]\n")
        by_name = {}
        for run in manifest.runs:
            by_name.setdefault(run.scenario, []).append(run)
        self.assertEqual([r.attempt for r in by_name["book"]], [1, 2, 3])
        self.assertEqual([r.attempt for r in by_name["once"]], [1])
        self.assertEqual([r.attempt for r in by_name["greet"]], [1])
        book = by_name["book"][0]
        self.assertEqual(book.kind, "simulation")
        self.assertEqual(book.attempts, 3)
        self.assertFalse(book.sweep)  # its runs are a requirement
        self.assertEqual(book.scenario_path, self.base / "scenarios" / "book.yaml")
        greet = by_name["greet"][0]
        self.assertEqual(greet.kind, "script")
        self.assertEqual(greet.attempts, 1)
        self.assertFalse(greet.sweep)
        # Attempt-major: every scenario's first attempt precedes any second one.
        self.assertEqual([r.attempt for r in manifest.runs], [1, 1, 1, 2, 3])

    def test_repeat_overrides_a_simulations_runs(self):
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [book]\n", repeat=2)
        self.assertEqual([r.attempt for r in manifest.runs], [1, 2])
        self.assertEqual(manifest.runs[0].attempts, 2)
        # A repeat makes the suite a measurement.
        self.assertTrue(all(r.sweep for r in manifest.runs))

    def test_a_repeat_of_one_is_an_override_too(self):
        """Set on the command line or in the manifest, 1 means one run, not the file's three."""
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [book]\n", repeat=1)
        self.assertEqual([r.attempt for r in manifest.runs], [1])
        manifest = self._manifest("repeat: 1\nsuite:\n  - bot: bot.py\n    scenarios: [book]\n")
        self.assertEqual([r.attempt for r in manifest.runs], [1])

    def test_a_folder_in_a_name_stays_under_the_scenarios_dir(self):
        (self.base / "scenarios" / "scripted").mkdir()
        (self.base / "scenarios" / "scripted" / "greet.yaml").write_text("name: greet\nturns: []\n")
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [scripted/greet]\n")
        run = manifest.runs[0]
        self.assertEqual(run.scenario, "greet")
        self.assertEqual(run.scenario_path, self.base / "scenarios" / "scripted" / "greet.yaml")

    def test_the_suite_filters_by_kind(self):
        from pipecat.evals.suite import EvalSuite

        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [greet, book]\n")
        runs = EvalSuite(manifest).filter(kind=EvalKind.SIMULATION)
        self.assertEqual({r.scenario for r in runs}, {"book"})
        self.assertEqual(len(runs), 3)

    def test_a_missing_scenario_still_gets_a_run(self):
        """Its kind can't be read, so it runs once as a scenario and reports the error."""
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [nope]\n")
        self.assertEqual(len(manifest.runs), 1)
        self.assertEqual(manifest.runs[0].kind, "script")
        self.assertEqual(manifest.runs[0].attempts, 1)


class TestSimulationRecords(unittest.TestCase):
    def test_result_roundtrips_through_the_worker_json(self):
        result = EvalSimulationResult(
            simulation_name="book",
            succeeded=True,
            reason="booked",
            metrics=[
                EvalSimulationMetricScore(
                    name="politeness",
                    score=0.5,
                    passed=False,
                    reason="turn 2: curt",
                    min_score=1.0,
                    verdicts=[
                        EvalSimulationTurnVerdict(1, True, "warm"),
                        EvalSimulationTurnVerdict(2, False, "curt"),
                    ],
                )
            ],
            messages=[{"role": "user", "content": "hi"}],
            turns=2,
            ended_by="end_call",
            end_call={"success": True, "reason": "done"},
            duration_ms=1234,
        )
        import dataclasses

        rebuilt = _simulation_result_from_dict(json.loads(json.dumps(dataclasses.asdict(result))))
        self.assertEqual(rebuilt, result)

    def test_a_scripted_turns_timing_round_trips_and_lands_in_results_jsonl(self):
        import dataclasses

        from pipecat.evals.results import (
            EvalScriptResult,
            EvalScriptTurnResult,
            EvalTurnTiming,
        )
        from pipecat.evals.suite import _result_from_dict

        timing = EvalTurnTiming(
            input_duration_ms=1200,
            llm_started_ms=300,
            first_token_ms=500,
            llm_response_ms=900,
            bot_started_speaking_ms=800,
            bot_speech_onset_ms=950,
            bot_metrics=[
                {"processor": "LLM#0", "ttfb_ms": 412, "processing_ms": None, "tokens": None}
            ],
        )
        result = EvalScriptResult(
            scenario_name="greet",
            passed=True,
            turns=[
                EvalScriptTurnResult(
                    turn_index=0, status="passed", duration_ms=1500, timing=timing
                ),
                EvalScriptTurnResult(turn_index=1),
            ],
        )
        # The worker writes the result as JSON; the suite rebuilds it, timing included.
        rebuilt = _result_from_dict(json.loads(json.dumps(dataclasses.asdict(result))))
        self.assertEqual(rebuilt, result)
        self.assertEqual(rebuilt.turns[0].timing.voice_to_voice_ms, 950)
        self.assertIsNone(rebuilt.turns[1].timing)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = EvalRun(
                bot="bot.py",
                scenario="greet",
                scenario_path=base / "greet.yaml",
                status="done",
                duration_ms=2000,
                result=result,
            )
            _append_result(base / "results.jsonl", run, "bot.py__greet", base, None)
            record = json.loads((base / "results.jsonl").read_text())
        self.assertTrue(record["passed"])
        self.assertEqual(
            record["turns"][0]["timing"],
            {
                "input_duration_ms": 1200,
                "llm_started_ms": 300,
                "first_token_ms": 500,
                "llm_response_ms": 900,
                "function_call_ms": None,
                "bot_started_speaking_ms": 800,
                "bot_speech_onset_ms": 950,
                "bot_stopped_speaking_ms": None,
                "bot_metrics": [
                    {"processor": "LLM#0", "ttfb_ms": 412, "processing_ms": None, "tokens": None}
                ],
                "voice_to_voice_ms": 950,
                "speech_padding_ms": 150,
            },
        )
        self.assertIsNone(record["turns"][1]["timing"])

    def test_results_jsonl_record_for_a_simulation_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = EvalRun(
                bot="flows/x.py",
                scenario="book",
                scenario_path=base / "book.yaml",
                kind=EvalKind.SIMULATION,
                attempts=3,
                attempt=2,
                status="done",
                duration_ms=1234,
                result=EvalSimulationResult(
                    simulation_name="book",
                    succeeded=False,
                    reason="no table",
                    turns=3,
                    ended_by="bot",
                    events_seen=[{"type": "llm_started"}],
                ),
            )
            _append_result(base / "results.jsonl", run, "flows_x.py__book__002", base, None)
            record = json.loads((base / "results.jsonl").read_text())
            self.assertEqual(record["scenario"], "book")
            self.assertEqual(record["kind"], "simulation")
            self.assertEqual(record["attempt"], 2)
            self.assertFalse(record["passed"])
            self.assertFalse(record["succeeded"])
            self.assertEqual(record["ended_by"], "bot")
            self.assertEqual(record["reason"], "no table")
            self.assertEqual(record["events_seen"], [{"type": "llm_started"}])
