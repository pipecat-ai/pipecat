#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval suite's manifest parsing, per-run log capture, and run updates."""

import asyncio
import json
import os
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
    runner_body:
      path: bodies/cat.yaml
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
        self.assertEqual(special.runner_body_path, self.base / "bodies" / "cat.yaml")
        self.assertIsNone(special.runner_body)

    def test_runner_body_given_inline(self):
        self.manifest_path.write_text(
            "suite:\n"
            "  - bot: a.py\n"
            "    runner_body:\n"
            "      data: {model: gpt-4o-mini, question: hi}\n"
            "    scenarios: [x]\n"
        )
        run = EvalManifest.load(self.manifest_path).runs[0]
        self.assertEqual(run.runner_body, {"model": "gpt-4o-mini", "question": "hi"})
        self.assertIsNone(run.runner_body_path)

    def test_bare_runner_body_path_is_deprecated(self):
        self.manifest_path.write_text(
            "suite:\n  - bot: a.py\n    runner_body: bodies/cat.yaml\n    scenarios: [x]\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = EvalManifest.load(self.manifest_path).runs[0]
        self.assertEqual([w.category for w in caught], [DeprecationWarning])
        self.assertIn("runner_body: {path: <file>}", str(caught[0].message))
        self.assertEqual(run.runner_body_path, self.base / "bodies" / "cat.yaml")

    def test_runner_body_must_be_a_path_or_data(self):
        for bad in (
            "{path: a.yaml, data: {}}",
            "{}",
            "{file: a.yaml}",
            "[a.yaml]",
            "{data: a.yaml}",
        ):
            self.manifest_path.write_text(
                f"suite:\n  - bot: a.py\n    runner_body: {bad}\n    scenarios: [x]\n"
            )
            with self.assertRaises(ValueError, msg=bad) as ctx:
                EvalManifest.load(self.manifest_path)
            self.assertIn("'runner_body:'", str(ctx.exception))

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

    def test_an_entry_caps_its_own_concurrency(self):
        self.manifest_path.write_text(
            "suite:\n"
            "  - bot: a.py\n    concurrency: 2\n    scenarios: [x, y]\n"
            "  - bot: b.py\n    scenarios: [x]\n"
        )
        m = EvalManifest.load(self.manifest_path, concurrency=8)
        self.assertEqual([r.concurrency for r in m.runs], [2, 2, None])
        # The command line's cap is the suite's, not the entry's.
        self.assertEqual(m.concurrency, 8)

    def test_entry_concurrency_must_be_a_positive_integer(self):
        for bad in ("0", "-1", "two", "true", "1.5"):
            self.manifest_path.write_text(
                f"suite:\n  - bot: a.py\n    concurrency: {bad}\n    scenarios: [x]\n"
            )
            with self.assertRaises(ValueError, msg=bad) as ctx:
                EvalManifest.load(self.manifest_path)
            self.assertIn("'concurrency:'", str(ctx.exception))


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


class TestBotConcurrency(unittest.IsolatedAsyncioTestCase):
    """An entry holds one slot unless its cap says more; the suite's cap limits the whole."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        logger.remove()
        logger.add(sys.stderr)

    async def test_an_entry_runs_one_at_a_time_unless_its_cap_widens_it(self):
        runs = [
            EvalRun(
                bot=bot,
                scenario=f"s{i}",
                scenario_path=self.logs_dir / "s.yaml",
                bot_path=self.logs_dir / bot,
                concurrency=3 if bot == "wide.py" else None,
            )
            for bot in ("plain.py", "wide.py")
            for i in range(3)
        ]
        suite = EvalSuite(
            EvalManifest(
                runs=runs,
                spawn=DEFAULT_SPAWN,
                python=sys.executable,
                concurrency=4,
                repeat=1,
                base_port=7900,
                runs_dir=None,
                record=False,
                cache_dir=None,
            )
        )
        # Stand in for the bot and the harness: each run holds its slot for a
        # moment, and the test records how many of each bot were held at once.
        active: dict[str, int] = {}
        peak: dict[str, int] = {}

        async def spawn(run, port, files):
            active[run.bot] = active.get(run.bot, 0) + 1
            peak[run.bot] = max(peak.get(run.bot, 0), active[run.bot])
            await asyncio.sleep(0.05)
            active[run.bot] -= 1
            return None

        async def harness(run, port, files, *, debug, params):
            return None

        async def finish(run, files, bot, worker, results_path, logs_dir, record_dir):
            run.status = "done"

        suite._missing_file = lambda run: None
        suite._spawn_bot = spawn
        suite._run_harness = harness
        suite._finish = finish

        await suite.run(self.logs_dir)

        self.assertEqual(peak["plain.py"], 1)
        self.assertEqual(peak["wide.py"], 3)
        self.assertEqual([r.status for r in runs], ["done"] * 6)

    async def test_entries_take_slots_in_manifest_order_and_drain_their_queues(self):
        runs = [
            EvalRun(
                bot=bot,
                scenario=scenario,
                scenario_path=self.logs_dir / "s.yaml",
                bot_path=self.logs_dir / bot,
            )
            for bot, scenario in (
                ("a.py", "s1"),
                ("a.py", "s2"),
                ("a.py", "s3"),
                ("b.py", "s1"),
                ("c.py", "s1"),
                ("c.py", "s2"),
            )
        ]
        suite = EvalSuite(
            EvalManifest(
                runs=runs,
                spawn=DEFAULT_SPAWN,
                python=sys.executable,
                concurrency=1,
                repeat=1,
                base_port=7900,
                runs_dir=None,
                record=False,
                cache_dir=None,
            )
        )
        started: list[tuple[str, str]] = []

        async def spawn(run, port, files):
            started.append((run.bot, run.scenario))
            return None

        async def harness(run, port, files, *, debug, params):
            return None

        async def finish(run, files, bot, worker, results_path, logs_dir, record_dir):
            run.status = "done"

        suite._missing_file = lambda run: None
        suite._spawn_bot = spawn
        suite._run_harness = harness
        suite._finish = finish

        await suite.run(self.logs_dir)

        # With one slot, each entry drains before the next starts, in manifest order.
        self.assertEqual(
            started,
            [
                ("a.py", "s1"),
                ("a.py", "s2"),
                ("a.py", "s3"),
                ("b.py", "s1"),
                ("c.py", "s1"),
                ("c.py", "s2"),
            ],
        )


class TestRunFiles(unittest.TestCase):
    def test_prefix_carries_the_bot_and_the_name_when_the_entry_has_one(self):
        from pipecat.evals.suite import _RunFiles

        logs = Path("/logs")
        run = EvalRun(bot="turns/bot.py", scenario="scripted/turn", scenario_path=Path("x"))
        self.assertEqual(_RunFiles.for_run(run, logs, None).prefix, "turns_bot.py__scripted__turn")
        named = EvalRun(
            bot="turns/bot.py", name="groq/llama", scenario="scripted/turn", scenario_path=Path("x")
        )
        self.assertEqual(
            _RunFiles.for_run(named, logs, None).prefix, "turns_bot.py__groq_llama__scripted__turn"
        )
        repeated = EvalRun(
            bot="turns/bot.py",
            name="groq/llama",
            scenario="turn",
            scenario_path=Path("x"),
            attempts=2,
            attempt=2,
        )
        self.assertEqual(
            _RunFiles.for_run(repeated, logs, None).prefix, "turns_bot.py__groq_llama__turn__002"
        )


class TestSpawnWithRunnerBody(unittest.IsolatedAsyncioTestCase):
    """An inline body reaches the bot as a file; a body file sets the bot's directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        (self.base / "bodies").mkdir()
        # The "bot" records its argv and working directory, then exits.
        self.bot = self.base / "bot.py"
        self.bot.write_text(
            "import json, os, sys\nprint(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}))\n"
        )

    def tearDown(self):
        self._tmp.cleanup()

    async def _spawn(self, run: EvalRun) -> dict:
        from pipecat.evals.suite import _RunFiles

        suite = EvalSuite(
            EvalManifest(
                runs=[run],
                spawn="{python} {bot} --port {port}",
                python=sys.executable,
                concurrency=1,
                repeat=1,
                base_port=7900,
                runs_dir=None,
                record=False,
                cache_dir=None,
            )
        )
        files = _RunFiles.for_run(run, self.base / "logs", None)
        files.log.parent.mkdir(parents=True, exist_ok=True)
        proc = await suite._spawn_bot(run, 7900, files)
        await proc.wait()
        return json.loads(files.log.read_text())

    async def test_inline_body_is_written_for_the_bot(self):
        run = EvalRun(
            bot="bot.py",
            scenario="x",
            scenario_path=self.base / "x.yaml",
            bot_path=self.bot,
            runner_body={"model": "gpt-4o-mini"},
        )
        seen = await self._spawn(run)
        self.assertEqual(seen["argv"][:2], ["--port", "7900"])
        self.assertEqual(seen["argv"][2], "--runner-body")
        body_path = Path(seen["argv"][3])
        self.assertEqual(body_path.parent, self.base / "logs")
        self.assertEqual(json.loads(body_path.read_text()), {"model": "gpt-4o-mini"})
        # No body file to anchor it, so the bot runs where the suite does.
        self.assertEqual(seen["cwd"], os.getcwd())

    async def test_body_file_is_the_bots_directory(self):
        body = self.base / "bodies" / "cat.yaml"
        body.write_text("image_path: cat.jpg\n")
        run = EvalRun(
            bot="bot.py",
            scenario="x",
            scenario_path=self.base / "x.yaml",
            bot_path=self.bot,
            runner_body_path=body,
        )
        seen = await self._spawn(run)
        self.assertEqual(seen["argv"][2:], ["--runner-body", str(body)])
        self.assertEqual(Path(seen["cwd"]).resolve(), body.parent)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Simulations in a manifest's scenarios: list, and their results.jsonl records.
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402

from pipecat.evals.results import (  # noqa: E402
    EvalExpectationResult,
    EvalScriptResult,
    EvalScriptTurnResult,
    EvalSimulationMetricScore,
    EvalSimulationResult,
    EvalSimulationTurnVerdict,
)
from pipecat.evals.scenario import EvalKind  # noqa: E402
from pipecat.evals.suite import (  # noqa: E402
    _append_result,
    _result_from_dict,
    _simulation_result_from_dict,
)

SIMULATION = """
name: {name}
simulator: {{service: openai}}
scenarios:
  - name: {name}
    persona: "A caller."
    goal: "Get it done."
    success: "it got done"
    runs: {runs}
"""

SCRIPT = "name: {name}\nscenarios:\n  - name: {name}\n    turns: []\n"


class TestManifestSimulations(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        (self.base / "scenarios").mkdir()
        (self.base / "scenarios" / "book.yaml").write_text(SIMULATION.format(name="book", runs=3))
        (self.base / "scenarios" / "once.yaml").write_text(SIMULATION.format(name="once", runs=1))
        (self.base / "scenarios" / "greet.yaml").write_text(SCRIPT.format(name="greet"))

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
        self.assertEqual([r.attempt for r in by_name["book/book"]], [1, 2, 3])
        self.assertEqual([r.attempt for r in by_name["once/once"]], [1])
        self.assertEqual([r.attempt for r in by_name["greet/greet"]], [1])
        book = by_name["book/book"][0]
        self.assertEqual(book.kind, "simulation")
        self.assertEqual(book.attempts, 3)
        self.assertFalse(book.sweep)  # its runs are a requirement
        self.assertEqual(book.scenario_path, self.base / "scenarios" / "book.yaml")
        greet = by_name["greet/greet"][0]
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
        (self.base / "scenarios" / "scripted" / "greet.yaml").write_text(
            SCRIPT.format(name="greet")
        )
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [scripted/greet]\n")
        run = manifest.runs[0]
        self.assertEqual(run.scenario, "greet/greet")
        self.assertEqual(run.scenario_path, self.base / "scenarios" / "scripted" / "greet.yaml")

    def test_the_suite_filters_by_kind(self):
        from pipecat.evals.suite import EvalSuite

        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [greet, book]\n")
        runs = EvalSuite(manifest).filter(kind=EvalKind.SIMULATION)
        self.assertEqual({r.scenario for r in runs}, {"book/book"})
        self.assertEqual(len(runs), 3)

    def test_a_missing_scenario_still_gets_a_run(self):
        """Its kind can't be read, so it runs once as a scenario and reports the error."""
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [nope]\n")
        self.assertEqual(len(manifest.runs), 1)
        self.assertEqual(manifest.runs[0].scenario, "nope")
        self.assertEqual(manifest.runs[0].kind, "script")
        self.assertEqual(manifest.runs[0].attempts, 1)

    def test_a_file_contributes_a_run_per_scenario(self):
        """Each scenario runs under its own name and as its own kind."""
        (self.base / "scenarios" / "mixed.yaml").write_text(
            "name: mixed\n"
            "scenarios:\n"
            "  - name: hi\n    turns: []\n"
            "  - name: call\n    persona: p\n    goal: g\n    success: s\n    runs: 2\n"
        )
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [greet, mixed]\n")
        first = [r for r in manifest.runs if r.attempt == 1]
        self.assertEqual([r.scenario for r in first], ["greet/greet", "mixed/hi", "mixed/call"])
        self.assertEqual([r.kind for r in first], ["script", "script", "simulation"])
        self.assertEqual([r.attempts for r in first], [1, 1, 2])
        self.assertTrue(all(r.scenario_path.name == "mixed.yaml" for r in first[1:]))
        # A scenario's name is a file stem without the slash.
        self.assertEqual(first[1].stem, "mixed__hi")

    def test_the_suite_filters_by_scenario_name_or_either_half(self):
        from pipecat.evals.suite import EvalSuite

        (self.base / "scenarios" / "mixed.yaml").write_text(
            "name: mixed\nscenarios:\n  - name: hi\n    turns: []\n  - name: bye\n    turns: []\n"
        )
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [greet, mixed]\n")
        names = lambda runs: [r.scenario for r in runs]
        self.assertEqual(names(EvalSuite(manifest).filter(scenario="mixed/hi")), ["mixed/hi"])
        self.assertEqual(names(EvalSuite(manifest).filter(scenario="hi")), ["mixed/hi"])
        self.assertEqual(
            names(EvalSuite(manifest).filter(scenario="mixed")), ["mixed/hi", "mixed/bye"]
        )
        self.assertEqual(names(EvalSuite(manifest).filter(scenario="greet")), ["greet/greet"])

    def test_a_run_loads_its_own_scenario(self):
        (self.base / "scenarios" / "mixed.yaml").write_text(
            "name: mixed\nscenarios:\n  - name: hi\n    turns: []\n  - name: bye\n    turns: []\n"
        )
        manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [mixed]\n")
        # Built from a loaded file, the runs hold their scenarios already.
        self.assertTrue(all(r.loaded is not None for r in manifest.runs))
        self.assertEqual([r.load().name for r in manifest.runs], ["mixed/hi", "mixed/bye"])
        # A run that only knows its file and name reads the file.
        path = manifest.runs[0].scenario_path
        self.assertEqual(
            EvalRun(bot="b", scenario="mixed/bye", scenario_path=path).load().name, "mixed/bye"
        )
        with self.assertRaises(KeyError) as cm:
            EvalRun(bot="b", scenario="mixed/nope", scenario_path=path).load()
        self.assertIn("no scenario called 'mixed/nope'", str(cm.exception))

    def test_an_entry_name_labels_its_runs(self):
        manifest = self._manifest(
            "suite:\n"
            "  - bot: bot.py\n    name: openai/gpt-4o-mini\n    scenarios: [greet]\n"
            "  - bot: bot.py\n    name: groq/llama\n    scenarios: [greet]\n"
            "  - bot: other.py\n    scenarios: [greet]\n"
        )
        self.assertEqual(
            [r.label for r in manifest.runs], ["openai/gpt-4o-mini", "groq/llama", "other.py"]
        )
        self.assertEqual(
            [r.name for r in manifest.runs], ["openai/gpt-4o-mini", "groq/llama", None]
        )
        self.assertEqual([r.bot for r in manifest.runs], ["bot.py", "bot.py", "other.py"])
        # The pattern filter sees the name and the bot path alike.
        from pipecat.evals.suite import EvalSuite

        self.assertEqual(
            [r.label for r in EvalSuite(manifest).filter(pattern="groq")], ["groq/llama"]
        )
        self.assertEqual(len(EvalSuite(manifest).filter(pattern="bot.py")), 2)

    def test_two_entries_may_not_run_a_scenario_under_one_label(self):
        with self.assertRaises(ValueError) as cm:
            self._manifest(
                "suite:\n  - bot: bot.py\n    scenarios: [greet]\n  - bot: bot.py\n    scenarios: [greet]\n"
            )
        self.assertIn("'bot.py' runs 'greet/greet' twice", str(cm.exception))
        # The same bot on different scenarios is fine, as is a named second entry.
        self._manifest(
            "suite:\n  - bot: bot.py\n    scenarios: [greet]\n  - bot: bot.py\n    scenarios: [book]\n"
        )
        self._manifest(
            "suite:\n  - bot: bot.py\n    scenarios: [greet]\n"
            "  - bot: bot.py\n    name: again\n    scenarios: [greet]\n"
        )

    def test_an_entry_name_must_be_a_non_empty_string(self):
        for bad in ('""', "3", "[a]"):
            with self.assertRaises(ValueError, msg=bad) as cm:
                self._manifest(
                    f"suite:\n  - bot: bot.py\n    name: {bad}\n    scenarios: [greet]\n"
                )
            self.assertIn("'name:'", str(cm.exception))

    def test_a_flat_file_still_loads_and_warns(self):
        (self.base / "scenarios" / "old.yaml").write_text("name: old\nturns: []\n")
        with self.assertWarns(DeprecationWarning):
            manifest = self._manifest("suite:\n  - bot: bot.py\n    scenarios: [old]\n")
        self.assertEqual([r.scenario for r in manifest.runs], ["old"])


class TestScenarioRecords(unittest.TestCase):
    def _result(self) -> EvalScriptResult:
        return EvalScriptResult(
            scenario_name="greet",
            passed=True,
            failures=[],
            turns=[
                EvalScriptTurnResult(
                    turn_index=0,
                    status="passed",
                    expectations=[
                        EvalExpectationResult(0, "llm_marker", True, "◐"),
                        EvalExpectationResult(1, "llm_response", True, "Hi"),
                    ],
                    duration_ms=50,
                )
            ],
            duration_ms=60,
        )

    def test_result_roundtrips_through_the_worker_json(self):
        result = self._result()
        rebuilt = _result_from_dict(json.loads(json.dumps(dataclasses.asdict(result))))
        self.assertEqual(rebuilt, result)

    def test_results_jsonl_record_keeps_what_each_expectation_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = EvalRun(
                bot="voice/x.py",
                name="openai/gpt-4o-mini",
                scenario="greet",
                scenario_path=base / "greet.yaml",
                status="done",
                result=self._result(),
            )
            _append_result(base / "results.jsonl", run, "voice_x.py__greet", base, None)
            record = json.loads((base / "results.jsonl").read_text())
            self.assertTrue(record["passed"])
            self.assertEqual((record["bot"], record["name"]), ("voice/x.py", "openai/gpt-4o-mini"))
            self.assertEqual(
                record["turns"][0]["expectations"],
                [
                    {
                        "expectation_index": 0,
                        "event_name": "llm_marker",
                        "passed": True,
                        "matched": "◐",
                    },
                    {
                        "expectation_index": 1,
                        "event_name": "llm_response",
                        "passed": True,
                        "matched": "Hi",
                    },
                ],
            )
            self.assertNotIn("events_seen", record)


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
        rebuilt = _simulation_result_from_dict(json.loads(json.dumps(dataclasses.asdict(result))))
        self.assertEqual(rebuilt, result)

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
            self.assertEqual(record["name"], "flows/x.py")
            self.assertEqual(record["kind"], "simulation")
            self.assertEqual(record["attempt"], 2)
            self.assertFalse(record["passed"])
            self.assertFalse(record["succeeded"])
            self.assertEqual(record["ended_by"], "bot")
            self.assertEqual(record["reason"], "no table")
            self.assertEqual(record["events_seen"], [{"type": "llm_started"}])


class TestEntryQueues(unittest.TestCase):
    """Each entry gets its own queue, in manifest order, sized by its cap."""

    @staticmethod
    def _run(
        label: str, scenario: str, attempt: int = 1, concurrency: int | None = None
    ) -> EvalRun:
        return EvalRun(
            bot="bot.py",
            name=label,
            scenario=scenario,
            loaded=None,
            bot_path=Path("bot.py"),
            scenario_path=Path(f"{scenario}.yaml"),
            attempt=attempt,
            concurrency=concurrency,
        )

    def test_one_queue_per_entry_in_manifest_order(self):
        runs = [
            self._run("a", "s1"),
            self._run("a", "s2"),
            self._run("b", "s1"),
            self._run("a", "s3"),
            self._run("c", "s1"),
        ]
        queues = EvalSuite._entry_queues(runs)
        self.assertEqual(
            [[(r.label, r.scenario) for r in q] for _, q in queues],
            [[("a", "s1"), ("a", "s2"), ("a", "s3")], [("b", "s1")], [("c", "s1")]],
        )
        self.assertEqual([slots for slots, _ in queues], [1, 1, 1])

    def test_attempts_form_their_own_queues_attempt_major(self):
        runs = [
            self._run("a", "s1", 1),
            self._run("a", "s2", 1),
            self._run("b", "s1", 1),
            self._run("a", "s1", 2),
            self._run("a", "s2", 2),
            self._run("b", "s1", 2),
        ]
        queues = EvalSuite._entry_queues(runs)
        self.assertEqual(
            [[(r.label, r.scenario, r.attempt) for r in q] for _, q in queues],
            [
                [("a", "s1", 1), ("a", "s2", 1)],
                [("b", "s1", 1)],
                [("a", "s1", 2), ("a", "s2", 2)],
                [("b", "s1", 2)],
            ],
        )

    def test_an_entry_cap_is_its_slots_and_the_lowest_wins(self):
        runs = [
            self._run("a", "s1", concurrency=3),
            self._run("a", "s2", concurrency=2),
            self._run("b", "s1"),
        ]
        self.assertEqual([slots for slots, _ in EvalSuite._entry_queues(runs)], [2, 1])
