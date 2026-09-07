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
