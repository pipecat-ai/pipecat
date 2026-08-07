#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval suite's manifest parsing and per-run log capture."""

import tempfile
import unittest
from pathlib import Path

from loguru import logger

from pipecat.evals.suite import (
    DEFAULT_CONCURRENCY,
    DEFAULT_SPAWN,
    EvalManifest,
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

    def test_multi_scenario_file_expands_manifest_runs(self):
        (self.base / "collection.yaml").write_text(
            "scenarios:\n"
            "  - name: greeting\n"
            "    turns: [{user: hello}]\n"
            "  - name: capital_question\n"
            "    turns: [{user: 'What is the capital of France?'}]\n"
        )
        (self.base / "collection-manifest.yaml").write_text(
            "suite:\n  - bot: voice.py\n    scenarios: [collection.yaml]\n"
        )

        manifest = EvalManifest.load(self.base / "collection-manifest.yaml")

        self.assertEqual([run.scenario for run in manifest.runs], ["greeting", "capital_question"])
        self.assertEqual(
            [run.scenario_data.name for run in manifest.runs if run.scenario_data],
            ["greeting", "capital_question"],
        )

    def test_one_item_collection_uses_scenario_name(self):
        (self.base / "collection.yaml").write_text(
            "scenarios:\n  - name: internal_name\n    turns: []\n"
        )
        (self.base / "collection-manifest.yaml").write_text(
            "suite:\n  - bot: voice.py\n    scenarios: [collection.yaml]\n"
        )

        manifest = EvalManifest.load(self.base / "collection-manifest.yaml")

        self.assertEqual([run.scenario for run in manifest.runs], ["internal_name"])

    def test_malformed_scenario_is_deferred_to_run_validation(self):
        (self.base / "malformed.yaml").write_text("name: malformed\nturns: [\n")
        (self.base / "malformed-manifest.yaml").write_text(
            "suite:\n  - bot: voice.py\n    scenarios: [malformed.yaml]\n"
        )

        manifest = EvalManifest.load(self.base / "malformed-manifest.yaml")

        self.assertEqual(len(manifest.runs), 1)
        self.assertEqual(manifest.runs[0].scenario, "malformed")
        self.assertIsNone(manifest.runs[0].scenario_data)


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


if __name__ == "__main__":
    unittest.main()
