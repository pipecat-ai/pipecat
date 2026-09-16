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
from unittest import mock

from loguru import logger

from pipecat.evals.session import EvalSessionParams
from pipecat.evals.suite import (
    DEFAULT_CONCURRENCY,
    DEFAULT_SPAWN,
    WORKER_SAFETY_TIMEOUT_S,
    WORKER_TIMEOUT_MARGIN_S,
    EvalManifest,
    EvalRun,
    EvalSuite,
    _RunFiles,
    capture_pipeline_logs,
    worker_timeout_s,
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

    def test_worker_timeout_is_unset_by_default_and_the_command_line_wins(self):
        m = EvalManifest.load(self.manifest_path)
        self.assertIsNone(m.worker_timeout)
        (self.base / "capped.yaml").write_text("worker_timeout: 900\nsuite: []\n")
        self.assertEqual(EvalManifest.load(self.base / "capped.yaml").worker_timeout, 900.0)
        self.assertEqual(
            EvalManifest.load(self.base / "capped.yaml", worker_timeout=120).worker_timeout, 120
        )

    def test_a_worker_timeout_must_be_positive(self):
        (self.base / "bad.yaml").write_text("worker_timeout: 0\nsuite: []\n")
        with self.assertRaises(ValueError):
            EvalManifest.load(self.base / "bad.yaml")
        with self.assertRaises(ValueError):
            EvalManifest.load(self.manifest_path, worker_timeout=-1)

    def test_an_entry_can_carry_an_env_for_its_bot(self):
        m = EvalManifest.load(self.manifest_path)
        self.assertEqual(m.runs[0].env, {})
        self.manifest_path.write_text(
            "suite:\n"
            "  - bot: bot.py\n"
            "    name: claude (low)\n"
            "    env: {EFFORT: low, RETRIES: 3}\n"
            "    scenarios: [greet]\n"
            "  - bot: bot.py\n"
            "    name: claude (high)\n"
            "    env: {EFFORT: high}\n"
            "    scenarios: [greet]\n"
        )
        m = EvalManifest.load(self.manifest_path)
        self.assertEqual([r.label for r in m.runs], ["claude (low)", "claude (high)"])
        # Values are strings, since that is what an environment holds.
        self.assertEqual(m.runs[0].env, {"EFFORT": "low", "RETRIES": "3"})
        self.assertEqual(m.runs[1].env, {"EFFORT": "high"})

    def test_an_entry_env_must_be_a_mapping(self):
        self.manifest_path.write_text(
            "suite:\n  - bot: a.py\n    env: [EFFORT]\n    scenarios: [x]\n"
        )
        with self.assertRaises(ValueError) as ctx:
            EvalManifest.load(self.manifest_path)
        self.assertIn("'env:'", str(ctx.exception))


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
    """A bot's own cap limits its runs; the suite's cap limits the rest."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.logs_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        logger.remove()
        logger.add(sys.stderr)

    async def test_capped_bot_runs_one_at_a_time_while_others_overlap(self):
        runs = [
            EvalRun(
                bot=bot,
                scenario=f"s{i}",
                scenario_path=self.logs_dir / "s.yaml",
                bot_path=self.logs_dir / bot,
                concurrency=1 if bot == "capped.py" else None,
            )
            for bot in ("capped.py", "free.py")
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

        self.assertEqual(peak["capped.py"], 1)
        self.assertEqual(peak["free.py"], 3)
        self.assertEqual([r.status for r in runs], ["done"] * 6)


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


class TestWorkerTimeout(unittest.TestCase):
    """The cap on a run's harness worker: an override as is, else derived with the constant as floor."""

    def _script(self, budgets: list[int | None]) -> EvalRun:
        """A run of a scripted scenario with one ``response`` expectation per turn, budgeted as given."""
        from pipecat.evals.script import EvalExpectation, EvalScriptScenario, EvalScriptTurn

        turns = [
            EvalScriptTurn(user="hi", expect=[EvalExpectation(event="response", within_ms=b)])
            for b in budgets
        ]
        loaded = EvalScriptScenario(name="s", turns=turns)
        return EvalRun(bot="b", scenario="s", scenario_path=Path("s.yaml"), loaded=loaded)

    def test_an_override_is_taken_as_is(self):
        run = self._script([None] * 30)
        self.assertEqual(worker_timeout_s(run, 60000, 42.0), 42.0)

    def test_a_short_script_gets_the_floor_plus_the_margin(self):
        run = self._script([30000, None])
        self.assertEqual(
            worker_timeout_s(run, 60000, None), WORKER_SAFETY_TIMEOUT_S + WORKER_TIMEOUT_MARGIN_S
        )

    def test_a_long_script_sums_its_turn_budgets(self):
        # 30 turns: 20 at the default 60s, 10 at an explicit 45s -> 1650s, over the floor.
        run = self._script([None] * 20 + [45000] * 10)
        self.assertEqual(worker_timeout_s(run, 60000, None), 1650 + WORKER_TIMEOUT_MARGIN_S)
        # The default timeout is the run's, not a constant.
        self.assertEqual(worker_timeout_s(run, 90000, None), 2250 + WORKER_TIMEOUT_MARGIN_S)

    def test_a_turn_budget_is_its_largest_within_ms(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            (base / "scenarios").mkdir()
            (base / "scenarios" / "long.yaml").write_text(
                "name: long\n"
                "scenarios:\n"
                "  - name: long\n"
                "    turns:\n"
                "      - user: hi\n"
                "        expect:\n"
                "          - {event: llm_started, within_ms: 5000}\n"
                "          - {event: response, within_ms: 400000}\n"
                "      - user: more\n"
                "        expect: [{event: response}]\n"
                "      - user: last\n"
                "        expect: [{event: response, within_ms: 300000}]\n"
            )
            (base / "manifest.yaml").write_text("suite:\n  - bot: bot.py\n    scenarios: [long]\n")
            run = EvalManifest.load(base / "manifest.yaml").runs[0]
        # 400 + 60 + 300 = 760s, over the floor.
        self.assertEqual(worker_timeout_s(run, 60000, None), 760 + WORKER_TIMEOUT_MARGIN_S)

    def test_a_scenario_that_did_not_load_gets_the_floor(self):
        run = EvalRun(bot="b", scenario="nope", scenario_path=Path("nope.yaml"))
        self.assertEqual(
            worker_timeout_s(run, 60000, None), WORKER_SAFETY_TIMEOUT_S + WORKER_TIMEOUT_MARGIN_S
        )

    def test_a_simulation_starts_from_its_max_duration(self):
        from pipecat.evals.scenario import EvalKind
        from pipecat.evals.simulation import EvalSimulationScenario

        def simulation(max_duration_s: float) -> EvalRun:
            loaded = EvalSimulationScenario(
                name="s",
                persona="A caller.",
                goal="Get it done.",
                success="it got done",
                max_duration_s=max_duration_s,
            )
            return EvalRun(
                bot="b",
                scenario="s",
                scenario_path=Path("s.yaml"),
                kind=EvalKind.SIMULATION,
                loaded=loaded,
            )

        # Below the floor: the floor, the judge still runs after the conversation.
        self.assertEqual(
            worker_timeout_s(simulation(120.0), 60000, None),
            WORKER_SAFETY_TIMEOUT_S + WORKER_TIMEOUT_MARGIN_S,
        )
        self.assertEqual(
            worker_timeout_s(simulation(1800.0), 60000, None), 1800 + WORKER_TIMEOUT_MARGIN_S
        )


class _NeverExits:
    """A stand-in for a spawned process that never finishes on its own."""

    def __init__(self):
        self.returncode = None
        self.killed = False

    async def wait(self):
        if self.killed:
            self.returncode = -9
            return self.returncode
        await asyncio.Event().wait()

    def kill(self):
        self.killed = True

    def terminate(self):
        self.killed = True


class TestSuiteSpawning(unittest.IsolatedAsyncioTestCase):
    """What the suite hands the subprocesses: the run's environment, and the worker's cap."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        (self.base / "bot.py").write_text("")
        (self.base / "s.yaml").write_text("name: s\nturns: []\n")
        self.logs_dir = self.base / "logs"
        self.logs_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _suite(self, **manifest_kwargs) -> tuple[EvalSuite, EvalRun]:
        run = EvalRun(
            bot="bot.py",
            name="claude (low)",
            env={"EFFORT": "low"},
            scenario="s",
            scenario_path=self.base / "s.yaml",
            bot_path=self.base / "bot.py",
        )
        manifest = EvalManifest(
            runs=[run],
            spawn=DEFAULT_SPAWN,
            python=sys.executable,
            concurrency=1,
            repeat=1,
            base_port=7900,
            runs_dir=None,
            record=False,
            cache_dir=None,
            **manifest_kwargs,
        )
        return EvalSuite(manifest), run

    async def test_the_runs_env_is_laid_over_the_suites(self):
        suite, run = self._suite()
        files = _RunFiles.for_run(run, self.logs_dir, None)
        seen: list[dict] = []

        async def fake_exec(*argv, **kwargs):
            seen.append(kwargs)
            return _NeverExits()

        with mock.patch.dict(os.environ, {"SUITE_VAR": "yes"}):
            with mock.patch("pipecat.evals.suite.asyncio.create_subprocess_exec", fake_exec):
                await suite._spawn_bot(run, 7900, files)
        self.assertEqual(len(seen), 1)
        env = seen[0]["env"]
        self.assertEqual(env["EFFORT"], "low")
        self.assertEqual(env["SUITE_VAR"], "yes")

    async def test_the_worker_is_killed_at_the_manifests_cap(self):
        suite, run = self._suite(worker_timeout=0.05)
        files = _RunFiles.for_run(run, self.logs_dir, None)
        worker = _NeverExits()

        async def fake_exec(*argv, **kwargs):
            return worker

        with mock.patch("pipecat.evals.suite.asyncio.create_subprocess_exec", fake_exec):
            await suite._run_harness(run, 7900, files, debug=False, params=EvalSessionParams())
        self.assertTrue(worker.killed)
        self.assertIn("harness worker timed out", run.error)
        self.assertIsNone(run.result)


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
