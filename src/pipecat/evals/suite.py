#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Multi-bot eval suite runner.

An :class:`EvalManifest` lists bots to spawn and the scenarios to run
against each, scripted and simulated alike. An :class:`EvalSuite` spawns
each bot with its eval transport on its own port and drives it with the
harness in a subprocess, several at a time. ``pipecat eval suite`` is the
CLI in front of it; the release evals are a manifest plus that command.

Manifest format (YAML)::

    concurrency: 4
    repeat: 1                     # run each (bot, scenario) N times
    runs_dir: test-runs           # logs + recordings go to <runs_dir>/<timestamp>/
    record: false                 # record conversation audio
    cache_dir: null               # optional
    scenarios_dir: scenarios      # resolved relative to this manifest file
    # {python}=interpreter (default sys.executable), {bot}=bot path,
    # {port}=assigned per run by the suite runner
    spawn: "{python} {bot} -t eval --port {port}"
    suite:
      - bot: examples/voice/voice-cartesia.py
        scenarios: [simple_math, greeting]
      - bot: examples/voice/voice-openai.py
        scenarios: [simple_math, interruption]
      - bot: examples/vision/vision-openai.py
        runner_body: scenarios/vision-cat.json   # passed to the bot as --runner-body
        scenarios: [vision_describe]
      - bot: examples/flows/restaurant_reservation.py
        scenarios: [book_table]                  # a simulation: its file has a persona

A ``scenarios:`` entry names a scenario file of either kind, a scripted one or a
simulation, and the file says which (see
:func:`~pipecat.evals.scenario.load_scenario_file`). A name resolves under
``scenarios_dir`` with ``.yaml`` added and may carry a folder, as
``scripted/greeting``; a name ending in ``.yaml`` is a path relative to the
manifest instead. A simulation runs as many
times as its ``runs:`` says, and every run must pass.

An optional ``runner_body:`` (a JSON file, resolved relative to the manifest) is
passed to the bot as ``--runner-body``, supplying runner-args data it would
normally receive in a ``/start`` request body (e.g. a vision bot's image path).
The bot is spawned with the body file's directory as its working directory, so
relative paths inside the body (like an image) resolve next to the file.

Manifest-relative paths (``bot``/``bots_dir``, ``scenarios_dir``,
``runs_dir``) resolve relative to the manifest file, so a manifest is portable;
the same values passed as CLI overrides resolve against the working directory.

``repeat`` (or ``--repeat``) runs every (bot, scenario) pair N times, which is how
a flaky behavior gets measured rather than sampled: a bot that passes a scenario
half the time looks identical to a reliable one in a single pass. Attempts are
interleaved across bots and carry an :attr:`EvalRun.attempt` number that joins
their artifact filenames, so no attempt overwrites another's logs.
"""

import asyncio
import contextlib
import json
import os
import shlex
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from loguru import logger

from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalScriptResult,
    EvalScriptTurnResult,
    EvalSimulationMetricScore,
    EvalSimulationResult,
    EvalSimulationTurnVerdict,
)
from pipecat.evals.scenario import EvalKind, load_scenario_file
from pipecat.evals.session import EvalSessionParams
from pipecat.evals.simulation import EvalSimulationScenario
from pipecat.utils.base_object import BaseObject

DEFAULT_BASE_PORT = 7900
DEFAULT_CONCURRENCY = 4
# How long to wait for a freshly spawned bot to start listening (the harness
# retries the connect, so this doubles as readiness waiting).
BOT_CONNECT_TIMEOUT_S = 60.0
# How long to wait for a bot subprocess to exit after the harness asks it to
# stop (via eval-cancel) before escalating to terminate/kill.
BOT_STOP_TIMEOUT_S = 10.0
# Safety net for a hung harness worker. The harness's own per-expectation timeouts
# bound a healthy run far below this; the cap only catches a worker that wedges, so
# it can't hold a concurrency slot forever.
WORKER_SAFETY_TIMEOUT_S = 600.0
# Default spawn template; {python}/{bot}/{port} are substituted per run.
DEFAULT_SPAWN = "{python} {bot} -t eval --port {port}"
# What a scenario file may be named, wherever one is looked for.
SCENARIO_SUFFIXES = (".yaml", ".yml")

# The harness runs three sub-pipelines in-process and tags each one's logs with an
# ``eval_pipeline`` context value via logger.contextualize (see harness.py),
# independent of which TTS/STT/LLM service is used. Anything untagged (RTVI,
# connection, harness internals) falls through to "harness". The label is the
# human heading used for that pipeline's section in the debug log.
PIPELINE_LOG_LABELS = {
    "speech": "user speech logs",
    "transcription": "bot speech transcription logs",
    "judge": "judge logs",
    "persona": "persona LLM logs",
    "harness": "harness logs",
}
PIPELINE_LOG_CATEGORIES = tuple(PIPELINE_LOG_LABELS)


@contextlib.contextmanager
def capture_pipeline_logs(
    logs_dir: Path, prefix: str, *, name: str, enabled: bool
) -> Iterator[None]:
    """Capture the harness's logs for one run into a single ``<prefix>.debug.log``.

    The logs are buffered in memory and written on exit, one section per
    pipeline. The run is tagged with ``prefix`` and the sink filters on it,
    so concurrent runs never mix. Writes nothing unless ``enabled``.

    Args:
        logs_dir: Directory the ``<prefix>.debug.log`` is written to.
        prefix: Filename stem; also the ``eval_run`` id the sink filters on.
        name: Human test name shown in each section heading.
        enabled: When False, do nothing and write no file.
    """
    if not enabled:
        yield
        return

    buffers: dict[str, list[str]] = {cat: [] for cat in PIPELINE_LOG_CATEGORIES}

    def sink(message) -> None:
        cat = message.record["extra"].get("eval_pipeline", "harness")
        buffers.setdefault(cat, []).append(str(message))

    sink_id = logger.add(
        sink, level="DEBUG", filter=lambda r, rid=prefix: r["extra"].get("eval_run") == rid
    )
    try:
        with logger.contextualize(eval_run=prefix):
            yield
    finally:
        logger.remove(sink_id)
        sections = [
            f"===== {PIPELINE_LOG_LABELS[cat]}: {name} =====\n\n{''.join(buffers[cat])}"
            for cat in PIPELINE_LOG_CATEGORIES
            if buffers.get(cat)
        ]
        if sections:
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / f"{prefix}.debug.log").write_text("\n".join(sections))


def _append_result(
    results_path: Path,
    run: "EvalRun",
    stem: str,
    logs_dir: Path,
    record_dir: Path | None,
) -> None:
    """Append one JSON line describing a finished run, flushed at once.

    The record carries enough to compute pass rates and group failures by
    kind, plus paths to the artifacts. ``events_seen`` is included only for
    runs that did not pass, since that is what a failure is diagnosed from.
    """
    artifacts = {"log": str(logs_dir / f"{stem}.log")}
    for suffix, key in ((".eval.log", "eval_log"), (".debug.log", "debug_log")):
        path = logs_dir / f"{stem}{suffix}"
        if path.exists():
            artifacts[key] = str(path)
    if record_dir is not None and (record_dir / f"{stem}.wav").exists():
        artifacts["recording"] = str(record_dir / f"{stem}.wav")
    if run.kind == EvalKind.SIMULATION:
        record = _simulation_record(run, artifacts)
    else:
        record = _scenario_record(run, artifacts)

    with contextlib.suppress(OSError):
        with results_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _scenario_record(run: "EvalRun", artifacts: dict) -> dict:
    """The results.jsonl record of a scenario run."""
    result = run.result if isinstance(run.result, EvalScriptResult) else None
    record = {
        "bot": run.bot,
        "scenario": run.scenario,
        "kind": run.kind,
        "attempt": run.attempt,
        "passed": bool(result and result.passed and not result.skipped),
        "skipped": result.skipped if result else None,
        "error": run.error,
        "duration_ms": run.duration_ms,
        "failures": [
            {
                "turn_index": f.turn_index,
                "expectation_index": f.expectation_index,
                "event_name": f.event_name,
                "kind": f.kind,
                "reason": f.reason,
            }
            for f in (result.failures if result else [])
        ],
        "turns": [
            {"turn_index": t.turn_index, "status": t.status, "duration_ms": t.duration_ms}
            for t in (result.turns if result else [])
        ],
        "artifacts": artifacts,
    }
    if result is not None and not record["passed"]:
        record["events_seen"] = result.events_seen
    return record


def _simulation_record(run: "EvalRun", artifacts: dict) -> dict:
    """The results.jsonl record of a simulation run: how it ended, what the judge said, each metric's score, the conversation, and for a failed run the bot's events."""
    result = run.result if isinstance(run.result, EvalSimulationResult) else None
    record = {
        "bot": run.bot,
        "scenario": run.scenario,
        "kind": run.kind,
        "attempt": run.attempt,
        "passed": bool(result and result.passed),
        "succeeded": bool(result and result.succeeded),
        "error": run.error or (result.error if result else None),
        "ended_by": result.ended_by if result else "error",
        "turns": result.turns if result else 0,
        "metrics": [
            {
                "name": m.name,
                "score": m.score,
                "passed": m.passed,
                "min_quality": m.min_quality,
                "value": m.value,
                "reason": m.reason,
                "verdicts": [
                    {"turn": v.turn, "passed": v.passed, "reason": v.reason} for v in m.verdicts
                ],
            }
            for m in (result.metrics if result else [])
        ],
        "reason": result.reason if result else "",
        "end_call": result.end_call if result else None,
        "duration_ms": run.duration_ms,
        "messages": result.messages if result else [],
        "artifacts": artifacts,
    }
    if result is not None and not record["passed"]:
        record["events_seen"] = result.events_seen
    return record


def _simulation_result_from_dict(data: dict) -> EvalSimulationResult:
    """Rebuild a :class:`EvalSimulationResult` from the JSON a harness worker writes back."""
    return EvalSimulationResult(
        simulation_name=data["simulation_name"],
        succeeded=data["succeeded"],
        reason=data.get("reason", ""),
        error=data.get("error"),
        metrics=[
            EvalSimulationMetricScore(
                **{k: v for k, v in m.items() if k != "verdicts"},
                verdicts=[EvalSimulationTurnVerdict(**v) for v in m.get("verdicts", [])],
            )
            for m in data.get("metrics", [])
        ],
        messages=data.get("messages", []),
        turns=data.get("turns", 0),
        ended_by=data.get("ended_by", "error"),
        end_call=data.get("end_call"),
        duration_ms=data.get("duration_ms", 0),
        events_seen=data.get("events_seen", []),
        debug_log=data.get("debug_log", []),
    )


def _result_from_dict(data: dict) -> EvalScriptResult:
    """Rebuild an :class:`EvalScriptResult` from the JSON a harness worker writes back."""
    return EvalScriptResult(
        scenario_name=data["scenario_name"],
        passed=data["passed"],
        failures=[EvalAssertionFailure(**f) for f in data.get("failures", [])],
        turns=[
            EvalScriptTurnResult(
                turn_index=t["turn_index"],
                status=t.get("status", "not_run"),
                failures=[EvalAssertionFailure(**f) for f in t.get("failures", [])],
                duration_ms=t.get("duration_ms", 0),
            )
            for t in data.get("turns", [])
        ],
        duration_ms=data.get("duration_ms", 0),
        events_seen=data.get("events_seen", []),
        debug_log=data.get("debug_log", []),
        skipped=data.get("skipped"),
    )


def _resolve_scenario(name: str, base: Path, default_dir: Path) -> tuple[str, Path]:
    """A manifest entry's display name and file.

    A name with a YAML suffix is a path relative to the manifest; any other
    names a file under ``default_dir``, folder allowed (``scripted/greeting``).
    The display name is the bare stem either way.
    """
    if name.endswith(SCENARIO_SUFFIXES):
        return Path(name).stem, (base / name).resolve()
    return Path(name).name, (default_dir / f"{name}.yaml").resolve()


@dataclass
class EvalRun:
    """Mutable per-(bot, scenario) state, updated in place so a live display can read it.

    Parameters:
        bot: Display name — the manifest's ``bot:`` (suite) or the bot URL (run).
        scenario: Display name (the scenario or simulation, without ``.yaml``).
        scenario_path: Path to the scenario or simulation file.
        kind: ``script`` (played by :class:`~pipecat.evals.script_session.EvalScriptSession`)
            or ``simulation`` (:class:`~pipecat.evals.simulation_session.EvalSimulationSession`).
        attempts: How many times this (bot, scenario) pair runs: the manifest's
            ``repeat``, or a simulation's own ``runs``. Above 1, each attempt's
            artifacts carry its number.
        sweep: Whether the attempts come from a ``repeat`` (a measurement: the
            suite reports a rate and a failure is data) rather than from a
            simulation's ``runs`` (a requirement: every attempt must pass).
        bot_path: The bot to spawn (suite); ``None`` when connecting to ``bot_url``.
        bot_url: Connect here instead of spawning (used by ``pipecat eval run``).
        runner_body_path: Optional ``--runner-body`` JSON for the bot's runner args.
        attempt: 1-based attempt number when the suite repeats (see
            :attr:`EvalManifest.repeat`); always 1 for a single pass.
        status: ``pending``, ``running``, or ``done``.
        result: The outcome, once the run is done.
        error: Spawn/connection error message, if the run failed before producing a result.
        started_at: Monotonic start time, for the live elapsed counter.
        duration_ms: Wall-clock time the run took, in milliseconds.
    """

    bot: str
    scenario: str
    scenario_path: Path
    bot_path: Path | None = None
    bot_url: str | None = None
    runner_body_path: Path | None = None
    kind: EvalKind = EvalKind.SCRIPT
    attempts: int = 1
    sweep: bool = False
    attempt: int = 1
    status: str = "pending"
    result: EvalScriptResult | EvalSimulationResult | None = None
    error: str | None = None
    started_at: float | None = None
    duration_ms: int | None = None


@dataclass(frozen=True)
class _ManifestSettings:
    """A manifest's settings once the command line's overrides are applied."""

    bots_dir: Path
    scenarios_dir: Path
    runs_dir: Path | None
    spawn: str
    python: str
    concurrency: int
    repeat: int
    repeat_given: bool
    base_port: int
    record: bool
    cache_dir: str | None


@dataclass
class EvalManifest:
    """A parsed eval-suite manifest.

    Parameters:
        runs: The (bot, scenario) and (bot, simulation) runs to execute.
        spawn: Spawn command template (``{python}``/``{bot}``/``{port}`` substituted).
        python: Interpreter used to spawn each bot.
        concurrency: How many runs to execute at once.
        repeat: How many times to run each (bot, scenario) pair. Attempts are
            interleaved rather than grouped per bot, so every bot meets the same
            machine conditions in the same stretch of the sweep and a transient
            slowdown shows up as a band across all of them instead of a regression
            in whichever bot happened to be running.
        base_port: First port to assign; each run gets ``base_port + index``, so
            the reserved range widens with ``repeat`` (bots x scenarios x repeat
            ports from ``base_port`` up).
        runs_dir: Base for run output (a ``<name>/`` subdir is added), or ``None``.
        record: Whether to record conversation audio.
        cache_dir: Directory for cached synthesized user audio, or ``None``.
    """

    runs: list[EvalRun]
    spawn: str
    python: str
    concurrency: int
    repeat: int
    base_port: int
    runs_dir: Path | None
    record: bool
    cache_dir: str | None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        bots_dir: str | Path | None = None,
        scenarios_dir: str | Path | None = None,
        runs_dir: str | Path | None = None,
        spawn: str | None = None,
        python: str | None = None,
        concurrency: int | None = None,
        repeat: int | None = None,
        base_port: int | None = None,
        record: bool | None = None,
        cache_dir: str | None = None,
    ) -> "EvalManifest":
        """Parse a manifest YAML into an :class:`EvalManifest`.

        A keyword that is not ``None`` overrides the manifest's value, so the
        CLI wins. Manifest paths resolve against the manifest's directory,
        overrides against the working directory.

        Args:
            path: Path to the manifest YAML.
            bots_dir: Override for the manifest's ``bots_dir`` (bot paths are relative to it).
            scenarios_dir: Override for the manifest's ``scenarios_dir``.
            runs_dir: Override for the manifest's ``runs_dir`` (base for run output).
            spawn: Override for the spawn command template.
            python: Override for the interpreter used to spawn bots.
            concurrency: Override for how many runs execute at once.
            repeat: Override for how many times each (bot, scenario) pair runs.
                Set here or in the manifest, it also replaces each simulation's
                own ``runs``, a repeat of 1 included.
            base_port: Override for the first port assigned.
            record: Override for whether to record conversation audio.
            cache_dir: Override for the synthesized-audio cache directory.

        Returns:
            The parsed :class:`EvalManifest`.
        """
        path = Path(path).resolve()
        base = path.parent
        data = yaml.safe_load(path.read_text()) or {}
        settings = cls._settings(
            data,
            base,
            path,
            bots_dir=bots_dir,
            scenarios_dir=scenarios_dir,
            runs_dir=runs_dir,
            spawn=spawn,
            python=python,
            concurrency=concurrency,
            repeat=repeat,
            base_port=base_port,
            record=record,
            cache_dir=cache_dir,
        )
        return cls(
            runs=cls._runs(data, base, settings),
            spawn=settings.spawn,
            python=settings.python,
            concurrency=settings.concurrency,
            repeat=settings.repeat,
            base_port=settings.base_port,
            runs_dir=settings.runs_dir,
            record=settings.record,
            cache_dir=settings.cache_dir,
        )

    @classmethod
    def _settings(
        cls,
        data: dict,
        base: Path,
        path: Path,
        *,
        bots_dir: str | Path | None,
        scenarios_dir: str | Path | None,
        runs_dir: str | Path | None,
        spawn: str | None,
        python: str | None,
        concurrency: int | None,
        repeat: int | None,
        base_port: int | None,
        record: bool | None,
        cache_dir: str | None,
    ) -> "_ManifestSettings":
        """The manifest's settings with the overrides applied: an override wins, and its paths resolve against the working directory rather than the manifest's."""

        def dir_value(override, key: str, default: str) -> Path:
            if override is not None:
                return Path(override).resolve()
            return (base / str(data.get(key, default))).resolve()

        if runs_dir is not None:
            runs_dir_p: Path | None = Path(runs_dir).resolve()
        elif data.get("runs_dir"):
            runs_dir_p = (base / str(data["runs_dir"])).resolve()
        else:
            runs_dir_p = None
        # A repeat set anywhere, the command line or the manifest, decides every
        # run's attempts, a simulation's included, even when it is 1; absent, a
        # simulation runs as many times as its file says.
        repeat_given = repeat is not None or "repeat" in data
        repeat = repeat if repeat is not None else int(data.get("repeat", 1))
        if repeat < 1:
            raise ValueError(f"{path}: 'repeat' must be at least 1")
        return _ManifestSettings(
            bots_dir=dir_value(bots_dir, "bots_dir", "."),
            scenarios_dir=dir_value(scenarios_dir, "scenarios_dir", "scenarios"),
            runs_dir=runs_dir_p,
            spawn=spawn or str(data.get("spawn", DEFAULT_SPAWN)),
            python=python or str(data.get("python") or sys.executable),
            concurrency=(
                concurrency
                if concurrency is not None
                else int(data.get("concurrency", DEFAULT_CONCURRENCY))
            ),
            repeat=repeat,
            repeat_given=repeat_given,
            base_port=(
                base_port
                if base_port is not None
                else int(data.get("base_port", DEFAULT_BASE_PORT))
            ),
            record=record if record is not None else bool(data.get("record", False)),
            cache_dir=cache_dir if cache_dir is not None else data.get("cache_dir"),
        )

    @classmethod
    def _runs(cls, data: dict, base: Path, settings: "_ManifestSettings") -> list[EvalRun]:
        """The runs the ``suite:`` list describes, one per bot, scenario, and attempt.

        The scenario file says which kind it is. A simulation runs as many
        times as its file says unless a repeat makes the suite a measurement; a
        file that fails to load still gets its run, which reports the error.
        Attempts are attempt-major (bot A #1, bot B #1, ..., bot A #2), so a
        sweep spreads each attempt across the bots without fast ones waiting on
        slow ones.
        """
        runs: list[EvalRun] = []
        for item in data.get("suite", []):
            bot = str(item["bot"])
            bot_path = (settings.bots_dir / bot).resolve()
            # A body file is passed to the bot as --runner-body: runner-args data
            # it would normally get from a /start request (a vision bot's image).
            runner_body = item.get("runner_body")
            runner_body_path = (base / str(runner_body)).resolve() if runner_body else None
            for scenario in item.get("scenarios", []):
                name, scenario_path = _resolve_scenario(str(scenario), base, settings.scenarios_dir)
                kind, attempts = EvalKind.SCRIPT, settings.repeat
                try:
                    loaded = load_scenario_file(scenario_path)
                except (ValueError, FileNotFoundError):
                    loaded = None
                if isinstance(loaded, EvalSimulationScenario):
                    kind = EvalKind.SIMULATION
                    attempts = settings.repeat if settings.repeat_given else loaded.runs
                runs.append(
                    EvalRun(
                        bot=bot,
                        scenario=name,
                        bot_path=bot_path,
                        scenario_path=scenario_path,
                        runner_body_path=runner_body_path,
                        kind=kind,
                        attempts=attempts,
                        sweep=settings.repeat_given,
                    )
                )
        most = max((run.attempts for run in runs), default=1)
        if most > 1:
            runs = [
                replace(run, attempt=n)
                for n in range(1, most + 1)
                for run in runs
                if n <= run.attempts
            ]
        return runs


@dataclass(frozen=True)
class _RunFiles:
    """Where one run's artifacts go, all named by the run's prefix.

    Parameters:
        prefix: ``<bot>__<scenario>``, with the attempt number when the run repeats.
        log: The bot's output.
        harness_log: The harness worker's output, kept only when it crashed.
        trace: The harness's decision trace, ``<prefix>.eval.log``.
        config: The worker's config, the handoff in.
        result: The worker's result, the handoff out.
        record: The conversation recording, or ``None`` when not recording.
    """

    prefix: str
    log: Path
    harness_log: Path
    trace: Path
    config: Path
    result: Path
    record: Path | None

    @classmethod
    def for_run(cls, run: "EvalRun", logs_dir: Path, record_dir: Path | None) -> "_RunFiles":
        """The files of ``run`` under ``logs_dir`` and ``record_dir``.

        The bot is part of the prefix because one bot can run several scenarios
        at once; the attempt number joins it when the suite repeats, so no
        attempt writes over another's artifacts.
        """
        prefix = f"{run.bot.replace('/', '_')}__{run.scenario}"
        if run.attempts > 1:
            prefix += f"__{run.attempt:03d}"
        return cls(
            prefix=prefix,
            log=logs_dir / f"{prefix}.log",
            harness_log=logs_dir / f"{prefix}.harness.log",
            trace=logs_dir / f"{prefix}.eval.log",
            config=logs_dir / f"{prefix}.config.json",
            result=logs_dir / f"{prefix}.result.json",
            record=(record_dir / f"{prefix}.wav") if record_dir else None,
        )


class EvalSuite(BaseObject):
    """Runs the (bot, scenario) runs of an :class:`EvalManifest`, spawning each bot.

    Each bot gets its eval transport on its own port and is driven by the
    harness in a subprocess, several at a time up to the manifest's
    ``concurrency``. The runs are updated in place as they go, so a live
    display can read their progress.

    Event handlers available:

    - on_update: Called with an :class:`EvalRun` whenever that run changes status.
      Runs are mutated in place, so handlers run synchronously and must return
      promptly; they see the run as the change left it rather than however it has
      moved on since.

    Example::

        manifest = EvalManifest.load("manifest.yaml")
        suite = EvalSuite(manifest)
        suite.filter(pattern="voice")

        @suite.event_handler("on_update")
        async def on_update(suite, run):
            print(run.scenario, run.status)

        await suite.run(Path("logs"))
    """

    def __init__(self, manifest: EvalManifest):
        """Initialize the suite from a parsed manifest.

        Args:
            manifest: The parsed :class:`EvalManifest`; its runs become the suite's
                working set (narrowed by :meth:`filter`, executed by :meth:`run`).
        """
        super().__init__()

        self.manifest = manifest
        self.runs = list(manifest.runs)

        # Synchronous: an EvalRun is mutated in place as it executes, so a handler
        # deferred to a task would read whatever the run has since become. A run
        # that fails before it spawns reaches "done" with no await in between.
        self._register_event_handler("on_update", sync=True)

    def filter(
        self,
        *,
        pattern: str | None = None,
        scenario: str | None = None,
        kind: EvalKind | None = None,
    ) -> list[EvalRun]:
        """Keep only the runs matching a bot-name substring, a scenario name, and/or a kind.

        Args:
            pattern: Keep only runs whose bot name contains this substring.
            scenario: Keep only runs for this exact scenario name.
            kind: Keep only runs of this kind.

        Returns:
            The matching runs, in their original order.
        """
        runs = self.runs
        if pattern:
            runs = [r for r in runs if pattern in r.bot]
        if scenario:
            runs = [r for r in runs if r.scenario == scenario]
        if kind:
            runs = [r for r in runs if r.kind == kind]
        self.runs = runs
        return runs

    async def run(
        self,
        logs_dir: Path,
        *,
        record_dir: Path | None = None,
        results_path: Path | None = None,
        on_update: Callable[[EvalRun], None] | None = None,
        debug: bool = False,
        params: EvalSessionParams | None = None,
    ) -> None:
        """Run all of the suite's runs, in place, with the manifest's concurrency.

        Each run gets its own port (``base_port + index``). Runs come off one
        queue with no barrier between attempts, so a slow bot never holds up the
        rest.

        Args:
            logs_dir: Directory for per-run logs.
            record_dir: Directory for per-run conversation recordings, or ``None``.
            results_path: JSONL file to append one record per finished run, or
                ``None`` to write none. Each line is flushed as its run completes,
                so an interrupted sweep keeps everything already finished.
            on_update: Called whenever a run changes status, for live display.

                .. deprecated:: 1.9.0
                    Use the ``on_update`` event handler instead.
                    Will be removed in 2.0.0.

            debug: When True, save each run's combined ``<run>.debug.log``.
            params: How each run behaves; ``None`` for the defaults. The suite
                sets what it owns on each run's copy: the connect timeout, the
                recording, the manifest's cache directory, and stopping the bot
                it spawned.
        """
        logger.remove()  # keep stdout clean for the caller's display
        logs_dir.mkdir(parents=True, exist_ok=True)
        if record_dir:
            record_dir.mkdir(parents=True, exist_ok=True)
        if results_path is not None:
            results_path.parent.mkdir(parents=True, exist_ok=True)
        self._bound_cpu_threads()
        handler = self._add_legacy_update_callback(on_update) if on_update is not None else None
        sem = asyncio.Semaphore(self.manifest.concurrency)
        try:
            await asyncio.gather(
                *(
                    self._run_one(
                        run,
                        self.manifest.base_port + i,
                        logs_dir,
                        record_dir,
                        results_path,
                        sem,
                        debug,
                        params or EvalSessionParams(),
                    )
                    for i, run in enumerate(self.runs)
                )
            )
        finally:
            if handler is not None:
                self.remove_event_handler("on_update", handler)

    async def _run_one(
        self,
        run: EvalRun,
        port: int,
        logs_dir: Path,
        record_dir: Path | None,
        results_path: Path | None,
        sem: asyncio.Semaphore,
        debug: bool,
        params: EvalSessionParams,
    ) -> None:
        """Spawn one bot, run its scenario against it, and record the outcome on ``run``."""
        async with sem:
            files = _RunFiles.for_run(run, logs_dir, record_dir)
            run.status = "running"
            run.started_at = time.monotonic()
            await self._call_event_handler("on_update", run)
            bot: asyncio.subprocess.Process | None = None
            worker: asyncio.subprocess.Process | None = None
            try:
                run.error = self._missing_file(run)
                if run.error is not None:
                    return
                bot = await self._spawn_bot(run, port, files)
                worker = await self._run_harness(run, port, files, debug=debug, params=params)
            except Exception as e:
                # The worker reports its own failures in its result; this is a
                # problem on the suite's side (spawning, reading the result back).
                run.error = f"error: {type(e).__name__}: {e}"
                with contextlib.suppress(OSError):
                    files.trace.write_text(traceback.format_exc())
            finally:
                await self._finish(run, files, bot, worker, results_path, logs_dir, record_dir)

    def _missing_file(self, run: EvalRun) -> str | None:
        """Why the run cannot start, when one of its files is missing."""
        if run.bot_path is None or not run.bot_path.exists():
            return f"bot not found: {run.bot_path}"
        if not run.scenario_path.exists():
            return f"{run.kind} not found: {run.scenario_path}"
        if run.runner_body_path is not None and not run.runner_body_path.exists():
            return f"body not found: {run.runner_body_path}"
        return None

    async def _spawn_bot(
        self, run: EvalRun, port: int, files: "_RunFiles"
    ) -> asyncio.subprocess.Process:
        """Start the bot with its eval transport on ``port``, its output going to the bot log.

        A body file's directory is the bot's working directory, so relative
        paths inside the body (an image) resolve next to the file.
        """
        assert run.bot_path is not None
        cwd = str(run.runner_body_path.parent) if run.runner_body_path else None
        with files.log.open("wb") as logf:
            return await asyncio.create_subprocess_exec(
                *self._spawn_argv(run.bot_path, port, run.runner_body_path),
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )

    async def _run_harness(
        self,
        run: EvalRun,
        port: int,
        files: "_RunFiles",
        *,
        debug: bool,
        params: EvalSessionParams,
    ) -> asyncio.subprocess.Process:
        """Run the harness worker for this run and read its result back onto ``run``.

        The worker gets its config as a file and writes its result as one; a
        worker that times out or exits without a result leaves ``run.error``.
        """
        run_params = params.model_copy(
            update={
                "connect_timeout_s": BOT_CONNECT_TIMEOUT_S,
                "record_path": str(files.record) if files.record else None,
                "cache_dir": self.manifest.cache_dir,
                # The suite spawned this bot, so it is cancelled on teardown, which is
                # faster than the kill fallback.
                "stop_bot": True,
            }
        )
        config = {
            "scenario_path": str(run.scenario_path),
            "scenario_name": run.scenario,
            "bot_url": f"ws://localhost:{port}",
            "params": run_params.model_dump(),
            "debug": debug,
            "logs_dir": str(files.log.parent),
            "prefix": files.prefix,
            "result_path": str(files.result),
        }
        files.config.write_text(json.dumps(config))
        with contextlib.suppress(OSError):
            files.result.unlink()
        with files.harness_log.open("wb") as logf:
            worker = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pipecat.evals._session_subprocess",
                str(files.config),
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
            )
        try:
            await asyncio.wait_for(worker.wait(), timeout=WORKER_SAFETY_TIMEOUT_S)
        except TimeoutError:
            worker.kill()
            await worker.wait()
            run.error = f"error: harness worker timed out after {WORKER_SAFETY_TIMEOUT_S:.0f}s"
            return worker
        if worker.returncode != 0 or not files.result.exists():
            # The worker crashed before writing a result; its traceback is in the harness log.
            run.error = (
                f"error: harness worker exited {worker.returncode} (see {files.prefix}.harness.log)"
            )
            return worker
        data = json.loads(files.result.read_text())
        if run.kind == EvalKind.SIMULATION:
            run.result = _simulation_result_from_dict(data)
        else:
            run.result = _result_from_dict(data)
        return worker

    async def _finish(
        self,
        run: EvalRun,
        files: "_RunFiles",
        bot: asyncio.subprocess.Process | None,
        worker: asyncio.subprocess.Process | None,
        results_path: Path | None,
        logs_dir: Path,
        record_dir: Path | None,
    ) -> None:
        """Mark the run done, stop what is still running, and keep only the real artifacts."""
        # The duration is measured the way the live counter ticks, and excludes the teardown.
        if run.started_at is not None:
            run.duration_ms = int((time.monotonic() - run.started_at) * 1000)
        run.status = "done"
        await self._call_event_handler("on_update", run)
        # A cancelled suite (Ctrl+C) may leave the worker running; it must not outlive the suite.
        if worker is not None and worker.returncode is None:
            worker.kill()
            with contextlib.suppress(ProcessLookupError):
                await worker.wait()
        if bot is not None:
            await self._stop_bot(bot)
        # The worker's stdout is only the import banner on success; it is kept
        # when no result came back, since it then holds the traceback.
        if run.result is not None:
            with contextlib.suppress(OSError):
                files.harness_log.unlink()
        for handoff in (files.config, files.result):
            with contextlib.suppress(OSError):
                handoff.unlink()
        if run.result is not None and run.result.debug_log:
            files.trace.write_text("\n".join(run.result.debug_log) + "\n")
        if results_path is not None:
            _append_result(results_path, run, files.prefix, logs_dir, record_dir)

    def _bound_cpu_threads(self) -> None:
        """Cap each model's OpenMP threads to cores / concurrency, so concurrent transcriptions share the cores.

        CTranslate2 (CPU Whisper) honors ``OMP_NUM_THREADS`` and otherwise takes
        every core per model, which oversubscribes the CPU when several runs
        transcribe at once. The ONNX Runtime models do not use OpenMP, so for
        them this is a no-op. An explicit setting in the environment wins.
        """
        cores = os.cpu_count() or 1
        os.environ.setdefault(
            "OMP_NUM_THREADS", str(max(1, cores // max(1, self.manifest.concurrency)))
        )

    def _add_legacy_update_callback(self, on_update: Callable[[EvalRun], None]):
        """Register a bare ``on_update`` callback as an event handler; returns the handler, to remove after the run."""
        warnings.warn(
            "`on_update` is deprecated since 1.9.0 and will be removed in 2.0.0. "
            "Use the `on_update` event handler instead.",
            DeprecationWarning,
            stacklevel=3,
        )

        # Event handlers take the suite as their first argument; the callback
        # takes only the run.
        def forward_update(_suite: "EvalSuite", run: EvalRun) -> None:
            on_update(run)

        self.add_event_handler("on_update", forward_update)
        return forward_update

    def _spawn_argv(
        self, bot_path: Path, port: int, runner_body_path: Path | None = None
    ) -> list[str]:
        """The spawn argv, with ``{python}``, ``{bot}``, and ``{port}`` substituted per token so a path with spaces stays one entry."""
        subs = {"python": self.manifest.python, "bot": str(bot_path), "port": str(port)}
        argv = [tok.format(**subs) for tok in shlex.split(self.manifest.spawn)]
        if runner_body_path is not None:
            argv += ["--runner-body", str(runner_body_path)]
        return argv

    @staticmethod
    async def _stop_bot(proc: asyncio.subprocess.Process) -> None:
        """Wait for the bot to exit, then terminate and kill if it lingers."""
        if proc.returncode is not None:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=BOT_STOP_TIMEOUT_S)
            return
        except TimeoutError:
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=BOT_STOP_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()
