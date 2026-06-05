#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Multi-agent eval suite runner.

A *manifest* lists agents to spawn and the scenarios to run against each. This
module spawns each agent with its eval transport on its own port, drives it with
the harness (:func:`pipecat.evals.harness.run_scenario`), and runs several
concurrently. ``pipecat eval suite`` is the CLI in front of it; the release evals
are just a manifest plus that command.

Manifest format (YAML)::

    concurrency: 4
    runs_dir: test-runs           # logs + recordings go to <runs_dir>/<timestamp>/
    record: false                 # record conversation audio
    cache_dir: null               # optional
    scenarios_dir: scenarios      # resolved relative to this manifest file
    # {python}=interpreter (default sys.executable), {agent}=agent path,
    # {port}=assigned per run by the suite runner
    spawn: "{python} {agent} -t eval --port {port}"
    suite:
      - agent: examples/voice/voice-cartesia.py
        scenarios: [simple_math, greeting]
      - agent: examples/voice/voice-openai.py
        scenarios: [simple_math, interruption]
      - agent: examples/vision/vision-openai.py
        runner_body: scenarios/vision-cat.json   # passed to the agent as --runner-body
        scenarios: [vision_describe]

An optional ``runner_body:`` (a JSON file, resolved relative to the manifest) is
passed to the agent as ``--runner-body``, supplying runner-args data it would
normally receive in a ``/start`` request body (e.g. a vision bot's image path).
The agent is spawned with the body file's directory as its working directory, so
relative paths inside the body (like an image) resolve next to the file.

Manifest-relative paths (``agent``/``agents_dir``, ``scenarios_dir``,
``runs_dir``) resolve relative to the manifest file, so a manifest is portable;
the same values passed as CLI overrides resolve against the working directory.
"""

import asyncio
import shlex
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from pipecat.evals.harness import EvalResult, run_scenario

DEFAULT_BASE_PORT = 7900
DEFAULT_CONCURRENCY = 4
# How long to wait for a freshly spawned agent to start listening (the harness
# retries the connect, so this doubles as readiness waiting).
AGENT_READY_TIMEOUT_S = 60.0
# How long to wait for an agent subprocess to exit after the harness asks it to
# stop (via eval-cancel) before escalating to terminate/kill.
AGENT_STOP_TIMEOUT_S = 10.0
# Default spawn template; {python}/{agent}/{port} are substituted per run.
DEFAULT_SPAWN = "{python} {agent} -t eval --port {port}"


@dataclass
class SuiteRun:
    """Mutable per-(agent, scenario) state, updated in place so a live display can read it."""

    agent: str  # display name (the manifest's ``agent:`` value)
    scenario: str  # display name (the scenario, without .yaml)
    agent_path: Path
    scenario_path: Path
    runner_body_path: Path | None = None  # optional --runner-body JSON for the agent's runner args
    status: str = "pending"  # pending | running | done
    result: EvalResult | None = None
    error: str | None = None
    started_at: float | None = None
    duration_ms: int | None = None


@dataclass
class Manifest:
    """A parsed eval-suite manifest."""

    runs: list[SuiteRun]
    spawn: str
    python: str
    concurrency: int
    base_port: int
    runs_dir: Path | None  # base for run output (a <timestamp>/ subdir is added)
    record: bool  # record conversation audio
    cache_dir: str | None
    base_dir: Path


def load_manifest(
    path: str | Path,
    *,
    agents_dir: str | Path | None = None,
    scenarios_dir: str | Path | None = None,
    runs_dir: str | Path | None = None,
    spawn: str | None = None,
    python: str | None = None,
    concurrency: int | None = None,
    base_port: int | None = None,
    record: bool | None = None,
    cache_dir: str | None = None,
) -> Manifest:
    """Parse a manifest YAML into a :class:`Manifest`, with optional overrides.

    Any keyword that is not ``None`` overrides the corresponding manifest value
    (so the CLI wins), which means a manifest can be just a ``suite:`` list with
    everything else supplied on the command line. Manifest-relative paths resolve
    against the manifest's directory; path overrides resolve against the current
    working directory.
    """
    try:
        import yaml
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError("PyYAML is required for eval suites: pip install pyyaml") from e

    path = Path(path).resolve()
    base = path.parent
    data = yaml.safe_load(path.read_text()) or {}

    def dir_value(override, key: str, default: str) -> Path:
        # Override (from the CLI) resolves against cwd; the manifest value resolves
        # against the manifest file's directory.
        if override is not None:
            return Path(override).resolve()
        return (base / str(data.get(key, default))).resolve()

    agents_dir_p = dir_value(agents_dir, "agents_dir", ".")
    scenarios_dir_p = dir_value(scenarios_dir, "scenarios_dir", "scenarios")

    if runs_dir is not None:
        runs_dir_p: Path | None = Path(runs_dir).resolve()
    elif data.get("runs_dir"):
        runs_dir_p = (base / str(data["runs_dir"])).resolve()
    else:
        runs_dir_p = None

    spawn = spawn or str(data.get("spawn", DEFAULT_SPAWN))
    python = python or str(data.get("python") or sys.executable)
    concurrency = (
        concurrency
        if concurrency is not None
        else int(data.get("concurrency", DEFAULT_CONCURRENCY))
    )
    base_port = (
        base_port if base_port is not None else int(data.get("base_port", DEFAULT_BASE_PORT))
    )
    record = record if record is not None else bool(data.get("record", False))
    cache_dir = cache_dir if cache_dir is not None else data.get("cache_dir")

    runs: list[SuiteRun] = []
    for item in data.get("suite", []):
        agent = str(item["agent"])
        agent_path = (agents_dir_p / agent).resolve()
        # A body file (resolved relative to the manifest) is passed to the agent
        # as --runner-body, supplying runner-args data the agent would normally
        # get from a /start request (e.g. a vision bot's image path).
        runner_body = item.get("runner_body")
        runner_body_path = (base / str(runner_body)).resolve() if runner_body else None
        for scenario in item.get("scenarios", []):
            scenario = str(scenario)
            # A scenario may be a bare name (resolved under scenarios_dir) or a
            # path/.yaml relative to the manifest.
            if scenario.endswith((".yaml", ".yml")) or "/" in scenario:
                scenario_path = (base / scenario).resolve()
                name = Path(scenario).stem
            else:
                scenario_path = (scenarios_dir_p / f"{scenario}.yaml").resolve()
                name = scenario
            runs.append(
                SuiteRun(
                    agent=agent,
                    scenario=name,
                    agent_path=agent_path,
                    scenario_path=scenario_path,
                    runner_body_path=runner_body_path,
                )
            )

    return Manifest(
        runs=runs,
        spawn=spawn,
        python=python,
        concurrency=concurrency,
        base_port=base_port,
        runs_dir=runs_dir_p,
        record=record,
        cache_dir=cache_dir,
        base_dir=base,
    )


def filter_runs(
    runs: list[SuiteRun], *, pattern: str | None = None, scenario: str | None = None
) -> list[SuiteRun]:
    """Subset the runs by agent-name substring (``pattern``) and/or scenario name."""
    out = runs
    if pattern:
        out = [r for r in out if pattern in r.agent]
    if scenario:
        out = [r for r in out if r.scenario == scenario]
    return out


def _spawn_argv(
    manifest: Manifest, agent_path: Path, port: int, runner_body_path: Path | None = None
) -> list[str]:
    """Build the spawn argv, substituting {python}/{agent}/{port} per token.

    Substituting per token (rather than into the whole string) keeps a path with
    spaces in one argv entry. If the run has a body file, ``--runner-body <path>``
    is appended so the agent's runner picks it up.
    """
    subs = {"python": manifest.python, "agent": str(agent_path), "port": str(port)}
    argv = [tok.format(**subs) for tok in shlex.split(manifest.spawn)]
    if runner_body_path is not None:
        argv += ["--runner-body", str(runner_body_path)]
    return argv


async def _stop_agent(proc: asyncio.subprocess.Process) -> None:
    """Wait for the agent to exit, escalating to terminate/kill if it lingers.

    The harness sends ``eval-cancel`` on teardown, so the agent should already be
    cancelling its pipeline and exiting; wait for that graceful exit first.
    """
    if proc.returncode is not None:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=AGENT_STOP_TIMEOUT_S)
        return
    except TimeoutError:
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=AGENT_STOP_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()


async def _run_one(
    run: SuiteRun,
    port: int,
    manifest: Manifest,
    logs_dir: Path,
    record_dir: Path | None,
    sem: asyncio.Semaphore,
    on_update: Callable[[SuiteRun], None] | None,
) -> None:
    """Spawn one agent, run its scenario against it, and record the outcome on ``run``."""
    async with sem:
        # Include the agent in the filename: one agent can run several scenarios
        # concurrently, and they must not share a log/recording file.
        safe = f"{run.agent.replace('/', '_')}__{run.scenario}"
        log_path = logs_dir / f"{safe}.log"

        run.status = "running"
        run.started_at = time.monotonic()
        if on_update:
            on_update(run)

        proc: asyncio.subprocess.Process | None = None
        logf = None
        try:
            if not run.agent_path.exists():
                run.error = f"agent not found: {run.agent_path}"
                return
            if not run.scenario_path.exists():
                run.error = f"scenario not found: {run.scenario_path}"
                return
            if run.runner_body_path is not None and not run.runner_body_path.exists():
                run.error = f"body not found: {run.runner_body_path}"
                return

            from pipecat.evals.scenario import load_scenario

            # Spawn the agent with the body file's directory as cwd, so relative
            # paths inside the body (e.g. an image) resolve next to the file.
            cwd = str(run.runner_body_path.parent) if run.runner_body_path else None

            logf = log_path.open("wb")
            proc = await asyncio.create_subprocess_exec(
                *_spawn_argv(manifest, run.agent_path, port, run.runner_body_path),
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )

            scenario = load_scenario(run.scenario_path)
            record_path = str(record_dir / f"{safe}.wav") if record_dir else None
            run.result = await run_scenario(
                scenario,
                f"ws://localhost:{port}",
                connect_timeout_s=AGENT_READY_TIMEOUT_S,
                record_path=record_path,
                cache_dir=manifest.cache_dir,
            )
        except Exception as e:
            run.error = f"error: {e}"
        finally:
            # Stamp the wall-clock and flip to "done" BEFORE tearing the agent
            # down, measured the same way as a live counter (now - started_at), so
            # the displayed time matches what was ticking and excludes shutdown.
            if run.started_at is not None:
                run.duration_ms = int((time.monotonic() - run.started_at) * 1000)
            run.status = "done"
            if on_update:
                on_update(run)
            if proc is not None:
                await _stop_agent(proc)
            if logf is not None:
                logf.close()
            # Save the harness's own decision trace next to the agent log.
            if run.result is not None and run.result.debug_log:
                (logs_dir / f"{safe}.eval.log").write_text("\n".join(run.result.debug_log) + "\n")


async def run_suite(
    runs: list[SuiteRun],
    manifest: Manifest,
    logs_dir: Path,
    *,
    record_dir: Path | None = None,
    on_update: Callable[[SuiteRun], None] | None = None,
) -> None:
    """Run all ``runs`` with the manifest's concurrency, mutating each in place.

    Each run is spawned on its own port (``base_port + index``). ``logs_dir`` and
    (optional) ``record_dir`` are where per-run logs and recordings go.
    ``on_update`` is called whenever a run changes status, for live display.
    """
    logger.remove()  # keep stdout clean for the caller's display
    logs_dir.mkdir(parents=True, exist_ok=True)
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(manifest.concurrency)
    await asyncio.gather(
        *(
            _run_one(run, manifest.base_port + i, manifest, logs_dir, record_dir, sem, on_update)
            for i, run in enumerate(runs)
        )
    )
