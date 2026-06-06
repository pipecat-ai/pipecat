#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Multi-bot eval suite runner.

A *manifest* lists bots to spawn and the scenarios to run against each. This
module spawns each bot with its eval transport on its own port, drives it with
the harness (:func:`pipecat.evals.harness.run_scenario`), and runs several
concurrently. ``pipecat eval suite`` is the CLI in front of it; the release evals
are just a manifest plus that command.

Manifest format (YAML)::

    concurrency: 4
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

An optional ``runner_body:`` (a JSON file, resolved relative to the manifest) is
passed to the bot as ``--runner-body``, supplying runner-args data it would
normally receive in a ``/start`` request body (e.g. a vision bot's image path).
The bot is spawned with the body file's directory as its working directory, so
relative paths inside the body (like an image) resolve next to the file.

Manifest-relative paths (``bot``/``bots_dir``, ``scenarios_dir``,
``runs_dir``) resolve relative to the manifest file, so a manifest is portable;
the same values passed as CLI overrides resolve against the working directory.
"""

import asyncio
import contextlib
import shlex
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from pipecat.evals.harness import EvalResult, run_scenario

DEFAULT_BASE_PORT = 7900
DEFAULT_CONCURRENCY = 4
# How long to wait for a freshly spawned bot to start listening (the harness
# retries the connect, so this doubles as readiness waiting).
BOT_READY_TIMEOUT_S = 60.0
# How long to wait for a bot subprocess to exit after the harness asks it to
# stop (via eval-cancel) before escalating to terminate/kill.
BOT_STOP_TIMEOUT_S = 10.0
# Default spawn template; {python}/{bot}/{port} are substituted per run.
DEFAULT_SPAWN = "{python} {bot} -t eval --port {port}"

# The harness runs three sub-pipelines in-process and tags each one's logs with an
# ``eval_pipeline`` context value via logger.contextualize (see harness.py),
# independent of which TTS/STT/LLM service is used. Anything untagged (RTVI,
# connection, harness internals) falls through to "harness". The label is the
# human heading used for that pipeline's section in the debug log.
PIPELINE_LOG_LABELS = {
    "voice": "user speech logs",
    "transcription": "bot speech transcription logs",
    "judge": "judge logs",
    "harness": "harness logs",
}
PIPELINE_LOG_CATEGORIES = tuple(PIPELINE_LOG_LABELS)


@contextlib.contextmanager
def capture_pipeline_logs(
    logs_dir: Path, prefix: str, *, name: str, enabled: bool
) -> Iterator[None]:
    """Capture the harness's logs for one run and write a single ``<prefix>.debug.log``.

    Rather than one file per sub-pipeline, the harness's logs are buffered in
    memory (bounded: one scenario's worth) and written on exit as one file with a
    ``===== <label>: <name> =====`` section per pipeline (see PIPELINE_LOG_LABELS).
    The run is tagged with ``prefix`` as its ``eval_run`` id and the sink filters on
    it, so the suite's concurrent runs never mix into each other's file. ``name`` is
    the human test name shown in each section heading.

    A no-op (and writes nothing) when ``enabled`` is False, so the debug log only
    appears under ``--debug``.
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


@dataclass
class EvalRun:
    """Mutable per-(bot, scenario) state, updated in place so a live display can read it."""

    bot: str  # display name (suite: the manifest's ``bot:``; run: the bot URL)
    scenario: str  # display name (the scenario, without .yaml)
    scenario_path: Path
    bot_path: Path | None = None  # the bot to spawn (suite); None when connecting to bot_url
    bot_url: str | None = None  # connect here instead of spawning (``pipecat eval run``)
    runner_body_path: Path | None = None  # optional --runner-body JSON for the bot's runner args
    status: str = "pending"  # pending | running | done
    result: EvalResult | None = None
    error: str | None = None
    started_at: float | None = None
    duration_ms: int | None = None


@dataclass
class Manifest:
    """A parsed eval-suite manifest."""

    runs: list[EvalRun]
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
    bots_dir: str | Path | None = None,
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

    bots_dir_p = dir_value(bots_dir, "bots_dir", ".")
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

    runs: list[EvalRun] = []
    for item in data.get("suite", []):
        bot = str(item["bot"])
        bot_path = (bots_dir_p / bot).resolve()
        # A body file (resolved relative to the manifest) is passed to the bot
        # as --runner-body, supplying runner-args data the bot would normally
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
                EvalRun(
                    bot=bot,
                    scenario=name,
                    bot_path=bot_path,
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
    runs: list[EvalRun], *, pattern: str | None = None, scenario: str | None = None
) -> list[EvalRun]:
    """Subset the runs by bot-name substring (``pattern``) and/or scenario name."""
    out = runs
    if pattern:
        out = [r for r in out if pattern in r.bot]
    if scenario:
        out = [r for r in out if r.scenario == scenario]
    return out


def _spawn_argv(
    manifest: Manifest, bot_path: Path, port: int, runner_body_path: Path | None = None
) -> list[str]:
    """Build the spawn argv, substituting {python}/{bot}/{port} per token.

    Substituting per token (rather than into the whole string) keeps a path with
    spaces in one argv entry. If the run has a body file, ``--runner-body <path>``
    is appended so the bot's runner picks it up.
    """
    subs = {"python": manifest.python, "bot": str(bot_path), "port": str(port)}
    argv = [tok.format(**subs) for tok in shlex.split(manifest.spawn)]
    if runner_body_path is not None:
        argv += ["--runner-body", str(runner_body_path)]
    return argv


async def _stop_bot(proc: asyncio.subprocess.Process) -> None:
    """Wait for the bot to exit, escalating to terminate/kill if it lingers.

    The harness sends ``eval-cancel`` on teardown, so the bot should already be
    cancelling its pipeline and exiting; wait for that graceful exit first.
    """
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


async def _run_one(
    run: EvalRun,
    port: int,
    manifest: Manifest,
    logs_dir: Path,
    record_dir: Path | None,
    sem: asyncio.Semaphore,
    on_update: Callable[[EvalRun], None] | None,
    debug: bool,
) -> None:
    """Spawn one bot, run its scenario against it, and record the outcome on ``run``."""
    async with sem:
        # Include the bot in the filename: one bot can run several scenarios
        # concurrently, and they must not share a log/recording file.
        safe = f"{run.bot.replace('/', '_')}__{run.scenario}"
        log_path = logs_dir / f"{safe}.log"

        run.status = "running"
        run.started_at = time.monotonic()
        if on_update:
            on_update(run)

        proc: asyncio.subprocess.Process | None = None
        logf = None
        try:
            bot_path = run.bot_path
            if bot_path is None or not bot_path.exists():
                run.error = f"bot not found: {run.bot_path}"
                return
            if not run.scenario_path.exists():
                run.error = f"scenario not found: {run.scenario_path}"
                return
            if run.runner_body_path is not None and not run.runner_body_path.exists():
                run.error = f"body not found: {run.runner_body_path}"
                return

            from pipecat.evals.scenario import load_scenario

            # Spawn the bot with the body file's directory as cwd, so relative
            # paths inside the body (e.g. an image) resolve next to the file.
            cwd = str(run.runner_body_path.parent) if run.runner_body_path else None

            logf = log_path.open("wb")
            proc = await asyncio.create_subprocess_exec(
                *_spawn_argv(manifest, bot_path, port, run.runner_body_path),
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )

            scenario = load_scenario(run.scenario_path)
            record_path = str(record_dir / f"{safe}.wav") if record_dir else None
            # Under --debug, capture the harness's own logs (transcription / voice
            # / judge) into a single <safe>.debug.log, scoped by this run's id so
            # concurrent runs don't mix. The bot's own logs are captured
            # separately in <safe>.log above.
            with capture_pipeline_logs(logs_dir, safe, name=run.scenario, enabled=debug):
                run.result = await run_scenario(
                    scenario,
                    f"ws://localhost:{port}",
                    connect_timeout_s=BOT_READY_TIMEOUT_S,
                    record_path=record_path,
                    cache_dir=manifest.cache_dir,
                )
        except Exception as e:
            run.error = f"error: {e}"
        finally:
            # Stamp the wall-clock and flip to "done" BEFORE tearing the bot
            # down, measured the same way as a live counter (now - started_at), so
            # the displayed time matches what was ticking and excludes shutdown.
            if run.started_at is not None:
                run.duration_ms = int((time.monotonic() - run.started_at) * 1000)
            run.status = "done"
            if on_update:
                on_update(run)
            if proc is not None:
                await _stop_bot(proc)
            if logf is not None:
                logf.close()
            # Save the harness's own decision trace next to the bot log.
            if run.result is not None and run.result.debug_log:
                (logs_dir / f"{safe}.eval.log").write_text("\n".join(run.result.debug_log) + "\n")


async def run_suite(
    runs: list[EvalRun],
    manifest: Manifest,
    logs_dir: Path,
    *,
    record_dir: Path | None = None,
    on_update: Callable[[EvalRun], None] | None = None,
    debug: bool = False,
) -> None:
    """Run all ``runs`` with the manifest's concurrency, mutating each in place.

    Each run is spawned on its own port (``base_port + index``). ``logs_dir`` and
    (optional) ``record_dir`` are where per-run logs and recordings go.
    ``on_update`` is called whenever a run changes status, for live display.
    ``debug`` saves each run's combined ``<run>.debug.log``.
    """
    logger.remove()  # keep stdout clean for the caller's display
    logs_dir.mkdir(parents=True, exist_ok=True)
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(manifest.concurrency)
    await asyncio.gather(
        *(
            _run_one(
                run, manifest.base_port + i, manifest, logs_dir, record_dir, sem, on_update, debug
            )
            for i, run in enumerate(runs)
        )
    )
