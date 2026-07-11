#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Multi-bot eval suite runner.

An :class:`EvalManifest` lists bots to spawn and the scenarios to run against
each; an :class:`EvalSuite` spawns each bot with its eval transport on its own
port, drives it with the harness (:meth:`pipecat.evals.harness.EvalSession.from_scenario`),
and runs several concurrently. ``pipecat eval suite`` is the CLI in front of it;
the release evals are just a manifest plus that command.

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
import os
import shlex
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from pipecat.evals.harness import DEFAULT_EVENT_TIMEOUT_MS, EvalResult, EvalSession

DEFAULT_BASE_PORT = 7900
DEFAULT_CONCURRENCY = 4
# How long to wait for a freshly spawned bot to start listening (the harness
# retries the connect, so this doubles as readiness waiting).
BOT_CONNECT_TIMEOUT_S = 60.0
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
    "speech": "user speech logs",
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
    ``===== <label>: <name> =====`` section per pipeline (see ``PIPELINE_LOG_LABELS``).
    The run is tagged with ``prefix`` as its ``eval_run`` id and the sink filters on
    it, so the suite's concurrent runs never mix into each other's file. A no-op
    (and writes nothing) when ``enabled`` is False, so the debug log only appears
    under ``--debug``.

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


@dataclass
class EvalRun:
    """Mutable per-(bot, scenario) state, updated in place so a live display can read it.

    Parameters:
        bot: Display name — the manifest's ``bot:`` (suite) or the bot URL (run).
        scenario: Display name (the scenario, without ``.yaml``).
        scenario_path: Path to the scenario file.
        bot_path: The bot to spawn (suite); ``None`` when connecting to ``bot_url``.
        bot_url: Connect here instead of spawning (used by ``pipecat eval run``).
        runner_body_path: Optional ``--runner-body`` JSON for the bot's runner args.
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
    status: str = "pending"
    result: EvalResult | None = None
    error: str | None = None
    started_at: float | None = None
    duration_ms: int | None = None


@dataclass
class EvalManifest:
    """A parsed eval-suite manifest.

    Parameters:
        runs: The (bot, scenario) runs to execute.
        spawn: Spawn command template (``{python}``/``{bot}``/``{port}`` substituted).
        python: Interpreter used to spawn each bot.
        concurrency: How many runs to execute at once.
        base_port: First port to assign; each run gets ``base_port + index``.
        runs_dir: Base for run output (a ``<name>/`` subdir is added), or ``None``.
        record: Whether to record conversation audio.
        cache_dir: Directory for cached synthesized user audio, or ``None``.
    """

    runs: list[EvalRun]
    spawn: str
    python: str
    concurrency: int
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
        base_port: int | None = None,
        record: bool | None = None,
        cache_dir: str | None = None,
    ) -> "EvalManifest":
        """Parse a manifest YAML into an :class:`EvalManifest`, with optional overrides.

        Any keyword that is not ``None`` overrides the corresponding manifest value
        (so the CLI wins), which means a manifest can be just a ``suite:`` list with
        everything else supplied on the command line. Manifest-relative paths resolve
        against the manifest's directory; path overrides resolve against the current
        working directory.

        Args:
            path: Path to the manifest YAML.
            bots_dir: Override for the manifest's ``bots_dir`` (bot paths are relative to it).
            scenarios_dir: Override for the manifest's ``scenarios_dir``.
            runs_dir: Override for the manifest's ``runs_dir`` (base for run output).
            spawn: Override for the spawn command template.
            python: Override for the interpreter used to spawn bots.
            concurrency: Override for how many runs execute at once.
            base_port: Override for the first port assigned.
            record: Override for whether to record conversation audio.
            cache_dir: Override for the synthesized-audio cache directory.

        Returns:
            The parsed :class:`EvalManifest`.
        """
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

        return cls(
            runs=runs,
            spawn=spawn,
            python=python,
            concurrency=concurrency,
            base_port=base_port,
            runs_dir=runs_dir_p,
            record=record,
            cache_dir=cache_dir,
        )


class EvalSuite:
    """Runs the (bot, scenario) runs of an :class:`EvalManifest`, spawning each bot.

    Spawns each bot with its eval transport on its own port, drives it with the
    harness (:meth:`pipecat.evals.harness.EvalSession.from_scenario`), and runs several
    concurrently (up to the manifest's ``concurrency``). The runs are mutated in
    place as they execute so a live display can read their progress.

    Example::

        manifest = EvalManifest.load("manifest.yaml")
        suite = EvalSuite(manifest)
        suite.filter(pattern="voice")
        await suite.run(Path("logs"))
    """

    def __init__(self, manifest: EvalManifest):
        """Initialize the suite from a parsed manifest.

        Args:
            manifest: The parsed :class:`EvalManifest`; its runs become the suite's
                working set (narrowed by :meth:`filter`, executed by :meth:`run`).
        """
        self.manifest = manifest
        self.runs = list(manifest.runs)

    def filter(self, *, pattern: str | None = None, scenario: str | None = None) -> list[EvalRun]:
        """Subset the suite's runs by bot-name substring and/or scenario name.

        Narrows :attr:`runs` in place (and returns it) so only matching runs are
        executed and displayed.

        Args:
            pattern: Keep only runs whose bot name contains this substring.
            scenario: Keep only runs for this exact scenario name.

        Returns:
            The matching runs, in their original order.
        """
        runs = self.runs
        if pattern:
            runs = [r for r in runs if pattern in r.bot]
        if scenario:
            runs = [r for r in runs if r.scenario == scenario]
        self.runs = runs
        return runs

    async def run(
        self,
        logs_dir: Path,
        *,
        record_dir: Path | None = None,
        on_update: Callable[[EvalRun], None] | None = None,
        debug: bool = False,
        use_cache: bool = True,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
    ) -> None:
        """Run all of the suite's runs with the manifest's concurrency, in place.

        Each run is spawned on its own port (``base_port + index``).

        Args:
            logs_dir: Directory for per-run logs.
            record_dir: Directory for per-run conversation recordings, or ``None``.
            on_update: Called whenever a run changes status, for live display.
            debug: When True, save each run's combined ``<run>.debug.log``.
            use_cache: When False, ignore cached user audio and force fresh synthesis.
            default_timeout_ms: Per-expectation budget for expectations without
                their own ``within_ms``. Defaults to 60s.
        """
        logger.remove()  # keep stdout clean for the caller's display
        logs_dir.mkdir(parents=True, exist_ok=True)
        if record_dir:
            record_dir.mkdir(parents=True, exist_ok=True)

        # Bound per-model CPU threads so concurrent CPU transcriptions share the
        # cores instead of each spawning ~all-cores of OpenMP threads. This bites
        # when the transcriber is (CPU) Whisper: CTranslate2 honors OMP_NUM_THREADS
        # and otherwise grabs every core per model, so N lockstep scenarios whose
        # first transcriptions fire simultaneously oversubscribe the CPU N-fold
        # (the first turn crawls, then speeds up only once the runs drift out of
        # sync). Capping each to cores/concurrency keeps total threads ~= cores on
        # every turn. The ONNX Runtime models (Moonshine, Kokoro) don't use OpenMP
        # by default, so for them this is a harmless no-op. setdefault respects an
        # explicit override. Set before any transcriber loads its model.
        cores = os.cpu_count() or 1
        os.environ.setdefault(
            "OMP_NUM_THREADS", str(max(1, cores // max(1, self.manifest.concurrency)))
        )

        sem = asyncio.Semaphore(self.manifest.concurrency)
        await asyncio.gather(
            *(
                self._run_one(
                    run,
                    self.manifest.base_port + i,
                    logs_dir,
                    record_dir,
                    sem,
                    on_update,
                    debug,
                    use_cache,
                    default_timeout_ms,
                )
                for i, run in enumerate(self.runs)
            )
        )

    async def _run_one(
        self,
        run: EvalRun,
        port: int,
        logs_dir: Path,
        record_dir: Path | None,
        sem: asyncio.Semaphore,
        on_update: Callable[[EvalRun], None] | None,
        debug: bool,
        use_cache: bool,
        default_timeout_ms: int,
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

                from pipecat.evals.scenario import EvalScenario

                # Spawn the bot with the body file's directory as cwd, so relative
                # paths inside the body (e.g. an image) resolve next to the file.
                cwd = str(run.runner_body_path.parent) if run.runner_body_path else None

                logf = log_path.open("wb")
                proc = await asyncio.create_subprocess_exec(
                    *self._spawn_argv(bot_path, port, run.runner_body_path),
                    stdout=logf,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=cwd,
                )

                scenario = EvalScenario.load(run.scenario_path)
                record_path = str(record_dir / f"{safe}.wav") if record_dir else None
                # Under --debug, capture the harness's own logs (transcription / voice
                # / judge) into a single <safe>.debug.log, scoped by this run's id so
                # concurrent runs don't mix. The bot's own logs are captured
                # separately in <safe>.log above.
                with capture_pipeline_logs(logs_dir, safe, name=run.scenario, enabled=debug):
                    run.result = await EvalSession.from_scenario(
                        scenario,
                        f"ws://localhost:{port}",
                        connect_timeout_s=BOT_CONNECT_TIMEOUT_S,
                        default_timeout_ms=default_timeout_ms,
                        record_path=record_path,
                        cache_dir=self.manifest.cache_dir,
                        use_cache=use_cache,
                        # The suite spawns a bot per run, so cancel it on teardown to
                        # shut it down gracefully (faster than the kill fallback).
                        stop_bot=True,
                    ).run()
            except Exception as e:
                # Errors raised inside EvalSession.run() are caught there and
                # returned as a structured result; this catches the rest (scenario
                # load, building the judge/voice/transcriber sub-pipelines, which
                # load local models and can fail under load). Keep the exception
                # type and stash the full traceback in <safe>.eval.log so the
                # cause is recoverable instead of vanishing as a bare "error: ".
                run.error = f"error: {type(e).__name__}: {e}"
                with contextlib.suppress(OSError):
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    (logs_dir / f"{safe}.eval.log").write_text(traceback.format_exc())
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
                    await self._stop_bot(proc)
                if logf is not None:
                    logf.close()
                # Save the harness's own decision trace next to the bot log.
                if run.result is not None and run.result.debug_log:
                    (logs_dir / f"{safe}.eval.log").write_text(
                        "\n".join(run.result.debug_log) + "\n"
                    )

    def _spawn_argv(
        self, bot_path: Path, port: int, runner_body_path: Path | None = None
    ) -> list[str]:
        """Build the spawn argv, substituting {python}/{bot}/{port} per token.

        Substituting per token (rather than into the whole string) keeps a path with
        spaces in one argv entry. If the run has a body file, ``--runner-body <path>``
        is appended so the bot's runner picks it up.
        """
        subs = {"python": self.manifest.python, "bot": str(bot_path), "port": str(port)}
        argv = [tok.format(**subs) for tok in shlex.split(self.manifest.spawn)]
        if runner_body_path is not None:
            argv += ["--runner-body", str(runner_body_path)]
        return argv

    @staticmethod
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
