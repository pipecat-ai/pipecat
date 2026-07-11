#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``pipecat eval`` typer commands.

Thin CLI wrappers over :mod:`pipecat.evals` — all the load/run logic lives in
that module. Mounted as the ``eval`` subcommand of the ``pipecat`` CLI (see
:mod:`pipecat.cli.main`) and also reachable as ``python -m pipecat.evals``.
"""

import asyncio
import contextlib
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from pipecat.evals.harness import EvalSession, EvalTurnProgress
from pipecat.evals.scenario import EvalScenario, describe_config
from pipecat.evals.suite import (
    EvalManifest,
    EvalRun,
    EvalSuite,
    capture_pipeline_logs,
)

_console = Console()

# Columns of clearance to keep between a truncated detail and the terminal edge,
# so a long llm_response is visibly cut off rather than butting against the side.
_DETAIL_RIGHT_MARGIN = 12


def _terminal_width() -> int:
    if _console.is_terminal:
        return _console.width
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def _fit_detail(detail: str, used_cols: int) -> str:
    """Collapse detail to a single line and truncate to fit the terminal.

    ``used_cols`` is what the line already spends on indent + badge + event name
    + separator. The result ends well before the terminal edge.
    """
    avail = max(20, _terminal_width() - used_cols - _DETAIL_RIGHT_MARGIN)
    one_line = " ".join(detail.split())
    if len(one_line) > avail:
        one_line = one_line[: avail - 1].rstrip() + "…"
    return one_line


def _format_detail(p: EvalTurnProgress) -> str:
    """Detail text for a resolved expectation.

    Matched prose (the bot's output) is quoted to set it apart from failure
    reasons; a matched ``function_call`` signature and failure reasons are left
    as-is.
    """
    if p.status == "matched" and p.event_name != "function_call":
        return f'"{p.detail}"'
    return p.detail


eval_app = typer.Typer(
    name="eval",
    help="Run behavioral evals against a Pipecat bot",
    no_args_is_help=True,
)


@eval_app.callback()
def _eval_callback() -> None:
    """Anchor for the subcommand structure.

    Required so typer treats ``run`` (and future verbs like ``list``) as
    explicit subcommands rather than collapsing the single-command case
    into a flat app.
    """


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(s: str) -> str:
    return _color(s, "32")


def _red(s: str) -> str:
    return _color(s, "31")


def _dim(s: str) -> str:
    return _color(s, "2")


def _print_progress(p: EvalTurnProgress) -> None:
    """Print a per-turn / per-expectation line (verbose mode)."""
    if p.status == "turn":
        label = f'"{p.event_name}"' if p.event_name else "(observe)"
        print(f"      {_dim(f'turn {p.turn_index}')} → {label}")
    else:
        badge = _green("✓") if p.status == "matched" else _red("✗")
        line = f"        {badge} {p.event_name}"
        if p.detail:
            detail = _format_detail(p)
            used = 8 + 2 + len(p.event_name) + 3  # indent + badge + name + " — "
            line += f" {_dim(f'— {_fit_detail(detail, used)}')}"
        print(line)


def _record_path(record_dir: str | None, scenario_name: str) -> str | None:
    """Per-scenario recording path under ``record_dir``, or None when recording is off."""
    if not record_dir:
        return None
    return str(Path(record_dir) / f"{scenario_name}.wav")


def _build_scenario_runs(paths: list[Path], bot_url: str) -> list[EvalRun]:
    """Build an EvalRun per scenario YAML, each run against ``bot_url`` (no spawn).

    A scenario that fails to load becomes an EvalRun already marked done with an
    error, so it shows in the dashboard and the final tally like any other failure.
    """
    runs: list[EvalRun] = []
    for path in paths:
        try:
            scenario = EvalScenario.load(path)
        except (ValueError, FileNotFoundError) as e:
            run = EvalRun(bot=bot_url, scenario=path.stem, scenario_path=path, bot_url=bot_url)
            run.status = "done"
            run.error = f"failed to load: {e}"
            runs.append(run)
            continue
        runs.append(
            EvalRun(bot=bot_url, scenario=scenario.name, scenario_path=path, bot_url=bot_url)
        )
    return runs


async def _execute_scenario(
    run: EvalRun,
    *,
    audio: bool,
    record_dir: str,
    cache_dir: str | None,
    use_cache: bool,
    default_timeout_ms: int,
    logs_dir: str,
    debug: bool,
    stop_bot: bool,
    trigger_disconnect: bool,
    on_progress,
) -> None:
    """Run one scenario against its ``bot_url``, updating ``run`` in place.

    The ``eval run`` counterpart to the suite's _run_one: it connects to a fixed
    URL instead of spawning, always writes the decision trace (``<scenario>.eval.log``)
    and, under ``--debug``, the combined ``<scenario>.debug.log``.
    """
    run.status = "running"
    run.started_at = time.monotonic()
    url = run.bot_url
    assert url is not None  # always set by _build_scenario_runs
    try:
        scenario = EvalScenario.load(run.scenario_path)
        record_path = _record_path(record_dir, run.scenario) if audio else None
        with capture_pipeline_logs(Path(logs_dir), run.scenario, name=run.scenario, enabled=debug):
            run.result = await EvalSession.from_scenario(
                scenario,
                url,
                default_timeout_ms=default_timeout_ms,
                on_progress=on_progress,
                record_path=record_path,
                cache_dir=cache_dir,
                use_cache=use_cache,
                stop_bot=stop_bot,
                trigger_disconnect=trigger_disconnect,
            ).run()
        if run.result.debug_log:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            (Path(logs_dir) / f"{run.scenario}.eval.log").write_text(
                "\n".join(run.result.debug_log) + "\n"
            )
    except Exception as e:  # noqa: BLE001
        # Errors raised inside EvalSession.run() are caught there and returned as
        # a structured result; this catches the rest (scenario load, building the
        # judge/speech/transcriber). Keep the exception type and stash the full
        # traceback in <scenario>.eval.log, mirroring the suite's behavior.
        run.error = f"error: {type(e).__name__}: {e}"
        with contextlib.suppress(OSError):
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            (Path(logs_dir) / f"{run.scenario}.eval.log").write_text(traceback.format_exc())
    finally:
        if run.started_at is not None:
            run.duration_ms = int((time.monotonic() - run.started_at) * 1000)
        run.status = "done"


async def _run_scenarios_all(
    runs: list[EvalRun],
    *,
    audio: bool,
    record_dir: str,
    cache_dir: str | None,
    use_cache: bool,
    default_timeout_ms: int,
    logs_dir: str,
    verbose: bool,
    debug: bool,
    stop_bot: bool,
    trigger_disconnect: bool,
    started: float,
) -> None:
    """Run scenarios sequentially against a fixed bot, with the suite's display.

    A live dashboard in an interactive terminal; ``--verbose`` (per-turn lines) or
    a piped stdout fall back to streamed result lines instead.
    """

    async def go(run: EvalRun, on_progress) -> None:
        await _execute_scenario(
            run,
            audio=audio,
            record_dir=record_dir,
            cache_dir=cache_dir,
            use_cache=use_cache,
            default_timeout_ms=default_timeout_ms,
            logs_dir=logs_dir,
            debug=debug,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            on_progress=on_progress,
        )

    if _console.is_terminal and not verbose:
        with Live(_EvalDashboard(runs, started), console=_console, refresh_per_second=12.5):
            for run in runs:
                if run.status != "done":  # skip a build-time load error
                    await go(run, None)
    else:
        for run in runs:
            if run.status != "done":
                await go(run, _print_progress if verbose else None)
            _print_eval_line(run)


@eval_app.command("run")
def run(
    scenarios: list[Path] = typer.Argument(..., help="One or more scenario YAML files."),
    bot_url: str = typer.Option(
        "ws://localhost:7860",
        "--bot-url",
        help="WebSocket URL of the bot's eval transport.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print a line for each turn and expectation as it resolves.",
    ),
    audio: bool = typer.Option(
        False,
        "-a",
        "--audio",
        help="Record each scenario's conversation audio (audio-mode scenarios).",
    ),
    record_dir: str = typer.Option(
        "recordings",
        "--record-dir",
        help="Directory for --audio recordings: <record-dir>/<scenario>.wav.",
    ),
    cache_dir: str = typer.Option(
        None,
        "--cache-dir",
        help="Directory for cached synthesized user audio (default <user-cache-dir>/pipecat/tts).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable the user-audio cache: re-synthesize every turn (no reads or writes).",
    ),
    timeout: int = typer.Option(
        60,
        "-t",
        "--timeout",
        help="Default per-expectation timeout in seconds (for expectations without their own "
        "within_ms).",
    ),
    logs_dir: str = typer.Option(
        ".",
        "--logs-dir",
        help="Directory for each scenario's logs: <logs-dir>/<scenario>.eval.log (+ .debug.log).",
    ),
    debug: bool = typer.Option(
        False,
        "-d",
        "--debug",
        help="Also save <scenario>.debug.log with the harness's full per-pipeline logs.",
    ),
    stop_bot: bool = typer.Option(
        False,
        "--stop-bot",
        help="Cancel the bot's pipeline (exit it) after the run. By default the "
        "bot is left running so it can serve more scenarios.",
    ),
    trigger_disconnect: bool = typer.Option(
        False,
        "--trigger-disconnect",
        help="Fire the bot's on_client_disconnected handler when the eval client "
        "disconnects. Bots often cancel their pipeline there, so it's off by "
        "default. A scenario's 'trigger_disconnect:' field opts in on its own.",
    ),
) -> None:
    """Run one or more evals against an already-running bot.

    A list of scenarios is treated as a one-bot suite: configs are printed up
    front, then a live dashboard (or streamed lines when piped / ``--verbose``)
    shows each scenario's status and timing, the running tally, and the total
    time, sharing the display with ``pipecat eval suite``.
    """
    # pipecat's own logs are captured to <scenario>.debug.log under --debug; either
    # way, silence the console sink so it can't corrupt the live display.
    logger.remove()

    runs = _build_scenario_runs(scenarios, bot_url)
    _print_scenario_configs(runs)

    started = time.monotonic()
    asyncio.run(
        _run_scenarios_all(
            runs,
            audio=audio,
            record_dir=record_dir,
            cache_dir=cache_dir,
            use_cache=not no_cache,
            default_timeout_ms=timeout * 1000,
            logs_dir=logs_dir,
            verbose=verbose,
            debug=debug,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            started=started,
        )
    )
    dashboard_shown = _console.is_terminal and not verbose
    exit_code = _finalize_evals(
        runs, Path(logs_dir).resolve(), time.monotonic() - started, dashboard_shown
    )
    raise typer.Exit(code=exit_code)


#
# `pipecat eval suite` — spawn the bots in a manifest and run their scenarios.
#

_EVAL_GLYPH = {
    "passed": ("✓", "green", "32"),
    "failed": ("✗", "red", "31"),
    "skipped": ("⊘", "yellow", "33"),
    "error": ("✗", "red", "31"),
}


def _eval_verdict(r: EvalRun) -> str:
    """Collapse a run into a display verdict."""
    if r.status != "done":
        return r.status  # pending | running
    if r.error or r.result is None:
        return "error"
    if r.result.skipped:
        return "skipped"
    return "passed" if r.result.passed else "failed"


def _fmt_duration(seconds: float) -> str:
    """Human-friendly elapsed time, e.g. ``12.3s`` or ``2m 04s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def _eval_status_cell(r: EvalRun):
    """A rich renderable for the status column (spinner while running)."""
    if r.status == "pending":
        return Text("·", style="dim")
    if r.status == "running":
        return Spinner("dots", style="cyan")
    glyph, style, _ = _EVAL_GLYPH[_eval_verdict(r)]
    return Text(glyph, style=style)


class _EvalDashboard:
    """Live table rendered every frame from the shared list of runs."""

    def __init__(self, runs: list[EvalRun], started_at: float):
        self.runs = runs
        self.started_at = started_at

    def __rich__(self) -> Group:
        rows = self.runs
        n = len(rows)

        # Show only as many scenario rows as fit above the pinned tally, and slide
        # the window to follow the active runs. Otherwise, once the first screenful
        # finishes, Rich crops the bottom — hiding the still running/pending rows and
        # the tally itself — and there's no way to see the rest make progress.
        term_h = _console.size.height if _console.is_terminal else 24
        avail = max(3, term_h - 3)  # leave room for a blank line + the tally (+1 slack)
        if n <= avail:
            start, end = 0, n
        else:
            # Anchor on the active frontier (the last running run, or the first
            # pending one) and keep a couple of upcoming rows in view below it, so
            # completed runs scroll off the top as new ones start.
            running = [i for i, r in enumerate(rows) if r.status == "running"]
            done_n = sum(1 for r in rows if r.status == "done")
            anchor = max(running) if running else min(done_n, n - 1)
            body = max(1, avail - 2)  # 2 lines reserved for the ↑/↓ "more" markers
            end = min(n, anchor + 1 + min(2, body - 1))
            start = max(0, end - body)
            end = min(n, start + body)

        table = Table.grid(padding=(0, 2))
        table.add_column()  # status
        table.add_column()  # bot
        table.add_column()  # scenario
        table.add_column(justify="right")  # timing
        for r in rows[start:end]:
            if r.status == "running" and r.started_at is not None:
                detail = f"{int(time.monotonic() - r.started_at)}s"
            elif r.status == "done" and r.duration_ms is not None:
                detail = f"{r.duration_ms}ms"
            else:
                detail = ""
            table.add_row(
                _eval_status_cell(r),
                Text(r.bot),
                Text(r.scenario, style="cyan"),
                Text(detail, style="dim"),
            )

        total = n
        done = sum(1 for r in rows if r.status == "done")
        passed = sum(1 for r in rows if _eval_verdict(r) == "passed")
        # The total time ticks live next to the tally, so it doubles as the
        # "still working" signal (no spinner needed); it keeps advancing through
        # the bot-teardown tail (see _stop_bot) until Live exits, leaving the
        # final time on screen. The first column is kept blank so the tally stays
        # aligned with the rows above. _finalize_evals intentionally does not
        # reprint this line (it would duplicate the last frame).
        elapsed = _fmt_duration(time.monotonic() - self.started_at)
        summary = Table.grid(padding=(0, 1))
        summary.add_column()  # blank, for alignment with the status column
        summary.add_column()  # tally
        summary.add_row(
            Text(" "),
            Text(f"{passed}/{total} passed  ·  {done}/{total} done  ·  {elapsed}", "bold"),
        )

        # Replace Rich's bottom-crop "…" with scroll markers that count what's
        # hidden above/below the window, so the tally below always stays on screen.
        parts: list = []
        if start > 0:
            parts.append(Text(f"   ↑ {start} more", style="dim"))
        parts.append(table)
        if end < n:
            parts.append(Text(f"   ↓ {n - end} more", style="dim"))
        return Group(*parts, Text(""), summary)


def _print_eval_line(r: EvalRun) -> None:
    """Print a single result line (non-TTY fallback, streamed as each finishes)."""
    if r.status != "done":
        return
    glyph, _, code = _EVAL_GLYPH[_eval_verdict(r)]
    extra = f"({r.duration_ms}ms)" if r.duration_ms is not None and not r.error else (r.error or "")
    print(f"  {_color(glyph, code)} {r.bot} {_color(r.scenario, '36')} {_dim(extra)}", flush=True)


def _print_scenario_configs(runs: list[EvalRun]) -> None:
    """Print each distinct scenario's config once, up front (shared by run + suite).

    Done before the runs (not per-run) so it doesn't interleave with the live
    display, with a trailing blank line separating it from the runs.
    """
    print("Scenarios:")
    seen: set[str] = set()
    for r in runs:
        if r.scenario in seen:
            continue
        seen.add(r.scenario)
        try:
            cfg = describe_config(EvalScenario.load(r.scenario_path), color=sys.stdout.isatty())
        except Exception as e:  # noqa: BLE001
            cfg = f"(failed to load: {e})"
        print(f"  {_color(r.scenario + ':', '1;36')}")
        for line in cfg.splitlines():
            print(f"    {line}")
    print()


def _finalize_evals(
    runs: list[EvalRun], runs_dir: Path, elapsed_s: float, dashboard_shown: bool
) -> int:
    """Print the failed set + final tally; return the process exit code."""
    failed = [r for r in runs if _eval_verdict(r) in ("failed", "error")]
    passed = sum(1 for r in runs if _eval_verdict(r) == "passed")
    skipped = sum(1 for r in runs if _eval_verdict(r) == "skipped")

    if failed:
        print()
        print(f"  {_color(f'Failed ({len(failed)}):', '1;31')}")
        for r in failed:
            if r.error:
                print(f"  {_red('✗')} {r.bot} {_color(r.scenario, '36')} {_dim('— ' + r.error)}")
            elif r.result is not None:
                print(f"  {_red('✗')} {r.bot} {_color(r.scenario, '36')}")
                for f in r.result.failures:
                    print(f"      {_red('•')} {f}")

    print()
    # When the live dashboard ran, its last frame already shows the tally and the
    # (now final) elapsed time, so reprinting it here would just duplicate that
    # line. Without a dashboard (piped, or run --verbose) print the tally here.
    if not dashboard_shown:
        summary = f"{passed}/{len(runs)} passed"
        if failed:
            summary += f", {len(failed)} failed"
        if skipped:
            summary += f", {skipped} skipped"
        summary += f"  ·  {_fmt_duration(elapsed_s)}"
        print(f"  {_color(summary, '31' if failed else '32')}")
    print(f"  logs: {runs_dir}")
    print()
    return 0 if not failed else 1


async def _run_suite_all(
    suite: EvalSuite,
    logs_dir: Path,
    record_dir: Path | None,
    started: float,
    debug: bool,
    use_cache: bool,
    default_timeout_ms: int,
) -> None:
    """Run the suite with a live dashboard (TTY) or streamed lines (piped)."""
    if _console.is_terminal:
        with Live(_EvalDashboard(suite.runs, started), console=_console, refresh_per_second=12.5):
            await suite.run(
                logs_dir,
                record_dir=record_dir,
                debug=debug,
                use_cache=use_cache,
                default_timeout_ms=default_timeout_ms,
            )
    else:
        print(f"Running {len(suite.runs)} eval(s), {suite.manifest.concurrency} at a time...")
        await suite.run(
            logs_dir,
            record_dir=record_dir,
            on_update=_print_eval_line,
            debug=debug,
            use_cache=use_cache,
            default_timeout_ms=default_timeout_ms,
        )


@eval_app.command("suite")
def suite(
    manifest_path: Path = typer.Argument(..., help="Manifest YAML listing bots + scenarios."),
    pattern: str = typer.Option(
        None, "-p", "--pattern", help="Only bots whose path contains this."
    ),
    scenario: str = typer.Option(None, "-s", "--scenario", help="Only this scenario name."),
    name: str = typer.Option(
        None, "-n", "--name", help="Run subdir name under runs_dir (default a timestamp)."
    ),
    runs_dir: Path = typer.Option(
        None,
        "--runs-dir",
        help="Output base, overriding the manifest's runs_dir (a <name>/ subdir with "
        "logs/ and recordings/ is created under it; default eval-runs).",
    ),
    bots_dir: Path = typer.Option(None, "--bots-dir", help="Override manifest bots_dir."),
    scenarios_dir: Path = typer.Option(
        None, "--scenarios-dir", help="Override manifest scenarios_dir."
    ),
    concurrency: int = typer.Option(
        None, "-c", "--concurrency", help="Override manifest concurrency."
    ),
    base_port: int = typer.Option(None, "--base-port", help="Override manifest base_port."),
    cache_dir: str = typer.Option(None, "--cache-dir", help="Override manifest cache_dir."),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable the user-audio cache: re-synthesize every turn (no reads or writes).",
    ),
    timeout: int = typer.Option(
        60,
        "-t",
        "--timeout",
        help="Default per-expectation timeout in seconds (for expectations without their own "
        "within_ms).",
    ),
    spawn: str = typer.Option(None, "--spawn", help="Override manifest spawn template."),
    python: str = typer.Option(None, "--python", help="Override manifest python interpreter."),
    audio: bool = typer.Option(False, "-a", "--audio", help="Record conversation audio."),
    debug: bool = typer.Option(
        False,
        "-d",
        "--debug",
        help="Also save <run>.debug.log with the harness's full per-pipeline logs.",
    ),
) -> None:
    """Spawn the bots in a manifest and run their scenarios concurrently.

    Everything except the ``suite:`` list can be set in the manifest or overridden
    here (the command line wins), so a manifest can be just a ``suite:`` list.
    """
    manifest = EvalManifest.load(
        manifest_path,
        bots_dir=bots_dir,
        scenarios_dir=scenarios_dir,
        runs_dir=runs_dir,
        spawn=spawn,
        python=python,
        concurrency=concurrency,
        base_port=base_port,
        record=audio or None,
        cache_dir=cache_dir,
    )

    suite = EvalSuite(manifest)
    runs = suite.filter(pattern=pattern, scenario=scenario)
    if not runs:
        print("No runs match.")
        raise typer.Exit(code=1)

    # A per-run subdir named by --name (default a timestamp) holds this run's logs
    # and recordings, under the (resolved) runs_dir.
    base = manifest.runs_dir or Path("eval-runs")
    run_dir = base / (name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    logs_dir = run_dir / "logs"
    record_dir = (run_dir / "recordings") if manifest.record else None

    _print_scenario_configs(runs)

    started = time.monotonic()
    asyncio.run(
        _run_suite_all(suite, logs_dir, record_dir, started, debug, not no_cache, timeout * 1000)
    )
    exit_code = _finalize_evals(
        runs, run_dir, time.monotonic() - started, dashboard_shown=_console.is_terminal
    )
    raise typer.Exit(code=exit_code)
