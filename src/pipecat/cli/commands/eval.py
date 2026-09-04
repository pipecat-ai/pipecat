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

from pipecat.evals.results import EvalScriptResult, EvalScriptTurnProgress, EvalSimulationResult
from pipecat.evals.scenario import describe_config
from pipecat.evals.script_session import EvalScriptSession
from pipecat.evals.simulation import EvalSimulationScenario, describe_simulation, load_scenario_file
from pipecat.evals.simulation_session import EvalSimulationSession
from pipecat.evals.suite import (
    SCENARIO_SUFFIXES,
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


def _format_detail(p: EvalScriptTurnProgress) -> str:
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


def _print_progress(session: EvalScriptSession, p: EvalScriptTurnProgress) -> None:
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


def _print_simulation_detail(result: EvalSimulationResult) -> None:
    """Print what a simulation's one-line verdict leaves out (verbose mode).

    The judge's reason, each metric's score and reason, the persona's own claim
    from its ``end_call``, how many turns it took, and the conversation.
    """
    if result.error:
        return
    print(f"    {_dim('judge:')} {result.reason}")
    for metric in result.metrics:
        print(f"    {_dim(metric.name + ':')} {metric.score:g}  {_dim(metric.reason)}")
    if result.end_call is not None:
        claim = "succeeded" if result.end_call.get("success") else "gave up"
        print(f"    {_dim('persona:')} {claim}: {result.end_call.get('reason', '')}")
    print(f"    {_dim('turns:')} {result.turns} persona turn(s), ended by {result.ended_by}")
    if result.messages:
        print("    Conversation:")
        for message in result.messages:
            who = "user" if message["role"] == "user" else "bot"
            print(f"      {_dim(who + ':')} {message['content']}")


def _record_path(record_dir: str | None, scenario_name: str) -> str | None:
    """Per-scenario recording path under ``record_dir``, or None when recording is off."""
    if not record_dir:
        return None
    return str(Path(record_dir) / f"{scenario_name}.wav")


def _expand_scenario_paths(paths: list[Path]) -> list[Path]:
    """Expand directory arguments into sorted YAML scenario paths.

    Both YAML suffixes are taken, matching the scenario names a manifest
    resolves.
    """
    expanded: list[Path] = []
    for path in paths:
        if not path.is_dir():
            expanded.append(path)
            continue

        scenario_paths = sorted(
            scenario_path
            for scenario_path in path.iterdir()
            if scenario_path.suffix in SCENARIO_SUFFIXES and scenario_path.is_file()
        )
        if not scenario_paths:
            raise typer.BadParameter(f"No .yaml or .yml scenario files found in {path}")
        expanded.extend(scenario_paths)
    return expanded


def _build_scenario_runs(paths: list[Path], bot_url: str) -> list[EvalRun]:
    """Build an EvalRun per scenario file, scripted or a simulation, run against ``bot_url``.

    A file that fails to load becomes an EvalRun already marked done with an
    error, so it shows in the dashboard and the final tally like any other failure.
    """
    runs: list[EvalRun] = []
    for path in paths:
        try:
            loaded = load_scenario_file(path)
        except (ValueError, FileNotFoundError) as e:
            run = EvalRun(bot=bot_url, scenario=path.stem, scenario_path=path, bot_url=bot_url)
            run.status = "done"
            run.error = f"failed to load: {e}"
            runs.append(run)
            continue
        kind = "simulation" if isinstance(loaded, EvalSimulationScenario) else "script"
        runs.append(
            EvalRun(
                bot=bot_url, scenario=loaded.name, scenario_path=path, bot_url=bot_url, kind=kind
            )
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
    verbose: bool,
) -> None:
    """Run one scenario file, scripted or a simulation, against its ``bot_url``.

    Updates ``run`` in place.

    The ``eval run`` counterpart to the suite's _run_one: it connects to a fixed
    URL instead of spawning, always writes the decision trace (``<scenario>.eval.log``)
    and, under ``--debug``, the combined ``<scenario>.debug.log``.
    """
    run.status = "running"
    run.started_at = time.monotonic()
    url = run.bot_url
    assert url is not None  # always set by _build_scenario_runs
    try:
        loaded = load_scenario_file(run.scenario_path)
        record_path = _record_path(record_dir, run.scenario) if audio else None
        with capture_pipeline_logs(Path(logs_dir), run.scenario, name=run.scenario, enabled=debug):
            session: EvalScriptSession | EvalSimulationSession
            if isinstance(loaded, EvalSimulationScenario):
                session = EvalSimulationSession.from_simulation(
                    loaded,
                    url,
                    record_path=record_path,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    stop_bot=stop_bot,
                    trigger_disconnect=trigger_disconnect,
                )
            else:
                session = EvalScriptSession.from_scenario(
                    loaded,
                    url,
                    default_timeout_ms=default_timeout_ms,
                    record_path=record_path,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                    stop_bot=stop_bot,
                    trigger_disconnect=trigger_disconnect,
                )
                if verbose:
                    session.add_event_handler("on_progress", _print_progress)
            run.result = await session.run()
        if run.result.debug_log:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            (Path(logs_dir) / f"{run.scenario}.eval.log").write_text(
                "\n".join(run.result.debug_log) + "\n"
            )
    except Exception as e:  # noqa: BLE001
        # Errors raised inside the session's run() are caught there and returned
        # as a structured result; this catches the rest (the file's load, building
        # the judge/persona/speech/transcriber). Keep the exception type and stash the full
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

    A live dashboard in an interactive terminal; ``--verbose`` (per-turn lines, and
    a simulation's verdict detail and conversation) or a piped stdout fall back to
    streamed result lines instead.
    """

    async def go(run: EvalRun, verbose: bool) -> None:
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
            verbose=verbose,
        )

    if _console.is_terminal and not verbose:
        with Live(_EvalDashboard(runs, started), console=_console, refresh_per_second=12.5):
            for run in runs:
                if run.status != "done":  # skip a build-time load error
                    await go(run, False)
    else:
        for run in runs:
            if run.status != "done":
                await go(run, verbose)
            _print_eval_line(run)
            if verbose and isinstance(run.result, EvalSimulationResult):
                _print_simulation_detail(run.result)


@eval_app.command("run")
def run(
    scenarios: list[Path] = typer.Argument(
        ...,
        help="One or more scenario YAML files (scripted, or simulations), or directories of them.",
    ),
    bot_url: str = typer.Option(
        "ws://localhost:7860",
        "--bot-url",
        help="WebSocket URL of the bot's eval transport.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print a line for each turn and expectation as it resolves; for a "
        "simulation, the judge's reasons and the conversation once it ends.",
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
    """Run one or more scenarios, scripted or simulations, against an already-running bot.

    A list of scenarios is treated as a one-bot suite: configs are printed up
    front, then a live dashboard (or streamed lines when piped / ``--verbose``)
    shows each scenario's status and timing, the running tally, and the total
    time, sharing the display with ``pipecat eval suite``. A simulation runs once
    here: a success rate over several runs is the suite's job, since each run
    needs a fresh bot.
    """
    # pipecat's own logs are captured to <scenario>.debug.log under --debug; either
    # way, silence the console sink so it can't corrupt the live display.
    logger.remove()

    runs = _build_scenario_runs(_expand_scenario_paths(scenarios), bot_url)
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
    if isinstance(r.result, EvalScriptResult) and r.result.skipped:
        return "skipped"
    if isinstance(r.result, EvalSimulationResult) and r.result.error:
        return "error"
    return "passed" if r.result.passed else "failed"


def _turn_tally(r: EvalRun) -> str:
    """The run's detail beside its verdict, or ``""``.

    A scenario: ``7/10 turns`` for a run that drove every turn and failed some.
    A rate needs every turn scored. A run that stopped at its first failure left
    the rest undriven, and a fraction over those would read as turns that failed
    when they were never attempted — what it stopped on is in the failure listing
    instead. A run that passed is already said by the ✓.

    A simulation: its quality and how it ended, e.g. ``quality 0.75 · end_call``.
    """
    result = r.result
    if isinstance(result, EvalSimulationResult):
        parts = []
        if result.quality is not None:
            parts.append(f"quality {result.quality:.2f}")
        parts.append(result.ended_by)
        return " · ".join(parts)
    if result is None or not result.turns:
        return ""
    if any(t.status == "not_run" for t in result.turns):
        return ""
    failed = sum(1 for t in result.turns if t.status == "failed")
    if not failed:
        return ""
    return f"{len(result.turns) - failed}/{len(result.turns)} turns"


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


def _pass_rate(passed: int, done: int) -> str:
    """``23/50 (46%)`` over the attempts that finished, or a placeholder if none have."""
    if not done:
        return "—"
    return f"{passed}/{done} ({100 * passed // done}%)"


# Rate colors, as a rich style and the matching ANSI code for the non-TTY path.
_RATE_ANSI = {"green": "32", "yellow": "33", "red": "31", "dim": "2"}


def _rate_level(passed: int, done: int) -> str:
    """Color for a pass rate: green when perfect, red below half, yellow between."""
    if not done:
        return "dim"
    if passed == done:
        return "green"
    return "red" if (100 * passed // done) < 50 else "yellow"


def _mean_duration(runs: list[EvalRun]) -> str:
    """Mean wall-clock of the finished runs, e.g. ``~53.2s each``."""
    times = [r.duration_ms for r in runs if r.duration_ms is not None]
    if not times:
        return ""
    return f"~{_fmt_duration(sum(times) / len(times) / 1000)} each"


def _group_key(r: EvalRun) -> tuple[str, str]:
    return (r.bot, r.scenario)


def _grouped_runs(runs: list[EvalRun]) -> dict[tuple[str, str], list[EvalRun]]:
    """Runs bucketed by (bot, scenario), in first-seen order."""
    groups: dict[tuple[str, str], list[EvalRun]] = {}
    for r in runs:
        groups.setdefault(_group_key(r), []).append(r)
    return groups


class _EvalDashboard:
    """Live table rendered every frame from the shared list of runs.

    With ``grouped``, attempts of the same (bot, scenario) collapse into a single
    row carrying a pass counter — a repeated sweep is hundreds of runs, which as
    one row apiece would scroll the table faster than it could be read.
    """

    def __init__(self, runs: list[EvalRun], started_at: float, *, grouped: bool = False):
        self.runs = runs
        self.started_at = started_at
        self.grouped = grouped

    def __rich__(self) -> Group:
        if self.grouped:
            return self._render_grouped()
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

    def _render_grouped(self) -> Group:
        """One row per (bot, scenario), showing that pair's pass rate and pace.

        A simulation's row also carries its mean quality, and its final ✓ or ✗ is
        measured against its ``pass_threshold`` rather than every attempt passing.
        """
        table = Table.grid(padding=(0, 2))
        table.add_column()  # status
        table.add_column()  # bot
        table.add_column()  # scenario
        table.add_column(justify="right")  # pass rate
        table.add_column(justify="right")  # mean quality (simulations)
        table.add_column(justify="right")  # remaining
        table.add_column(justify="right")  # mean run time

        for (bot, scenario), group in _grouped_runs(self.runs).items():
            done = [r for r in group if r.status == "done"]
            passed, _, _, quality = _group_outcome(group)
            # Three states, and only the last is a verdict: spinning while a slot is
            # held, a still dot while waiting for one, and ✓/✗ once every attempt is
            # in. Motion is reserved for the runs actually in flight — most rows of a
            # long sweep are waiting, and animating those too would leave nothing for
            # movement to mean.
            if len(done) == len(group):
                if group[0].pass_threshold is not None:
                    ok = not _group_below_threshold(group)
                else:
                    ok = passed == len(group)
                glyph, style, _ = _EVAL_GLYPH["passed" if ok else "failed"]
                status = Text(glyph, style=style)
            elif any(r.status == "running" for r in group):
                status = Spinner("dots", style="cyan")
            else:
                status = Text("·", style="dim")
            # The rate is over attempts that finished, so it reads as a real rate
            # while the sweep is still going; what's left is its own column rather
            # than a second denominator competing with it.
            remaining = len(group) - len(done)
            table.add_row(
                status,
                Text(bot),
                Text(scenario, style="cyan"),
                Text(_pass_rate(passed, len(done)), style=_rate_level(passed, len(done))),
                Text(f"quality {quality:.2f}" if quality is not None else "", style="dim"),
                Text(f"{remaining} left" if remaining else "", style="dim"),
                Text(_mean_duration(done), style="dim"),
            )

        total = len(self.runs)
        done = sum(1 for r in self.runs if r.status == "done")
        passed = sum(1 for r in self.runs if _eval_verdict(r) == "passed")
        elapsed = _fmt_duration(time.monotonic() - self.started_at)
        summary = Table.grid(padding=(0, 1))
        summary.add_column()
        summary.add_column()
        summary.add_row(
            Text(" "),
            Text(f"{passed}/{total} passed  ·  {done}/{total} done  ·  {elapsed}", "bold"),
        )
        return Group(table, Text(""), summary)


def _print_eval_line(r: EvalRun, *, show_attempt: bool = False) -> None:
    """Print a single result line (non-TTY fallback, streamed as each finishes)."""
    if r.status != "done":
        return
    glyph, _, code = _EVAL_GLYPH[_eval_verdict(r)]
    extra = f"({r.duration_ms}ms)" if r.duration_ms is not None and not r.error else (r.error or "")
    tally = _turn_tally(r)
    if tally:
        extra = f"{tally} {extra}"
    scenario = f"{r.scenario} #{r.attempt}" if show_attempt else r.scenario
    print(f"  {_color(glyph, code)} {r.bot} {_color(scenario, '36')} {_dim(extra)}", flush=True)


def _plural(count: int, noun: str) -> str:
    """``1 eval`` / ``2 evals`` — count plus its noun, pluralized with a trailing -s."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _print_run_settings(runs: list[EvalRun], concurrency: int, record: bool) -> None:
    """Print how the suite will execute, in the shape of the configs above it.

    Only what the terminal doesn't otherwise show while the run is going, and that
    a manifest can turn on without it appearing on the command line: the run count
    (arithmetic, and the difference between 8 runs and several thousand), the
    concurrency, and whether audio is being recorded — a few MB per run, so it
    grows with the sweep. What the caller just typed stays out; it's already in
    their scrollback.
    """
    total = len(runs)
    counts = str(total)
    attempts = sorted({r.attempts for r in runs})
    if attempts and attempts[-1] > 1:
        evals = len(_grouped_runs(runs))
        if len(attempts) == 1:
            counts = f"{total} ({_plural(evals, 'eval')} x {_plural(attempts[0], 'attempt')})"
        else:
            counts = (
                f"{total} ({_plural(evals, 'eval')}, {attempts[0]}-{attempts[-1]} attempts each)"
            )
    print("Settings:")
    for label, value in (
        ("runs", counts),
        ("concurrency", str(concurrency)),
        ("recording", "on" if record else "off"),
    ):
        print(f"  {_color(f'{label:11s}', '1')} -> {value}")
    print()


def _print_scenario_configs(runs: list[EvalRun]) -> None:
    """Print each distinct scenario's and simulation's config once, up front.

    Done before the runs (not per-run) so it doesn't interleave with the live
    display, with a trailing blank line separating it from the runs.
    """
    for kind, heading in (
        ("script", "Scripted scenarios:"),
        ("simulation", "Simulated scenarios:"),
    ):
        seen: set[str] = set()
        for r in runs:
            if r.kind != kind or r.scenario in seen:
                continue
            if not seen:
                print(heading)
            seen.add(r.scenario)
            try:
                loaded = load_scenario_file(r.scenario_path)
                if isinstance(loaded, EvalSimulationScenario):
                    cfg = describe_simulation(loaded, color=sys.stdout.isatty())
                else:
                    cfg = describe_config(loaded, color=sys.stdout.isatty())
            except Exception as e:  # noqa: BLE001
                cfg = f"(failed to load: {e})"
            print(f"  {_color(r.scenario + ':', '1;36')}")
            for line in cfg.splitlines():
                print(f"    {line}")
        if seen:
            print()


def _print_failures(failed: list[EvalRun], total: int, *, show_attempt: bool) -> None:
    """Print every failed run: what it was, and what went wrong.

    Every failure is listed, however many there are. Reading them is how a person
    tells a bot that misbehaved from one that never started, and only the failure's
    own ``reason`` carries that — ``kind`` says which assertion gave way, not what
    the bot did. Counting and grouping belong to the ``results.jsonl`` written
    alongside, which is the structured record of the same failures. A simulation
    that failed carries the judge's verdict instead of assertions.

    Args:
        failed: The runs that failed or errored.
        total: How many runs there were, for the header's denominator.
        show_attempt: Include each run's attempt number — worth the noise only
            when a sweep repeats, since that is what says which log to open.
    """
    if not failed:
        return
    print()
    print(f"  {_color(f'Failures ({len(failed)} of {total}):', '1;31')}")
    for r in failed:
        attempt = f" {_dim('#' + str(r.attempt))}" if show_attempt else ""
        tally = _turn_tally(r)
        header = f"  {_red('✗')} {r.bot} {_color(r.scenario, '36')}{attempt}"
        if tally and isinstance(r.result, EvalScriptResult):
            header = f"{header} {_dim(tally + ' passed')}"
        if r.error:
            print(f"{header} {_dim('— ' + r.error)}")
        elif isinstance(r.result, EvalSimulationResult):
            result = r.result
            if result.error:
                print(f"{header} {_dim('— ' + result.error)}")
            else:
                print(header)
                print(f"      {_red('•')} goal not achieved {_dim(tally)} — {result.reason}")
        elif r.result is not None:
            print(header)
            for f in r.result.failures:
                turn = f"turn {f.turn_index}" if f.turn_index >= 0 else "run"
                print(
                    f"      {_red('•')} {turn} {f.event_name} {_color(f.kind, '31')} — {f.reason}"
                )


def _group_outcome(group: list[EvalRun]) -> tuple[int, int, int, float | None]:
    """A (bot, scenario) group's ``(passed, completed, errored, mean quality)``.

    Errored runs are left out of the completed count: a run that crashed or never
    connected says nothing about whether the bot did its job, and folding it into
    the rate would both drag the rate down unfairly and hide an infrastructure
    problem behind a behavioral one.
    """
    verdicts = [_eval_verdict(r) for r in group]
    passed = sum(1 for v in verdicts if v == "passed")
    errored = sum(1 for v in verdicts if v == "error")
    completed = sum(1 for v in verdicts if v in ("passed", "failed", "skipped"))
    qualities = [
        r.result.quality
        for r in group
        if isinstance(r.result, EvalSimulationResult) and r.result.quality is not None
    ]
    quality = sum(qualities) / len(qualities) if qualities else None
    return passed, completed, errored, quality


def _group_below_threshold(group: list[EvalRun]) -> bool:
    """Whether a simulation group's success rate misses its ``pass_threshold``."""
    threshold = group[0].pass_threshold
    if threshold is None:
        return False
    passed, completed, _, _ = _group_outcome(group)
    # Compared at the precision the rate is printed at, so two of three meets a
    # threshold written as 0.67 rather than missing it by a rounding error.
    return completed == 0 or round(passed / completed, 2) < threshold


def _print_repeat_summary(runs: list[EvalRun], failed: list[EvalRun], *, show_rates: bool) -> None:
    """Print per-(bot, scenario) pass rates and every failure.

    A repeated sweep is measuring a rate, not a verdict, so the useful output is
    how often each pair passed and which runs account for the rest. A simulation's
    rate is measured against its ``pass_threshold``, marked ✓ or ✗ beside it, with
    its mean quality and how many runs errored (those are outside the rate).
    ``show_rates`` is False when the live dashboard ran: its final frame is already
    a per-(bot, scenario) rate table, so repeating it here would just duplicate it.
    """
    if show_rates:
        print()
        print(f"  {_color('Pass rate:', '1')}")
        groups = _grouped_runs(runs)
        bot_w = max(len(bot) for bot, _ in groups)
        scenario_w = max(len(scenario) for _, scenario in groups)
        for (bot, scenario), group in groups.items():
            passed, completed, errored, quality = _group_outcome(group)
            rate_text = f"{_pass_rate(passed, completed):>14s}"
            rate = _color(rate_text, _RATE_ANSI[_rate_level(passed, completed)])
            extra = []
            if errored:
                extra.append(f"{errored} errored")
            if quality is not None:
                extra.append(f"quality {quality:.2f}")
            extra.append(_mean_duration(group))
            mark = ""
            threshold = group[0].pass_threshold
            if threshold is not None:
                below = _group_below_threshold(group)
                mark = f"  {_red('✗') if below else _green('✓')}"
                if below:
                    extra.append(f"below {int(threshold * 100)}%")
            print(
                f"  {bot:{bot_w}s}  {_color(f'{scenario:{scenario_w}s}', '36')}  {rate}{mark}  "
                f"{_dim(' · '.join(e for e in extra if e))}"
            )
    _print_failures(failed, len(runs), show_attempt=True)


def _finalize_evals(
    runs: list[EvalRun],
    runs_dir: Path,
    elapsed_s: float,
    dashboard_shown: bool,
) -> int:
    """Print the failed set + final tally; return the process exit code."""
    failed = [r for r in runs if _eval_verdict(r) in ("failed", "error")]
    passed = sum(1 for r in runs if _eval_verdict(r) == "passed")
    skipped = sum(1 for r in runs if _eval_verdict(r) == "skipped")
    if any(r.attempts > 1 for r in runs):
        _print_repeat_summary(runs, failed, show_rates=not dashboard_shown)
        print()
        if not dashboard_shown:
            summary = f"{passed}/{len(runs)} passed  ·  {_fmt_duration(elapsed_s)}"
            print(f"  {_color(summary, '31' if failed else '32')}")
        print(f"  logs: {runs_dir}")
        print(f"  results: {runs_dir / 'results.jsonl'}")
        print()
        # A repeated sweep measures a pass rate; what counts as acceptable is the
        # caller's policy, so failures here are data rather than a build break.
        # A simulation carries its policy as its pass_threshold, and a rate below
        # it is a break.
        below = [g for g in _grouped_runs(runs).values() if _group_below_threshold(g)]
        return 1 if below else 0
    _print_failures(failed, len(runs), show_attempt=False)
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
    results_path: Path | None,
    started: float,
    debug: bool,
    use_cache: bool,
    default_timeout_ms: int,
) -> None:
    """Run the suite with a live dashboard (TTY) or streamed lines (piped)."""
    grouped = any(r.attempts > 1 for r in suite.runs)
    if _console.is_terminal:
        dashboard = _EvalDashboard(suite.runs, started, grouped=grouped)
        with Live(dashboard, console=_console, refresh_per_second=12.5):
            await suite.run(
                logs_dir,
                record_dir=record_dir,
                results_path=results_path,
                debug=debug,
                use_cache=use_cache,
                default_timeout_ms=default_timeout_ms,
            )
    else:
        suite.add_event_handler(
            "on_update", lambda _suite, run: _print_eval_line(run, show_attempt=grouped)
        )
        await suite.run(
            logs_dir,
            record_dir=record_dir,
            results_path=results_path,
            debug=debug,
            use_cache=use_cache,
            default_timeout_ms=default_timeout_ms,
        )


@eval_app.command("suite")
def suite(
    manifest_path: Path = typer.Argument(
        ..., help="Manifest YAML listing bots + their scenarios (scripted, or simulations)."
    ),
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
    repeat: int = typer.Option(
        None,
        "-r",
        "--repeat",
        help="Run each (bot, scenario) this many times, to measure flakiness, and "
        "each simulation this many times instead of its own runs. Attempts "
        "interleave across bots and each writes its own logs.",
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
    here (the command line wins), so a manifest can be just a ``suite:`` list. A
    scenario file is scripted or a simulation, and the file says which; a
    simulation runs as many times as its file says and passes if its success
    rate meets its threshold.
    """
    manifest = EvalManifest.load(
        manifest_path,
        bots_dir=bots_dir,
        scenarios_dir=scenarios_dir,
        runs_dir=runs_dir,
        spawn=spawn,
        python=python,
        concurrency=concurrency,
        repeat=repeat,
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
    _print_run_settings(runs, manifest.concurrency, manifest.record)

    started = time.monotonic()
    asyncio.run(
        _run_suite_all(
            suite,
            logs_dir,
            record_dir,
            run_dir / "results.jsonl",
            started,
            debug,
            not no_cache,
            timeout * 1000,
        )
    )
    exit_code = _finalize_evals(
        runs, run_dir, time.monotonic() - started, dashboard_shown=_console.is_terminal
    )
    raise typer.Exit(code=exit_code)
