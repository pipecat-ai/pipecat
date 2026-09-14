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
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from pipecat.evals.results import (
    EvalProgress,
    EvalScriptResult,
    EvalScriptTurnProgress,
    EvalSimulationProgress,
    EvalSimulationResult,
)
from pipecat.evals.scenario import (
    EvalKind,
    EvalSimulationScenario,
    describe_config,
    describe_simulation,
    is_scenario_file,
    load_scenario_file,
)
from pipecat.evals.session import EvalSession, EvalSessionParams
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
    """Load the nearest ``.env`` before any eval subcommand runs.

    The harness's own services (a simulation's persona LLM, a hosted judge)
    read their credentials from the environment, so the ``.env`` the bots load
    for themselves is loaded here too: the first one found walking up from the
    working directory. Variables already set in the shell win.

    Also anchors the subcommand structure: typer treats ``run`` and ``suite``
    as explicit subcommands rather than collapsing a single command into a
    flat app.
    """
    load_dotenv(find_dotenv(usecwd=True))


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


def _bold(s: str) -> str:
    return _color(s, "1")


# The speakers in a simulation's conversation: the bot green, the persona cyan.
_SPEAKER_COLOR = {"bot": "32", "user": "36"}
# How a simulation ended: a party hanging up is green (the persona) or yellow
# (the bot); a cap cutting it short is red.
_ENDING_COLOR = {"end_call": "32", "bot": "33"}


def _print_progress(session: EvalSession, p: EvalProgress) -> None:
    """Print a progress record as it arrives (verbose mode).

    A scripted scenario's per-turn and per-expectation lines, or a
    simulation's conversation as it is spoken and how it ended.
    """
    if isinstance(p, EvalSimulationProgress):
        if p.status == "ended":
            ending = _color(p.text, _ENDING_COLOR.get(p.text, "31"))
            print()
            print(f"      {_dim('ended by')} {ending} {_dim(f'after {p.turn} persona turn(s)')}")
        else:
            print(f"      {_color(p.status + ':', _SPEAKER_COLOR[p.status])} {p.text}")
    elif p.status == "turn":
        label = f'"{p.event_name}"' if p.event_name else "(observe)"
        print(f"      {_dim(f'turn {p.turn_index}')} → {label}")
    elif p.status == "timing":
        # The turn's latency, under its expectations: harness-measured time to
        # the first LLM token and, for a spoken turn, voice-to-voice.
        print(f"        {_dim(p.detail)}")
    else:
        badge = _green("✓") if p.status == "matched" else _red("✗")
        line = f"        {badge} {p.event_name}"
        if p.detail:
            detail = _format_detail(p)
            used = 8 + 2 + len(p.event_name) + 3  # indent + badge + name + " — "
            line += f" {_dim(f'— {_fit_detail(detail, used)}')}"
        print(line)


def _print_simulation_detail(result: EvalSimulationResult) -> None:
    """Print a simulation's verdict detail (verbose mode), under the conversation.

    The judge's reason, each metric's score and reason, and the persona's own
    claim from its ``end_call``. The conversation itself was printed as it
    happened, and the one-line verdict follows.
    """
    if result.error:
        return
    # A blank line sets each section apart, and one more before the verdict line.
    print()
    print(f"    {_bold('judge:')} {result.reason}")
    if result.metrics:
        print()
        print(f"    {_bold('metrics:')}")
        for metric in result.metrics:
            score = (_green if metric.passed else _red)(
                "unscored" if metric.score is None else f"{metric.score:.2f}"
            )
            bound = f" (min {metric.min_score:.2f})" if metric.min_score is not None else ""
            failed = [v for v in metric.verdicts if not v.passed]
            summary = f"{len(metric.verdicts) - len(failed)}/{len(metric.verdicts)} turns"
            print(
                f"      {_color(metric.name + ':', '36')} {score}{_dim(bound)}"
                f"{_dim(' | ' + (summary if metric.verdicts else metric.reason))}"
            )
            bot_turns = [m["content"] for m in result.messages if m["role"] == "assistant"]
            for verdict in failed:
                said = bot_turns[verdict.turn - 1] if verdict.turn <= len(bot_turns) else ""
                print(
                    f"        {_red('✗')} {_dim(f'turn {verdict.turn}:')} "
                    f"{_fit_detail(said, 24)} {_dim('— ' + verdict.reason)}"
                )
    if result.end_call is not None:
        print()
        claim = _green("succeeded") if result.end_call.get("success") else _red("gave up")
        print(f"    {_bold('persona:')} {claim}: {result.end_call.get('reason', '')}")
    print()


def _record_path(record_dir: str | None, scenario_name: str) -> str | None:
    """Per-scenario recording path under ``record_dir``, or None when recording is off."""
    if not record_dir:
        return None
    return str(Path(record_dir) / f"{scenario_name}.wav")


def _expand_scenario_paths(paths: list[Path]) -> list[Path]:
    """Expand directory arguments into sorted YAML scenario paths.

    Both YAML suffixes are taken, matching the scenario names a manifest
    resolves. The fragments scenarios ``!include`` (judge, user, and simulator
    blocks, which have no ``name:``) are left out; a file given explicitly is
    always taken.
    """
    expanded: list[Path] = []
    for path in paths:
        if not path.is_dir():
            expanded.append(path)
            continue

        scenario_paths = sorted(
            scenario_path
            for scenario_path in path.iterdir()
            if scenario_path.suffix in SCENARIO_SUFFIXES
            and scenario_path.is_file()
            and is_scenario_file(scenario_path)
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
        kind = (
            EvalKind.SIMULATION if isinstance(loaded, EvalSimulationScenario) else EvalKind.SCRIPT
        )
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
    params: EvalSessionParams,
    logs_dir: str,
    debug: bool,
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
            session = EvalSession.from_scenario(
                loaded, url, params=params.model_copy(update={"record_path": record_path})
            )
            if verbose:
                session.add_event_handler("on_progress", _print_progress)
                if isinstance(loaded, EvalSimulationScenario):
                    print(f"    {_bold('conversation:')}")
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
    params: EvalSessionParams,
    logs_dir: str,
    verbose: bool,
    debug: bool,
    started: float,
) -> None:
    """Run scenarios sequentially against a fixed bot, with the suite's display.

    A live dashboard in an interactive terminal; ``--verbose`` (per-turn lines, a
    simulation's conversation as it happens, and its verdict detail) or a piped
    stdout fall back to streamed result lines instead.
    """

    async def go(run: EvalRun, verbose: bool) -> None:
        await _execute_scenario(
            run,
            audio=audio,
            record_dir=record_dir,
            params=params,
            logs_dir=logs_dir,
            debug=debug,
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
            # A simulation's detail closes the conversation above it; the verdict
            # line comes last, as a scripted scenario's does after its turns.
            if verbose and isinstance(run.result, EvalSimulationResult):
                _print_simulation_detail(run.result)
            _print_eval_line(run)


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
        "simulation, the conversation as it happens and the judge's reasons at the end.",
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
    # Printed only under -a, to say which of the runs will actually record.
    if audio:
        _print_run_settings(runs, None, audio)

    params = EvalSessionParams(
        default_timeout_ms=timeout * 1000,
        cache_dir=cache_dir,
        use_cache=not no_cache,
        stop_bot=stop_bot,
        trigger_disconnect=trigger_disconnect,
    )
    started = time.monotonic()
    asyncio.run(
        _run_scenarios_all(
            runs,
            audio=audio,
            record_dir=record_dir,
            params=params,
            logs_dir=logs_dir,
            verbose=verbose,
            debug=debug,
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

    A simulation: how it ended, e.g. ``end_call``.
    """
    result = r.result
    if isinstance(result, EvalSimulationResult):
        return result.ended_by
    if result is None or not result.turns:
        return ""
    if any(t.status == "not_run" for t in result.turns):
        return ""
    failed = sum(1 for t in result.turns if t.status == "failed")
    if not failed:
        return ""
    return f"{len(result.turns) - failed}/{len(result.turns)} turns"


def _fmt_duration(seconds: float) -> str:
    """Human-friendly elapsed time, e.g. ``12.3s`` or ``2m 04s``.

    Tenths under a minute, then minutes and seconds, deciding on the rounded
    value so nothing ever reads ``60.0s``.
    """
    tenths = round(seconds, 1)
    if tenths < 60:
        return f"{tenths:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def _clock(seconds: float) -> str:
    """A running clock, padded to the width of ``2m 04s`` so what sits beside it never moves."""
    return f"{_fmt_duration(seconds):>6}"


def _eval_status_cell(r: EvalRun, spinner: Spinner):
    """A rich renderable for the status column.

    Args:
        r: The run the cell is for.
        spinner: The spinner shown while it runs. One instance outlives the
            frames: a spinner animates from the time it was first drawn, so a
            fresh one per frame would sit on its first glyph forever.
    """
    if r.status == "pending":
        return Text("·", style="dim")
    if r.status == "running":
        return spinner
    glyph, style, _ = _EVAL_GLYPH[_eval_verdict(r)]
    return Text(glyph, style=style)


def _pass_rate(passed: int, done: int) -> str:
    """``23/50 (46%)`` over the attempts that finished, or a placeholder if none have."""
    if not done:
        return "—"
    return f"{passed}/{done} ({100 * passed // done}%)"


# Rate colors, as a rich style and the matching ANSI code for the non-TTY path.
_RATE_ANSI = {"green": "32", "red": "31", "dim": "2"}


def _rate_level(passed: int, done: int) -> str:
    """Color for a pass rate: green when every finished attempt passed, red once one has not, dim before any finished."""
    if not done:
        return "dim"
    return "green" if passed == done else "red"


def _row_seconds(group: list[EvalRun]) -> float | None:
    """How long a row has been going: from its first attempt's start to now, or to its last attempt's end.

    ``None`` before any attempt has started. An attempt without a start time,
    one loaded rather than run, counts for its duration alone.
    """
    started = [r.started_at for r in group if r.started_at is not None]
    if not started:
        durations = [r.duration_ms for r in group if r.duration_ms is not None]
        return sum(durations) / 1000 if durations else None
    if any(r.status != "done" for r in group):
        return time.monotonic() - min(started)
    ended = [
        r.started_at + r.duration_ms / 1000
        for r in group
        if r.started_at is not None and r.duration_ms is not None
    ]
    return (max(ended) if ended else time.monotonic()) - min(started)


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
        # Shared by every running row and kept across frames, so it animates.
        self._spinner = Spinner("dots", style="cyan")

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Rendered against the height Rich hands over at draw time, which is the
        # terminal's as it is now (a tmux pane, a resized window), rather than
        # the size the module's console measured at import.
        height, width = options.max_height, options.max_width
        if self.grouped:
            yield self._render_grouped(height, width)
            return
        rows = self.runs
        n = len(rows)
        start, end = self._window([r.status for r in rows], height)

        cells = []
        for r in rows[start:end]:
            if r.status == "running" and r.started_at is not None:
                detail = _clock(time.monotonic() - r.started_at)
            elif r.status == "done" and r.duration_ms is not None:
                detail = f"{r.duration_ms}ms"
            else:
                detail = ""
            cells.append(
                (
                    _eval_status_cell(r, self._spinner),
                    Text(r.bot),
                    Text(r.scenario, style="cyan"),
                    Text(detail, style="dim"),
                )
            )

        yield self._framed(self._table(cells, 4, width), start, end, n)

    @staticmethod
    def _table(cells: list[tuple], columns: int, width: int) -> Table:
        """A grid of ``cells`` that fits ``width`` with every row on one line.

        The window counts rows as lines, so a row that wrapped would push the
        tally off the bottom of the screen. The status glyph and the numbers
        are as wide as their widest cell, and the bot and scenario columns get
        what is left, cut with an ellipsis only when the row would not fit. A
        narrow pane thus shortens a path rather than dropping the glyph or the
        numbers, and a wide one keeps the columns together at the left, the
        timing next to its row.

        Args:
            cells: One tuple of renderables per row: status, bot, scenario,
                then the numbers, right-justified.
            columns: How many cells a row has, for an empty window.
            width: The terminal's width.
        """
        widths = [
            max((1 if isinstance(row[i], Spinner) else row[i].cell_len for row in cells), default=1)
            for i in range(columns)
        ]
        fixed = sum(w for i, w in enumerate(widths) if i not in (1, 2)) + 2 * (columns - 1)
        budget = max(20, width - fixed - 1)
        paths = widths[1] + widths[2]
        if paths > budget:
            # Cut the cells rather than cap the columns: a grid whose natural
            # width fits is one Rich never shrinks, so the numbers stay whole.
            bot_width = max(10, budget * widths[1] // paths)
            scenario_width = max(10, budget - bot_width)
            for row in cells:
                row[1].truncate(bot_width, overflow="ellipsis")
                row[2].truncate(scenario_width, overflow="ellipsis")

        table = Table.grid(padding=(0, 2))
        for i in range(columns):
            table.add_column(no_wrap=True, justify="right" if i >= 3 else "left")
        for row in cells:
            table.add_row(*row)
        return table

    def _window(self, statuses: list[str], height: int) -> tuple[int, int]:
        """The slice of rows to show, sized to the terminal and sliding with the active runs.

        Only as many rows as fit above the pinned tally are shown. Otherwise,
        once the first screenful finishes, Rich crops the bottom, hiding the
        still running and pending rows and the tally itself, and there is no
        way to see the rest make progress.

        Args:
            statuses: Each row's status, ``running``, ``done`` or pending, in
                display order.
            height: The lines available to the whole dashboard.

        Returns:
            The ``(start, end)`` of the rows to render.
        """
        n = len(statuses)
        avail = max(3, height - 3)  # leave room for a blank line + the tally (+1 slack)
        if n <= avail:
            return 0, n
        # Anchor on the active frontier (the last running row, or the first
        # pending one) and keep a couple of upcoming rows in view below it, so
        # completed rows scroll off the top as new ones start.
        running = [i for i, status in enumerate(statuses) if status == "running"]
        done_n = sum(1 for status in statuses if status == "done")
        anchor = max(running) if running else min(done_n, n - 1)
        body = max(1, avail - 2)  # 2 lines reserved for the ↑/↓ "more" markers
        end = min(n, anchor + 1 + min(2, body - 1))
        start = max(0, end - body)
        end = min(n, start + body)
        return start, end

    def _framed(self, table: Table, start: int, end: int, n: int) -> Group:
        """The table between its scroll markers, with the tally pinned below."""
        total = len(self.runs)
        done = sum(1 for r in self.runs if r.status == "done")
        passed = sum(1 for r in self.runs if _eval_verdict(r) == "passed")
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

    def _render_grouped(self, height: int, width: int) -> Group:
        """One row per (bot, scenario), showing that pair's pass rate and pace."""
        groups = list(_grouped_runs(self.runs).items())
        n = len(groups)
        start, end = self._window(
            [
                "done"
                if all(r.status == "done" for r in group)
                else "running"
                if any(r.status == "running" or r.stopping for r in group)
                else "pending"
                for _, group in groups
            ],
            height,
        )

        # The progress columns are as wide as the widest text any row can reach,
        # "10 left" and "3/3 (100%)", from the first frame, so nothing moves as
        # the counts change.
        left_width = max((len(f"{len(g)} left") for _, g in groups), default=0)
        rate_width = max((len(_pass_rate(len(g), len(g))) for _, g in groups), default=0)

        cells = []
        for (bot, scenario), group in groups[start:end]:
            done = [r for r in group if r.status == "done"]
            passed, _, _ = _group_outcome(group)
            # Three states, and only the last is a verdict: spinning while a slot is
            # held, a still dot while waiting for one, and ✓/✗ once every attempt is
            # in. A slot stays held while a finished attempt's bot is stopped, so
            # the row keeps spinning through that tail rather than looking idle.
            # Motion is reserved for the runs actually in flight — most rows of a
            # long sweep are waiting, and animating those too would leave nothing
            # for movement to mean.
            if len(done) == len(group):
                glyph, style, _ = _EVAL_GLYPH["passed" if passed == len(group) else "failed"]
                status = Text(glyph, style=style)
            elif any(r.status == "running" or r.stopping for r in group):
                status = self._spinner
            else:
                status = Text("·", style="dim")
            # Two columns say where the row is: what is left while attempts
            # remain, and how many of the row's attempts have passed. The rate
            # is over every attempt, so it climbs as they come in, and its color
            # is a verdict on the ones that finished. Beside them the row's
            # clock, running from its first attempt's start and stopped at its
            # last one's end.
            remaining = len(group) - len(done)
            left = Text(f"{remaining} left".rjust(left_width) if remaining else "", style="dim")
            rate = Text(
                _pass_rate(passed, len(group)).rjust(rate_width),
                style=_rate_level(passed, len(done)),
            )
            seconds = _row_seconds(group)
            clock = "" if seconds is None else _clock(seconds)
            cells.append(
                (
                    status,
                    Text(bot),
                    Text(scenario, style="cyan"),
                    left,
                    rate,
                    Text(clock, style="dim"),
                )
            )

        return self._framed(self._table(cells, 6, width), start, end, n)


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


def _audio_runs(runs: list[EvalRun]) -> int:
    """How many of ``runs`` are audio-mode, the only kind the harness records.

    A run whose scenario fails to load counts as text: it fails before it could
    record anyway.
    """
    bot_audio: dict[Path, bool] = {}
    for r in runs:
        if r.scenario_path not in bot_audio:
            try:
                bot_audio[r.scenario_path] = load_scenario_file(r.scenario_path).bot_audio
            except Exception:  # noqa: BLE001
                bot_audio[r.scenario_path] = False
    return sum(bot_audio[r.scenario_path] for r in runs)


def _recording_setting(runs: list[EvalRun], record: bool) -> str:
    """The ``recording`` settings value: whether the selected runs will produce audio.

    Only an audio-mode run records: a text-mode bot skips TTS and the user's
    turns go over as text, so there is nothing to record. The note names the
    text-mode runs recording skips.
    """
    if not record:
        return "off"
    audio, total = _audio_runs(runs), len(runs)
    if audio == total:
        return "on"
    if audio == 0:
        return f"off {_color('(all runs text mode)', '33')}"
    return f"on {_color(f'({audio} of {total} runs; text mode skipped)', '33')}"


def _print_run_settings(runs: list[EvalRun], concurrency: int | None, record: bool) -> None:
    """Print how the suite will execute, in the shape of the configs above it.

    Only what the terminal doesn't otherwise show while the run is going, and that
    a manifest can turn on without it appearing on the command line: the run count
    (arithmetic, and the difference between 8 runs and several thousand), the
    concurrency, and whether audio is being recorded — a few MB per run, so it
    grows with the sweep, and only for audio-mode runs. What the caller just typed
    stays out; it's already in their scrollback. ``concurrency`` is ``None`` when
    the runs go one at a time against a fixed bot.
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
    settings = [("runs", counts)]
    if concurrency is not None:
        settings.append(("concurrency", str(concurrency)))
    settings.append(("recording", _recording_setting(runs, record)))
    print("Settings:")
    for label, value in settings:
        print(f"  {_color(f'{label:11s}', '1')} -> {value}")
    print()


def _print_scenario_configs(runs: list[EvalRun]) -> None:
    """Print each distinct scenario's and simulation's config once, up front.

    Done before the runs (not per-run) so it doesn't interleave with the live
    display, with a trailing blank line separating it from the runs.
    """
    for kind, heading in (
        (EvalKind.SCRIPT, "Scripted scenarios:"),
        (EvalKind.SIMULATION, "Simulated scenarios:"),
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
    print()
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
                print(f"      {_red('•')} {result.failure} {_dim(tally)}")
        elif r.result is not None:
            print(header)
            for f in r.result.failures:
                turn = f"turn {f.turn_index}" if f.turn_index >= 0 else "run"
                print(
                    f"      {_red('•')} {turn} {f.event_name} {_color(f.kind, '31')} — {f.reason}"
                )


def _group_outcome(group: list[EvalRun]) -> tuple[int, int, int]:
    """A (bot, scenario) group's ``(passed, completed, errored)``.

    Errored runs are left out of the completed count: a run that crashed or never
    connected says nothing about whether the bot did its job, and folding it into
    the rate would both drag the rate down unfairly and hide an infrastructure
    problem behind a behavioral one.
    """
    verdicts = [_eval_verdict(r) for r in group]
    passed = sum(1 for v in verdicts if v == "passed")
    errored = sum(1 for v in verdicts if v == "error")
    completed = sum(1 for v in verdicts if v in ("passed", "failed", "skipped"))
    return passed, completed, errored


def _print_repeat_summary(runs: list[EvalRun], failed: list[EvalRun], *, show_rates: bool) -> None:
    """Print per-(bot, scenario) pass rates and every failure.

    A repeated sweep is measuring a rate, not a verdict, so the useful output is
    how often each pair passed and which runs account for the rest. A simulation
    running its own ``runs`` is a requirement instead, marked ✓ or ✗ beside its
    rate: every run had to pass. Either way how many runs errored (those are
    outside the rate) and the mean run time follow.
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
            passed, completed, errored = _group_outcome(group)
            rate_text = f"{_pass_rate(passed, completed):>14s}"
            rate = _color(rate_text, _RATE_ANSI[_rate_level(passed, completed)])
            extra = []
            if errored:
                extra.append(f"{errored} errored")
            seconds = _row_seconds(group)
            if seconds is not None:
                extra.append(_fmt_duration(seconds))
            mark = ""
            if not group[0].sweep:
                mark = f"  {_green('✓') if passed == len(group) else _red('✗')}"
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
        # caller's policy, so failures there are data rather than a build break.
        # A simulation's own runs are a requirement, and one of them failing is
        # a break.
        return 1 if any(not r.sweep for r in failed) else 0
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
    params: EvalSessionParams,
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
                params=params,
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
            params=params,
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
    kind: EvalKind = typer.Option(None, "-k", "--kind", help="Only scenarios of this kind."),
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
        "each simulation this many times instead of its own runs (1 included); a "
        "repeated sweep reports rates and exits 0. Attempts interleave across bots "
        "and each writes its own logs.",
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
    runs = suite.filter(pattern=pattern, scenario=scenario, kind=kind)
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
            EvalSessionParams(default_timeout_ms=timeout * 1000, use_cache=not no_cache),
        )
    )
    exit_code = _finalize_evals(
        runs, run_dir, time.monotonic() - started, dashboard_shown=_console.is_terminal
    )
    raise typer.Exit(code=exit_code)
