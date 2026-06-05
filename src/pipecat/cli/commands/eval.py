#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``pipecat eval`` typer commands.

Thin CLI wrappers over :mod:`pipecat.evals` — all the load/run logic lives in
that module. Loaded by pipecat-cli via the ``pipecat_cli.extensions`` entry
point and also reachable as ``python -m pipecat.evals``.
"""

import asyncio
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from pipecat.evals.harness import AssertionFailure, EvalResult, TurnProgress, run_scenario
from pipecat.evals.scenario import describe_config, load_scenario
from pipecat.evals.suite import SuiteRun, filter_runs, load_manifest, run_suite

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


def _format_detail(p: TurnProgress) -> str:
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
    help="Run behavioral evals against a Pipecat bot.",
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


def _yellow(s: str) -> str:
    return _color(s, "33")


def _print_progress(p: TurnProgress) -> None:
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


def _print_result(result: EvalResult, label: str) -> None:
    if result.skipped:
        print(f"  {_yellow('⊘')} {label} {_dim(f'— skipped: {result.skipped}')}")
        return
    badge = _green("✓") if result.passed else _red("✗")
    print(f"  {badge} {label} {_dim(f'({result.duration_ms}ms)')}")
    if not result.passed:
        for f in result.failures:
            print(f"      {_red('•')} {f}")


def _rich_turn(p: TurnProgress) -> Text:
    """Render a turn/expectation line for the live display."""
    if p.status == "turn":
        label = f'"{p.event_name}"' if p.event_name else "(observe)"
        return Text.assemble("    ", (f"turn {p.turn_index}", "dim"), f" → {label}")
    ok = p.status == "matched"
    line = Text("      ", no_wrap=True, overflow="ellipsis")
    line.append("✓ " if ok else "✗ ", style="green" if ok else "red")
    line.append(p.event_name)
    if p.detail:
        used = 6 + 2 + len(p.event_name) + 4  # indent + badge + name + "  — "
        line.append(f"  — {_fit_detail(_format_detail(p), used)}", style="dim")
    return line


def _rich_header(label: str, result: EvalResult) -> Text:
    """Render the final (✓/✗/⊘) header that replaces the spinner."""
    if result.skipped:
        return Text.assemble(("⊘ ", "yellow"), label, (f" — skipped: {result.skipped}", "dim"))
    badge, style = ("✓", "green") if result.passed else ("✗", "red")
    return Text.assemble((f"{badge} ", style), label, (f" ({result.duration_ms}ms)", "dim"))


def _rich_failure(f: AssertionFailure) -> Text:
    return Text.assemble(("      • ", "red"), str(f))


async def _run_one_live(
    scenario, url: str, label: str, verbose: bool, record_path: str | None, cache_dir: str | None
) -> EvalResult:
    """Run one eval with a spinning header that updates to ✓/✗, turns streaming below."""
    lines: list[Text] = []
    spinner = Spinner("dots", text=Text(f" {label}"))
    with Live(spinner, console=_console, refresh_per_second=12.5) as live:

        def on_progress(p: TurnProgress) -> None:
            lines.append(_rich_turn(p))
            live.update(Group(spinner, *lines))

        result = await run_scenario(
            scenario,
            url,
            on_progress=on_progress if verbose else None,
            record_path=record_path,
            cache_dir=cache_dir,
        )
        # Connect-level failures aren't turn lines; surface them under the header.
        extra = [_rich_failure(f) for f in result.failures if f.turn_index < 0]
        live.update(Group(_rich_header(label, result), *lines, *extra))
    return result


def _record_path(record_dir: str | None, scenario_name: str) -> str | None:
    """Per-scenario recording path under ``record_dir``, or None when recording is off."""
    if not record_dir:
        return None
    return str(Path(record_dir) / f"{scenario_name}.wav")


async def _run_all(
    paths: list[Path],
    agent_url: str,
    verbose: bool,
    audio: bool,
    record_dir: str,
    cache_dir: str | None,
) -> int:
    passed = 0
    failed = 0
    skipped = 0
    for path in paths:
        try:
            scenario = load_scenario(path)
        except (ValueError, FileNotFoundError) as e:
            print(f"  {_red('✗')} {path.name}: {e}")
            failed += 1
            continue

        label = f"{path.name}::{scenario.name}"
        url = scenario.fixtures.get("agent_url") or scenario.fixtures.get("bot_url") or agent_url
        record_path = _record_path(record_dir, scenario.name) if audio else None
        print(f"  {_color(scenario.name + ':', '1;36')}")
        for line in describe_config(scenario, color=_supports_color()).splitlines():
            print(f"    {line}")
        if _console.is_terminal:
            result = await _run_one_live(scenario, url, label, verbose, record_path, cache_dir)
        else:
            on_progress = _print_progress if verbose else None
            result = await run_scenario(
                scenario, url, on_progress=on_progress, record_path=record_path, cache_dir=cache_dir
            )
            _print_result(result, label)
        if result.skipped:
            skipped += 1
        elif result.passed:
            passed += 1
        else:
            failed += 1

    print()
    ran = passed + failed
    summary = f"{passed}/{ran} scenarios passed"
    if skipped:
        summary += f", {skipped} skipped"
    if failed == 0:
        print(_green(f"PASS — {summary}"))
        return 0
    print(_red(f"FAIL — {summary}"))
    return 1


@eval_app.command("run")
def run(
    scenarios: list[Path] = typer.Argument(..., help="One or more scenario YAML files."),
    agent_url: str = typer.Option(
        "ws://localhost:7860",
        "--agent-url",
        help="WebSocket URL of the agent's eval transport. "
        "Overridden per-scenario by fixtures.agent_url.",
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
) -> None:
    """Run one or more evals against a bot."""
    # In an interactive terminal, quiet pipecat's INFO/DEBUG logs (e.g. from the
    # user-audio synthesis pipeline) so they don't corrupt the live display.
    if _console.is_terminal:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    exit_code = asyncio.run(_run_all(scenarios, agent_url, verbose, audio, record_dir, cache_dir))
    raise typer.Exit(code=exit_code)


#
# `pipecat eval suite` — spawn the agents in a manifest and run their scenarios.
#

_SUITE_GLYPH = {
    "passed": ("✓", "green", "32"),
    "failed": ("✗", "red", "31"),
    "skipped": ("⊘", "yellow", "33"),
    "error": ("✗", "red", "31"),
}


def _suite_verdict(r: SuiteRun) -> str:
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


def _suite_status_cell(r: SuiteRun):
    """A rich renderable for the status column (spinner while running)."""
    if r.status == "pending":
        return Text("·", style="dim")
    if r.status == "running":
        return Spinner("dots", style="cyan")
    glyph, style, _ = _SUITE_GLYPH[_suite_verdict(r)]
    return Text(glyph, style=style)


class _SuiteDashboard:
    """Live table rendered every frame from the shared list of runs."""

    def __init__(self, runs: list[SuiteRun]):
        self.runs = runs

    def __rich__(self) -> Group:
        table = Table.grid(padding=(0, 2))
        table.add_column()  # status
        table.add_column()  # agent
        table.add_column()  # scenario
        table.add_column(justify="right")  # timing
        for r in self.runs:
            if r.status == "running" and r.started_at is not None:
                detail = f"{int(time.monotonic() - r.started_at)}s"
            elif r.status == "done" and r.duration_ms is not None:
                detail = f"{r.duration_ms}ms"
            else:
                detail = ""
            table.add_row(
                _suite_status_cell(r),
                Text(r.agent),
                Text(r.scenario, style="cyan"),
                Text(detail, style="dim"),
            )
        done = sum(1 for r in self.runs if r.status == "done")
        passed = sum(1 for r in self.runs if _suite_verdict(r) == "passed")
        summary = Text(f"{passed}/{len(self.runs)} passed  ·  {done}/{len(self.runs)} done", "bold")
        return Group(table, Text(""), summary)


def _print_suite_line(r: SuiteRun) -> None:
    """Print a single result line (non-TTY fallback, streamed as each finishes)."""
    if r.status != "done":
        return
    glyph, _, code = _SUITE_GLYPH[_suite_verdict(r)]
    extra = f"({r.duration_ms}ms)" if r.duration_ms is not None and not r.error else (r.error or "")
    print(f"  {_color(glyph, code)} {r.agent} {_color(r.scenario, '36')} {_dim(extra)}", flush=True)


def _finalize_suite(runs: list[SuiteRun], runs_dir: Path, elapsed_s: float) -> int:
    """Print the failed set + final tally; return the process exit code."""
    failed = [r for r in runs if _suite_verdict(r) in ("failed", "error")]
    passed = sum(1 for r in runs if _suite_verdict(r) == "passed")
    skipped = sum(1 for r in runs if _suite_verdict(r) == "skipped")

    if failed:
        print()
        print(f"  {_color(f'Failed ({len(failed)}):', '1;31')}")
        for r in failed:
            if r.error:
                print(f"  {_red('✗')} {r.agent} {_color(r.scenario, '36')} {_dim('— ' + r.error)}")
            elif r.result is not None:
                print(f"  {_red('✗')} {r.agent} {_color(r.scenario, '36')}")
                for f in r.result.failures:
                    print(f"      {_red('•')} {f}")

    print()
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
    runs: list[SuiteRun], manifest, logs_dir: Path, record_dir: Path | None
) -> None:
    """Run the suite with a live dashboard (TTY) or streamed lines (piped)."""
    if _console.is_terminal:
        with Live(_SuiteDashboard(runs), console=_console, refresh_per_second=12.5):
            await run_suite(runs, manifest, logs_dir, record_dir=record_dir)
    else:
        print(f"Running {len(runs)} eval(s), {manifest.concurrency} at a time...")
        await run_suite(
            runs, manifest, logs_dir, record_dir=record_dir, on_update=_print_suite_line
        )


@eval_app.command("suite")
def suite(
    manifest_path: Path = typer.Argument(..., help="Manifest YAML listing agents + scenarios."),
    pattern: str = typer.Option(
        None, "-p", "--pattern", help="Only agents whose path contains this."
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
    concurrency: int = typer.Option(
        None, "-c", "--concurrency", help="Override manifest concurrency."
    ),
    base_port: int = typer.Option(None, "--base-port", help="Override manifest base_port."),
    cache_dir: str = typer.Option(None, "--cache-dir", help="Override manifest cache_dir."),
    spawn: str = typer.Option(None, "--spawn", help="Override manifest spawn template."),
    python: str = typer.Option(None, "--python", help="Override manifest python interpreter."),
    audio: bool = typer.Option(False, "-a", "--audio", help="Record conversation audio."),
) -> None:
    """Spawn the agents in a manifest and run their scenarios concurrently.

    Any option also settable in the manifest takes precedence when passed here.
    """
    manifest = load_manifest(manifest_path)
    # CLI overrides win over the manifest.
    if concurrency is not None:
        manifest.concurrency = concurrency
    if base_port is not None:
        manifest.base_port = base_port
    if cache_dir is not None:
        manifest.cache_dir = cache_dir
    if spawn is not None:
        manifest.spawn = spawn
    if python is not None:
        manifest.python = python

    runs = filter_runs(manifest.runs, pattern=pattern, scenario=scenario)
    if not runs:
        print("No runs match.")
        raise typer.Exit(code=1)

    # Output base (CLI overrides the manifest's runs_dir); a per-run subdir named by
    # --name (default a timestamp) holds this run's logs and recordings.
    base = runs_dir or manifest.runs_dir or Path("eval-runs")
    run_dir = base / (name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    logs_dir = run_dir / "logs"
    record_dir = (run_dir / "recordings") if (audio or manifest.record) else None

    # Print each distinct scenario's config up front (per-run would interleave).
    print("Scenarios:")
    seen: set[str] = set()
    for r in runs:
        if r.scenario in seen:
            continue
        seen.add(r.scenario)
        try:
            cfg = describe_config(load_scenario(r.scenario_path), color=sys.stdout.isatty())
        except Exception as e:  # noqa: BLE001
            cfg = f"(failed to load: {e})"
        print(f"  {_color(r.scenario + ':', '1;36')}")
        for line in cfg.splitlines():
            print(f"    {line}")
    print()

    started = time.monotonic()
    asyncio.run(_run_suite_all(runs, manifest, logs_dir, record_dir))
    exit_code = _finalize_suite(runs, run_dir, time.monotonic() - started)
    raise typer.Exit(code=exit_code)
