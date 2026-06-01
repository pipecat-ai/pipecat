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
import sys
from pathlib import Path

import typer

from pipecat.evals.harness import EvalResult, run_scenario
from pipecat.evals.scenario import load_scenario

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


def _print_result(result: EvalResult) -> None:
    badge = _green("✓") if result.passed else _red("✗")
    print(f"  {badge} {result.scenario_name} {_dim(f'({result.duration_ms}ms)')}")
    if not result.passed:
        for f in result.failures:
            print(f"      {_red('•')} {f}")


async def _run_all(paths: list[Path], bot_url: str) -> int:
    total = 0
    passed = 0
    for path in paths:
        try:
            scenario = load_scenario(path)
        except (ValueError, FileNotFoundError) as e:
            print(f"  {_red('✗')} {path}: {e}")
            total += 1
            continue

        url = scenario.fixtures.get("bot_url") or bot_url
        result = await run_scenario(scenario, url)
        _print_result(result)
        total += 1
        if result.passed:
            passed += 1

    print()
    summary = f"{passed}/{total} scenarios passed"
    if passed == total:
        print(_green(f"PASS — {summary}"))
        return 0
    print(_red(f"FAIL — {summary}"))
    return 1


@eval_app.command("run")
def run(
    scenarios: list[Path] = typer.Argument(..., help="One or more scenario YAML files."),
    bot_url: str = typer.Option(
        "ws://localhost:7860",
        "--bot-url",
        help="WebSocket URL of the bot's eval transport. "
        "Overridden per-scenario by fixtures.bot_url.",
    ),
) -> None:
    """Run one or more evals against a bot."""
    exit_code = asyncio.run(_run_all(scenarios, bot_url))
    raise typer.Exit(code=exit_code)
