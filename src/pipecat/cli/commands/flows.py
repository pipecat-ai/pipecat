#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``pipecat flows`` typer commands.

Thin CLI wrappers over :mod:`pipecat.flows.validation`. Mounted as the
``flows`` subcommand of the ``pipecat`` CLI (see :mod:`pipecat.cli.main`).
"""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import typer

flows_app = typer.Typer(
    name="flows",
    help="Check flow configs.",
    no_args_is_help=True,
)


@flows_app.command("validate")
def validate(
    config: Path = typer.Argument(
        ...,
        help="Flow config file (.yaml, .yml, or .json), or '-' to read YAML from stdin.",
        show_default=False,
    ),
    tools: str | None = typer.Option(
        None,
        "--tools",
        help=(
            "Tools module to check tool and handler references against: a path to a "
            ".py file or an importable module name. Importing it runs its top-level code."
        ),
        show_default=False,
    ),
    variables: list[str] = typer.Option(
        [],
        "--var",
        help="NAME=VALUE for a {{ variable }} the config uses. Repeatable. "
        "When any are given, every variable must be supplied.",
        show_default=False,
    ),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors."),
    json_output: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Check a flow config and report every error and warning.

    Structure is always checked. Tool and handler references are checked when
    --tools is given; variables when any --var is given. The verdict names
    what was checked. Exits 1 on errors (or on warnings with --strict). The
    format's JSON Schema ships as pipecat/flows/flow_config.schema.json.
    """
    from pipecat.flows.validation import validate_flow

    label = "stdin" if str(config) == "-" else str(config)
    source: str | Path = sys.stdin.read() if str(config) == "-" else config

    tools_ns = _load_tools(tools) if tools else None
    values = _parse_variables(variables) if variables else None

    report = validate_flow(source, tools=tools_ns, variables=values, base_dir=Path.cwd())

    failed = not report.ok or (strict and report.warnings)

    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
        raise typer.Exit(1 if failed else 0)

    checked = ["structure"]
    if tools_ns is not None:
        checked.append("tools")
    if values is not None:
        checked.append("variables")
    n_err, n_warn = len(report.errors), len(report.warnings)
    verdict = (
        "OK" if not report.issues else f"{_count(n_err, 'error')}, {_count(n_warn, 'warning')}"
    )
    typer.echo(f"{label}: {verdict} ({', '.join(checked)})")

    for issue in report.issues:
        typer.echo(f"{issue.level:<8}{issue.message}")

    raise typer.Exit(1 if failed else 0)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _parse_variables(pairs: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise typer.BadParameter(f"expected NAME=VALUE, got '{pair}'", param_hint="--var")
        values[name] = value
    return values


def _load_tools(spec: str) -> Any:
    """Import a tools module from a ``.py`` path or a module name.

    A file's directory goes on ``sys.path`` first, so the module's own
    imports resolve the way they do when the bot runs from that directory.
    """
    path = Path(spec)
    if path.suffix == ".py":
        if not path.is_file():
            raise typer.BadParameter(f"no such file: {spec}", param_hint="--tools")
        sys.path.insert(0, str(path.resolve().parent))
        module_spec = importlib.util.spec_from_file_location(f"_pipecat_tools_{path.stem}", path)
        if module_spec is None or module_spec.loader is None:
            raise typer.BadParameter(f"cannot import {spec}", param_hint="--tools")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module
    sys.path.insert(0, str(Path.cwd()))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(spec)
    except ImportError as e:
        raise typer.BadParameter(f"cannot import module '{spec}': {e}", param_hint="--tools") from e
