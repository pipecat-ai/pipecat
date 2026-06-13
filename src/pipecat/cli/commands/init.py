#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``pipecat init`` — make a project agent-ready.

Writes the Pipecat coding-agent guide into a project so an AI coding agent
(Claude Code, Codex, …) picks it up automatically:

- ``AGENTS.md`` — the guide itself (read natively by most coding agents).
- ``CLAUDE.md`` — a one-line ``@AGENTS.md`` import so Claude Code loads it too.
- ``GETTING_STARTED.md`` — for the *developer*: how to drive the agent well
  (MCP setup, how to write the first prompt, what to expect in a session).
  Deliberately not in CLAUDE.md/AGENTS.md — those cost agent context every
  session, and this guidance is for the human.

The guide then drives the agent to scaffold the app with ``pipecat create``.

This is intentionally distinct from ``pipecat create`` (the project scaffolder,
formerly ``pipecat init``): a developer runs ``pipecat init`` first to make a
project agent-ready, then their coding agent runs ``pipecat create``.

Editing policy for the bundled guide: keep API specifics (signatures, imports,
parameter names) out of AGENTS.md — it is a static snapshot, so anything that
churns belongs in the live sources the guide's §3 points agents at.
"""

from pathlib import Path

import typer
from rich.console import Console

import pipecat.cli

console = Console()


# Directory holding the bundled AGENTS.md / CLAUDE.md (shipped as package data; see
# pyproject [tool.setuptools.package-data] "pipecat.cli" and MANIFEST.in).
_AGENT_TEMPLATES = Path(pipecat.cli.__file__).parent / "agent_templates"

_AGENTS_FILE = "AGENTS.md"
_CLAUDE_FILE = "CLAUDE.md"
_GETTING_STARTED_FILE = "GETTING_STARTED.md"


def init_command(
    ctx: typer.Context,
    target: str | None = typer.Argument(
        None,
        help="Directory to make agent-ready. Created if missing.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=f"Also overwrite an existing {_CLAUDE_FILE} ({_AGENTS_FILE} is always refreshed).",
    ),
):
    """Make a project agent-ready: AGENTS.md + CLAUDE.md + GETTING_STARTED.md.

    Run this first, read GETTING_STARTED.md, then open a coding session — the
    guide drives your agent to scaffold the app with ``pipecat create``.

    Examples::

        pipecat init                 # prompt for a project directory
        pipecat init my-bot          # set up ./my-bot
        pipecat init my-bot --force  # overwrite existing files in ./my-bot
        pipecat init .               # set up the current directory
    """
    # `pipecat init quickstart` was the old scaffolder shortcut; it now belongs to `create`.
    if target == "quickstart":
        console.print(
            "[yellow]`pipecat init quickstart` is now `pipecat create quickstart`.[/yellow]\n"
            "`pipecat init` makes a project agent-ready; the scaffolder moved to "
            "[bold]`pipecat create`[/bold]."
        )
        raise typer.Exit(1)

    # The old scaffolder flags (`--name`, `--bot-type`, `--stt`, `-o`, …) are no longer valid
    # here. `ignore_unknown_options` (set at registration) drops them into ctx.args; redirect
    # with a clear message instead of writing a half-set-up project or erroring opaquely.
    if ctx.args:
        unexpected = " ".join(ctx.args)
        console.print(
            f"[red]Unexpected arguments:[/red] {unexpected}\n\n"
            "`pipecat init` now makes a project agent-ready (writes AGENTS.md, CLAUDE.md, "
            "and GETTING_STARTED.md); it takes only an optional target directory and `--force`.\n"
            "The project scaffolder moved to [bold]`pipecat create`[/bold] — run "
            "`pipecat create --help`."
        )
        raise typer.Exit(1)

    # No argument: prompt for the directory. `target` is just a path — `.` sets up the
    # current directory, any other value names a folder (created below if it doesn't exist).
    if target is None:
        target = typer.prompt("Project directory", default="pipecat-bot").strip()

    # Validate before any write so a failure never leaves a half-initialized directory.
    target_dir = Path(target or ".")
    if target_dir.exists() and not target_dir.is_dir():
        console.print(f"[red]Error:[/red] {target_dir} exists and is not a directory.")
        raise typer.Exit(1)

    try:
        agents_src = (_AGENT_TEMPLATES / _AGENTS_FILE).read_text(encoding="utf-8")
        claude_src = (_AGENT_TEMPLATES / _CLAUDE_FILE).read_text(encoding="utf-8")
        getting_started_src = (_AGENT_TEMPLATES / _GETTING_STARTED_FILE).read_text(encoding="utf-8")
    except OSError as e:  # bundled data missing — a packaging regression
        console.print(f"[red]Error: bundled agent guide not found:[/red] {e}")
        raise typer.Exit(1)

    # Provenance stamp: the written guide is a static snapshot that otherwise looks like
    # hand-written project docs. The footer tells a later reader (human or agent) that it
    # is generated and refreshable, and which pipecat-ai wrote it — so a project whose
    # pinned version has moved on can spot a stale guide.
    footer = (
        f"\n<!-- Generated by `pipecat init` (pipecat-ai {pipecat.__version__}). "
        "Re-run `pipecat init` to refresh. -->\n"
    )

    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        agents_path = target_dir / _AGENTS_FILE
        # AGENTS.md is pipecat-owned: always (re)write it so re-running refreshes the guide.
        refreshed = agents_path.exists()
        agents_path.write_text(agents_src + footer, encoding="utf-8")
        console.print(f"[green]✔[/green] {'Refreshed' if refreshed else 'Wrote'} {agents_path}")

        getting_started_path = target_dir / _GETTING_STARTED_FILE
        # GETTING_STARTED.md is pipecat-owned developer guidance: always (re)write, like
        # AGENTS.md. (Only CLAUDE.md, the developer's own entry point, is clobber-protected.)
        refreshed = getting_started_path.exists()
        getting_started_path.write_text(getting_started_src + footer, encoding="utf-8")
        console.print(
            f"[green]✔[/green] {'Refreshed' if refreshed else 'Wrote'} {getting_started_path}"
        )

        claude_path = target_dir / _CLAUDE_FILE
        # CLAUDE.md is the developer's entry point: never clobber an existing one without --force.
        if claude_path.exists() and not force:
            console.print(
                f"[yellow]•[/yellow] Kept existing {claude_path} (use --force to overwrite)."
            )
        else:
            claude_path.write_text(claude_src, encoding="utf-8")
            console.print(f"[green]✔[/green] Wrote {claude_path}")
    except OSError as e:
        console.print(f"[red]Error writing files:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"\n[bold]Project is agent-ready.[/bold] Read [bold]{_GETTING_STARTED_FILE}[/bold] for "
        "how to prompt your agent, then open a coding session here — it will scaffold the app "
        "with `pipecat create`."
    )
