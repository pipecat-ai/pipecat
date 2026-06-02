#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Main CLI entry point for Pipecat CLI."""

import importlib
from importlib.metadata import version

import typer
from rich.console import Console

from pipecat.cli.commands.init import init_app

app = typer.Typer(
    name="pipecat",
    help="CLI tool for scaffolding Pipecat AI voice agent projects",
    add_completion=False,
)

console = Console()

# Register commands
app.add_typer(init_app, name="init", help="Initialize a new Pipecat project")

# Load pipecat-cli extensions.
extensions = []
for ep in importlib.metadata.entry_points(group="pipecat_cli.extensions"):
    extension = ep.load()
    extensions.append((ep.name, extension))

# Sort by extension name (first tuple element)
extensions.sort(key=lambda x: x[0].lower())

# Add extensions.
for name, extension in extensions:
    app.add_typer(extension, name=name)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        try:
            pkg_version = version("pipecat-ai-cli")
        except Exception:
            pkg_version = "unknown"
        console.print(f"ᓚᘏᗢ Pipecat CLI Version: [green]{pkg_version}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Pipecat CLI - Build AI voice agents with ease."""
    pass


if __name__ == "__main__":
    app()
