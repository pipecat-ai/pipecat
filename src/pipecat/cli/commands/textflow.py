#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""``pipecat textflow`` typer commands.

Thin CLI wrappers over :mod:`pipecat.textflow`. Mounted as the ``textflow``
subcommand of the ``pipecat`` CLI (see :mod:`pipecat.cli.main`).
"""

import asyncio
import json
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from pipecat.textflow.capture import TextFlowCaptureError
from pipecat.textflow.render import render_trace
from pipecat.textflow.setup import TextFlowSetup, presets
from pipecat.textflow.trace import TextFlowRunError, run_textflow
from pipecat.textflow.transforms import registered_transforms

textflow_app = typer.Typer(
    name="textflow",
    help="Trace a theoretical LLM response through a pipeline's text path.",
    no_args_is_help=True,
)

_console = Console()


def _load_setup(
    setup_path: Path | None, preset: str | None, bot: Path | None = None
) -> TextFlowSetup:
    if bot is not None:
        return TextFlowSetup(name=bot.stem, bot=str(bot))
    if setup_path is not None:
        return TextFlowSetup.load(str(setup_path))
    try:
        return TextFlowSetup.preset(preset or "plain")
    except KeyError as e:
        _console.print(f"[red]{e.args[0]}[/red]")
        raise typer.Exit(1) from e


@textflow_app.command("run")
def run(
    text: str | None = typer.Argument(
        None, help="The LLM response to trace. The setup's sample text when omitted."
    ),
    text_file: Path | None = typer.Option(
        None, "--text-file", "-t", help="A file holding the LLM response to trace."
    ),
    bot: Path | None = typer.Option(
        None,
        "--bot",
        "-b",
        help="A bot file whose own processors to run. Overrides --setup and --preset.",
    ),
    live_tts: bool | None = typer.Option(
        None,
        "--live-tts/--stand-in-tts",
        help="With --bot: run the bot's TTS service itself, calling its provider (the "
        "default), or keep the stand-in with the bot's text handling.",
    ),
    setup_path: Path | None = typer.Option(
        None, "--setup", "-s", help="A setup YAML file. Overrides --preset."
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="A bundled setup to use. Defaults to 'plain'."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the trace as JSON instead of tables."
    ),
    words: bool = typer.Option(True, "--words/--no-words", help="Include the per-word table."),
    steps: bool = typer.Option(False, "--steps", help="List every step of the run, in order."),
    quiet: bool = typer.Option(True, "--quiet/--logs", help="Hide the pipeline's own logs."),
):
    """Run a response through a setup and show what happened to it."""
    if quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    setup = _load_setup(setup_path, preset, bot)
    if live_tts is not None:
        if bot is None:
            _console.print("[red]--live-tts and --stand-in-tts need --bot.[/red]")
            raise typer.Exit(2)
        setup.tts.live = live_tts
    if text_file is not None:
        if text is not None:
            _console.print("[red]Give the response as text or as --text-file, not both.[/red]")
            raise typer.Exit(2)
        text = text_file.read_text(encoding="utf-8").rstrip("\n")
    try:
        trace = asyncio.run(run_textflow(setup, text))
    except (TextFlowCaptureError, TextFlowRunError) as e:
        _console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    if json_output:
        print(json.dumps(trace.model_dump(mode="json"), indent=2))
    else:
        render_trace(trace, _console, words=words, steps=steps)


@textflow_app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Interface to listen on."),
    port: int = typer.Option(7865, "--port", help="Port to listen on."),
    bot: Path | None = typer.Option(
        None, "--bot", "-b", help="A bot file the page opens with, instead of a preset."
    ),
):
    """Serve the interactive page."""
    try:
        from pipecat.textflow.server import serve as serve_page
    except ImportError as e:
        _console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    _console.print(f"Serving the text flow page at http://{host}:{port}")
    serve_page(host=host, port=port, bot=str(bot.resolve()) if bot is not None else None)


@textflow_app.command("presets")
def list_presets(
    name: str | None = typer.Argument(
        None, help="Print this preset as YAML, to start a setup from."
    ),
):
    """List the bundled setups, or print one as YAML."""
    if name is not None:
        print(_load_setup(None, name).to_yaml())
        return
    table = Table(title="Presets")
    table.add_column("name", no_wrap=True)
    table.add_column("description")
    for preset in presets():
        table.add_row(preset.name, preset.description)
    _console.print(table)


@textflow_app.command("transforms")
def list_transforms():
    """List the transforms a setup can name."""
    table = Table(title="Transforms")
    table.add_column("name", no_wrap=True)
    table.add_column("description")
    table.add_column("options")
    for entry in registered_transforms():
        options = ", ".join(f"{o.name} ({o.kind}, default {o.default!r})" for o in entry.options)
        table.add_row(entry.name, entry.description, options)
    _console.print(table)
