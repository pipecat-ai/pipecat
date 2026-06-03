#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat CLI entry point.

The CLI ships with ``pipecat-ai`` but its dependencies are optional (the ``cli``
extra). ``run`` is the console-script target (``pipecat`` / ``pc``): it degrades
gracefully, printing an install hint instead of raising ``ImportError`` when the
``cli`` extra is not installed.

The Typer app is built lazily (``_build_app``) so importing this module never pulls
in the optional CLI dependencies; the module-level ``app`` attribute (used by tests
and the console script) is resolved on first access via ``__getattr__``.
"""

import sys

_INSTALL_HINT = (
    "The Pipecat CLI requires additional dependencies that are not installed.\n\n"
    "Install them with:\n\n"
    '    pip install "pipecat-ai[cli]"\n\n'
    "or install the CLI as an isolated tool:\n\n"
    '    uv tool install "pipecat-ai[cli]"\n'
)

_app = None


def _build_app():
    """Build the Typer app, importing the optional CLI dependencies lazily."""
    import importlib.metadata as importlib_metadata

    import typer
    from rich.console import Console

    from pipecat.cli.commands.init import init_app

    app = typer.Typer(
        name="pipecat",
        help="CLI tool for scaffolding Pipecat AI voice agent projects",
        add_completion=False,
    )
    console = Console()

    app.add_typer(init_app, name="init", help="Initialize a new Pipecat project")

    # Discover CLI extensions (e.g. `cloud` from pipecatcloud, `tail` from
    # pipecat-ai-tail). The entry-point group is intentionally still named
    # "pipecat_cli.extensions" for backward compatibility — renaming it would force
    # every plugin to re-release. (A future rename to "pipecat.cli.extensions" is a
    # separate, coordinated change.)
    extensions = [
        (ep.name, ep.load())
        for ep in importlib_metadata.entry_points(group="pipecat_cli.extensions")
    ]
    extensions.sort(key=lambda item: item[0].lower())
    for name, extension in extensions:
        app.add_typer(extension, name=name)

    def version_callback(value: bool):
        if value:
            try:
                pkg_version = importlib_metadata.version("pipecat-ai")
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

    return app


def __getattr__(name: str):
    # PEP 562: resolve `app` lazily so `from pipecat.cli.main import app` works without
    # importing typer at module load (keeps a bare, no-[cli] install import-safe).
    if name == "app":
        global _app
        if _app is None:
            _app = _build_app()
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run():
    """Console-script entry point; degrades gracefully when the ``cli`` extra is absent."""
    try:
        app = _build_app()
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        raise SystemExit(1)
    app()


if __name__ == "__main__":
    run()
