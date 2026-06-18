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
    "The Pipecat CLI needs its optional dependencies (the `cli` extra), which aren't "
    "installed.\n\n"
    "Install the extra wherever you want the `pipecat` command:\n\n"
    "  • As a global tool (on your PATH):\n"
    '        uv tool install "pipecat-ai[cli]"     # or: pipx install "pipecat-ai[cli]"\n\n'
    "  • In your current project or virtualenv:\n"
    '        uv pip install "pipecat-ai[cli]"      # or: pip install "pipecat-ai[cli]"\n'
)

# Official optional sub-CLIs. Each ships as a separate plugin package that registers
# a Typer app under the ``pipecat_cli.extensions`` group. We list them in ``--help``
# even when they're not installed (as stubs) so they're discoverable, and the stub
# prints how to enable the plugin. Only first-party plugins belong here.
_KNOWN_EXTENSIONS: dict[str, tuple[str, str]] = {
    "cloud": ("pipecatcloud", "Deploy and manage bots on Pipecat Cloud"),
}


def _enable_hint(name: str, package: str) -> str:
    """Message shown when an official-but-uninstalled sub-CLI is invoked."""
    return (
        f"The `pipecat {name}` command requires the optional `{package}` plugin, "
        "which isn't installed.\n\n"
        "Enable it where the `pipecat` command lives:\n\n"
        "  • As a global tool (on your PATH), reinstall with the plugin:\n"
        f'        uv tool install "pipecat-ai[cli]" --with {package}\n\n'
        "  • In your current project or virtualenv:\n"
        f"        uv pip install {package}     # or: pip install {package}\n"
    )


_app = None


def _build_app():
    """Build the Typer app, importing the optional CLI dependencies lazily."""
    import importlib.metadata as importlib_metadata

    import typer
    from rich.console import Console

    from pipecat.cli.commands.create import create_command
    from pipecat.cli.commands.eval import eval_app
    from pipecat.cli.commands.init import init_command

    app = typer.Typer(
        name="pipecat",
        help="Command-line tools for building Pipecat AI applications.",
        add_completion=False,
    )
    console = Console()

    # `create` is a plain command (not a sub-Typer group) so it can take an optional
    # positional target path followed by options (e.g. `pc create . --bot-type web`).
    app.command("create", help="Create a new Pipecat project")(create_command)

    # `init` makes a project agent-ready (writes AGENTS.md + CLAUDE.md). ignore_unknown_options
    # lets it catch legacy scaffolder flags (now `pipecat create`) and redirect with a clear
    # message instead of an opaque "no such option" error.
    app.command(
        "init",
        help="Make a project agent-ready (writes AGENTS.md + CLAUDE.md)",
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )(init_command)

    # `eval` is a first-party sub-Typer group, built in (not a plugin extension).
    app.add_typer(eval_app, name="eval")

    # Discover CLI extensions (e.g. `cloud` from pipecatcloud). The entry-point
    # group is intentionally still named
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

    # For official plugins that aren't installed, register a discoverable stub so the
    # command still shows up in `pipecat --help` and, when invoked, prints how to enable
    # it instead of a bare "No such command". Installed plugins (above) take precedence.
    def _make_extension_stub(cmd_name: str, package: str):
        def _stub(ctx: typer.Context):
            print(_enable_hint(cmd_name, package), file=sys.stderr)
            raise typer.Exit(1)

        return _stub

    installed = {name for name, _ in extensions}
    for cmd_name, (package, help_text) in sorted(_KNOWN_EXTENSIONS.items()):
        if cmd_name in installed:
            continue
        app.command(
            cmd_name,
            help=f"{help_text} (requires {package})",
            context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        )(_make_extension_stub(cmd_name, package))

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
