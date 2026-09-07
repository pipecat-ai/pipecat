"""Tests for optional sub-CLI (plugin) discovery and graceful enable hints.

When an official plugin (e.g. ``cloud`` → pipecatcloud) is not installed, the CLI
still lists it in ``--help`` as a stub and prints how to enable it when invoked —
rather than hiding it or erroring with "No such command".
"""

import importlib.metadata as importlib_metadata
import re
from types import SimpleNamespace

import click
import pytest
import typer
from typer.testing import CliRunner

from pipecat.cli.main import _KNOWN_EXTENSIONS, _build_app, _enable_hint, app

runner = CliRunner()

# rich emits ANSI color codes when it thinks the output is a terminal (e.g. in CI).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_installed = {ep.name for ep in importlib_metadata.entry_points(group="pipecat_cli.extensions")}

# Driven off the registry rather than a hard-coded list, so a new official plugin
# is covered the moment it is registered.
_OFFICIAL = sorted((name, pkg, help_text) for name, (pkg, help_text) in _KNOWN_EXTENSIONS.items())


def _skip_if_installed(name: str) -> None:
    """Skip a stub assertion for a plugin that is actually installed.

    The stub only exists when a plugin is absent; when it is installed its real
    Typer app is mounted instead. Skipping per plugin rather than per module
    means having one installed doesn't silently drop the assertions for the
    others — with two official plugins, a module-level skip is wrong in both
    directions.
    """
    if name in _installed:
        pytest.skip(f"the `{name}` plugin is installed; its stub path is not exercised")


def _norm(text: str) -> str:
    """Normalize help output so assertions survive rich's ANSI colors, wrapping, and borders."""
    text = _ANSI_RE.sub("", text)
    for ch in "│╭╮╰╯─":
        text = text.replace(ch, " ")
    return " ".join(text.split())


@pytest.mark.parametrize("name,package,help_text", _OFFICIAL)
class TestExtensionDiscovery:
    """`pipecat --help` advertises the official plugins even when uninstalled."""

    def test_help_lists_official_plugins_as_stubs(self, name, package, help_text):
        _skip_if_installed(name)
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = _norm(result.output)
        # The official sub-CLI is listed though it isn't installed...
        assert name in out
        # ...annotated with the package that provides it.
        assert f"requires {package}" in out


@pytest.mark.parametrize("name,package,help_text", _OFFICIAL)
class TestEnableHint:
    """Invoking an uninstalled official plugin prints an actionable hint."""

    def test_invoking_uninstalled_plugin_prints_hint_and_exits_1(self, name, package, help_text):
        _skip_if_installed(name)
        result = runner.invoke(app, [name])
        assert result.exit_code == 1
        # The actionable enable hint, not the bare Click error.
        assert "No such command" not in result.output
        assert f"--with {package}" in result.output

    def test_uninstalled_plugin_swallows_subcommands_and_options(self, name, package, help_text):
        # `pipecat cloud deploy --region x` must still reach the hint, not error on
        # the unknown `deploy` subcommand / `--region` option.
        _skip_if_installed(name)
        result = runner.invoke(app, [name, "somesubcommand", "--someflag", "x"])
        assert result.exit_code == 1
        assert "No such command" not in result.output
        assert f"--with {package}" in result.output

    def test_help_on_an_uninstalled_plugin_explains_how_to_install(self, name, package, help_text):
        """`--help` is the natural thing to type after spotting the command in
        `pipecat --help`. Without an empty help_option_names it renders an empty
        options panel and says nothing about installing the plugin.
        """
        _skip_if_installed(name)
        result = runner.invoke(app, [name, "--help"])
        assert f"--with {package}" in result.output

    def test_enable_hint_shows_both_install_forms(self, name, package, help_text):
        hint = _enable_hint(name, package)
        assert f'uv tool install "pipecat-ai[cli]" --with {package}' in hint
        assert f"pip install {package}" in hint


class _FakeEntryPoint:
    """Stands in for an installed plugin's entry point."""

    def __init__(self, name: str, dist_name: str, loader):
        self.name = name
        self.dist = SimpleNamespace(name=dist_name)
        self._loader = loader

    def load(self):
        return self._loader()


def _with_plugins(monkeypatch, *fakes):
    """Make plugin discovery return *fakes*, leaving other entry-point groups alone."""
    real = importlib_metadata.entry_points

    def _fake(*args, **kwargs):
        if kwargs.get("group") == "pipecat_cli.extensions":
            return list(fakes)
        return real(*args, **kwargs)

    monkeypatch.setattr(importlib_metadata, "entry_points", _fake)


class TestBrokenPluginIsolation:
    """One bad plugin must not take down the CLI.

    A plugin is third-party code imported on every invocation. Unguarded, a
    failed import propagated out of `_build_app` and was caught by `run()`'s
    `except ImportError`, so every command — `pipecat init` included — died
    printing "the `cli` extra isn't installed", which was not the problem.
    """

    def test_failing_plugin_is_skipped_and_cli_still_works(self, monkeypatch, capsys):
        def _explode():
            raise ImportError("missing transitive dependency")

        _with_plugins(monkeypatch, _FakeEntryPoint("broken", "brokenpkg", _explode))

        app_with_broken_plugin = _build_app()
        result = runner.invoke(app_with_broken_plugin, ["--help"])

        assert result.exit_code == 0
        assert "init" in _norm(result.output)
        assert "Warning: skipping the `broken` plugin" in capsys.readouterr().err

    def test_working_plugin_still_mounts_alongside_a_broken_one(self, monkeypatch):
        def _explode():
            raise RuntimeError("boom")

        good = typer.Typer(help="A working plugin.")

        @good.command()
        def ping():
            """Say hello."""

        _with_plugins(
            monkeypatch,
            _FakeEntryPoint("broken", "brokenpkg", _explode),
            _FakeEntryPoint("good", "goodpkg", lambda: good),
        )

        result = runner.invoke(_build_app(), ["--help"])
        assert result.exit_code == 0
        out = _norm(result.output)
        assert "good" in out
        assert "broken" not in out

    def test_non_typer_plugin_is_rejected_with_a_clear_message(self, monkeypatch, capsys):
        """add_typer accepts a click.Group, then fails later with an opaque AttributeError."""
        _with_plugins(monkeypatch, _FakeEntryPoint("bad", "badpkg", lambda: click.Group("bad")))

        result = runner.invoke(_build_app(), ["--help"])

        assert result.exit_code == 0
        err = capsys.readouterr().err
        assert "expected a typer.Typer" in err
        assert "Group" in err

    def test_installed_plugin_is_named_in_the_enable_hint(self, monkeypatch):
        """A broken plugin is still installed, so a reinstall hint must not drop it."""

        def _explode():
            raise ImportError("nope")

        _with_plugins(monkeypatch, _FakeEntryPoint("broken", "brokenpkg", _explode))

        result = runner.invoke(_build_app(), ["cloud"])
        assert "--with brokenpkg" in result.output
        assert "--with pipecatcloud" in result.output


class TestEnableHintPreservesInstalledPlugins:
    """`uv tool install --with` replaces the tool env, so the hint must name every plugin.

    Following a hint that lists only the missing plugin uninstalls the others —
    and their stubs then hint the user right back, each reinstall dropping
    whatever the last one added.
    """

    def test_installed_plugins_are_repeated(self):
        hint = _enable_hint("context-hub", "pipecat-ai-context-hub", ["pipecatcloud"])
        assert (
            'uv tool install "pipecat-ai[cli]" --with pipecatcloud '
            "--with pipecat-ai-context-hub" in hint
        )

    def test_no_installed_plugins_is_unchanged(self):
        assert 'uv tool install "pipecat-ai[cli]" --with pipecatcloud' in _enable_hint(
            "cloud", "pipecatcloud", []
        )

    def test_missing_package_is_not_repeated(self):
        """Defensive: the requested package appearing in both lists must not duplicate."""
        hint = _enable_hint("cloud", "pipecatcloud", ["pipecatcloud"])
        assert hint.count("--with pipecatcloud") == 1

    def test_pip_form_names_only_the_missing_package(self):
        """`uv pip install` adds to a venv, so it has no replacement problem."""
        hint = _enable_hint("context-hub", "pipecat-ai-context-hub", ["pipecatcloud"])
        assert "uv pip install pipecat-ai-context-hub" in hint
        assert "uv pip install pipecatcloud" not in hint
