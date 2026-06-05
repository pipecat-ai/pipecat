"""Tests for optional sub-CLI (plugin) discovery and graceful enable hints.

When an official plugin (e.g. ``cloud`` → pipecatcloud) is not installed, the CLI
still lists it in ``--help`` as a stub and prints how to enable it when invoked —
rather than hiding it or erroring with "No such command".
"""

import importlib.metadata as importlib_metadata
import re

import pytest
from typer.testing import CliRunner

from pipecat.cli.main import _enable_hint, app

runner = CliRunner()

# rich emits ANSI color codes when it thinks the output is a terminal (e.g. in CI).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# These assert the *stub* path, which only exists when the official plugins are NOT
# installed. If one is installed its real Typer app is mounted instead, so skip.
_installed = {ep.name for ep in importlib_metadata.entry_points(group="pipecat_cli.extensions")}
pytestmark = pytest.mark.skipif(
    "cloud" in _installed,
    reason="official CLI plugins are installed; stub path not exercised",
)


def _norm(text: str) -> str:
    """Normalize help output so assertions survive rich's ANSI colors, wrapping, and borders."""
    text = _ANSI_RE.sub("", text)
    for ch in "│╭╮╰╯─":
        text = text.replace(ch, " ")
    return " ".join(text.split())


class TestExtensionDiscovery:
    """`pipecat --help` advertises the official plugins even when uninstalled."""

    def test_help_lists_official_plugins_as_stubs(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = _norm(result.output)
        # The official sub-CLI is listed though it isn't installed...
        assert "cloud" in out
        # ...annotated with the package that provides it.
        assert "requires pipecatcloud" in out


class TestEnableHint:
    """Invoking an uninstalled official plugin prints an actionable hint."""

    def test_invoking_uninstalled_plugin_prints_hint_and_exits_1(self):
        result = runner.invoke(app, ["cloud"])
        assert result.exit_code == 1
        # The actionable enable hint, not the bare Click error.
        assert "No such command" not in result.output
        assert "--with pipecatcloud" in result.output

    def test_uninstalled_plugin_swallows_subcommands_and_options(self):
        # `pipecat cloud deploy --region x` must still reach the hint, not error on
        # the unknown `deploy` subcommand / `--region` option.
        result = runner.invoke(app, ["cloud", "deploy", "--region", "x"])
        assert result.exit_code == 1
        assert "No such command" not in result.output
        assert "--with pipecatcloud" in result.output

    def test_enable_hint_shows_both_install_forms(self):
        hint = _enable_hint("cloud", "pipecatcloud")
        assert 'uv tool install "pipecat-ai[cli]" --with pipecatcloud' in hint
        assert "pip install pipecatcloud" in hint
