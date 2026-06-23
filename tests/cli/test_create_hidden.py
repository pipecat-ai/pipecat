#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that `pipecat create` is hidden from `--help` yet stays fully functional.

`pipecat init` is the single advertised entry point. `create` is the underlying
scaffolder that coding agents and automation call non-interactively, so it must keep
working — it's just no longer listed in `pipecat --help`.
"""

import re

from typer.testing import CliRunner

from pipecat.cli.main import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _norm(text: str) -> str:
    """Normalize help output so assertions survive rich's ANSI colors and borders."""
    text = _ANSI_RE.sub("", text)
    for ch in "│╭╮╰╯─":
        text = text.replace(ch, " ")
    return " ".join(text.split())


class TestCreateHidden:
    """`create` is hidden from the top-level help but remains invocable."""

    def test_create_not_listed_in_top_level_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        out = _norm(result.output)
        # `init` is the advertised starting point...
        assert "init" in out
        # ...and `create` is not listed as a command.
        assert "create" not in out

    def test_create_help_still_works(self):
        result = runner.invoke(app, ["create", "--help"])
        # The command is hidden, not removed: `pipecat create --help` still resolves.
        assert result.exit_code == 0
        out = _norm(result.output)
        assert "create" in out
