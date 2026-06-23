#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the interactive `pc create` path (the wizard).

`create` with no `--name`/`--config` runs an interactive questionary wizard. These
tests stub only that wizard (`ask_project_questions`) and let the real
`scaffold_interactive` + `ProjectGenerator` run, so they exercise the interactive
dispatch and in-place handling end to end without needing a TTY.
"""

from typer.testing import CliRunner

import pipecat.cli.commands.create as create_mod
from pipecat.cli.main import app
from pipecat.cli.prompts import ProjectConfig

runner = CliRunner()


def _config(name="demo-bot"):
    """A minimal known-good web/cascade config, as the wizard would return."""
    return ProjectConfig(
        project_name=name,
        bot_type="web",
        transports=["smallwebrtc"],
        mode="cascade",
        stt_service="deepgram_stt",
        llm_service="openai_llm",
        tts_service="cartesia_tts",
        deploy_to_cloud=False,
    )


def test_interactive_generates_project(tmp_path, monkeypatch):
    # No --name/--config → interactive: the wizard runs, then the project is generated.
    monkeypatch.setattr(create_mod, "ask_project_questions", lambda default_name=None: _config())
    result = runner.invoke(app, ["create", "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    # Nested under the wizard-supplied name (no positional target).
    assert (tmp_path / "demo-bot" / "server" / "bot.py").exists()


def test_interactive_in_place_uses_derived_name(tmp_path, monkeypatch):
    # A positional target goes interactive in-place, deriving the wizard's default name
    # from the directory and scaffolding directly into it (no <name> subfolder).
    captured = {}

    def fake_questions(default_name=None):
        captured["default_name"] = default_name
        return _config(name=default_name or "demo-bot")

    monkeypatch.setattr(create_mod, "ask_project_questions", fake_questions)
    target = tmp_path / "my-bot"
    result = runner.invoke(app, ["create", str(target)])
    assert result.exit_code == 0, result.output
    assert captured["default_name"] == "my-bot"
    assert (target / "server" / "bot.py").exists()
    assert not (target / "my-bot").exists()
