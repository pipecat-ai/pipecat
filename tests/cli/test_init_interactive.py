#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the interactive scaffold path of `pipecat init` (the wizard).

When the developer chooses "Scaffold a runnable bot now", init runs the interactive
questionary wizard (`ask_project_questions`) and the real `scaffold_interactive` +
`ProjectGenerator`, in-place. These stub only the wizard and the build-method prompt,
exercising that path end to end without needing a TTY.
"""

from typer.testing import CliRunner

import pipecat.cli.commands.init as init_mod
import pipecat.cli.scaffold as scaffold_mod
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


def _force_scaffold_choice(monkeypatch):
    """Force interactive init and make the build-method prompt return 'scaffold'."""
    import questionary

    monkeypatch.setattr(init_mod, "_is_interactive", lambda: True)

    class _Q:
        def ask(self):
            return "scaffold"

    monkeypatch.setattr(questionary, "select", lambda *a, **k: _Q())


def test_interactive_scaffold_in_place_uses_derived_name(tmp_path, monkeypatch):
    # The build-method prompt returns "scaffold", so init runs the wizard in-place,
    # deriving the wizard's default name from the directory (no <name> subfolder).
    captured = {}

    def fake_questions(default_name=None):
        captured["default_name"] = default_name
        return _config(name=default_name or "demo-bot")

    monkeypatch.setattr(scaffold_mod, "ask_project_questions", fake_questions)
    _force_scaffold_choice(monkeypatch)

    target = tmp_path / "my-bot"
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert captured["default_name"] == "my-bot"
    assert (target / "server" / "bot.py").exists()
    assert not (target / "my-bot").exists()
    # init initializes for agent-led dev before scaffolding.
    assert (target / "AGENTS.md").exists()
