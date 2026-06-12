#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `pipecat init` — making a project agent-ready (AGENTS.md + CLAUDE.md)."""

from pathlib import Path

from typer.testing import CliRunner

import pipecat.cli
from pipecat.cli.main import app

runner = CliRunner()

AGENT_TEMPLATES = Path(pipecat.cli.__file__).parent / "agent_templates"


class TestInitAgentReady:
    """Behavior of the `pipecat init` file-drop command."""

    def test_writes_both_files(self, tmp_path):
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0, result.output
        agents = tmp_path / "AGENTS.md"
        claude = tmp_path / "CLAUDE.md"
        assert agents.read_text(encoding="utf-8").strip()
        assert "pipecat create" in agents.read_text(encoding="utf-8")
        assert claude.read_text(encoding="utf-8").strip() == "@AGENTS.md"

    def test_prompts_for_directory_when_omitted(self, tmp_path, monkeypatch):
        # Bare `init` prompts; the typed value is just a path → becomes the project folder.
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"], input="mybot\n")
        assert result.exit_code == 0, result.output
        assert (tmp_path / "mybot" / "AGENTS.md").exists()

    def test_prompt_dot_sets_up_current_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"], input=".\n")
        assert result.exit_code == 0, result.output
        assert (tmp_path / "AGENTS.md").exists()

    def test_creates_missing_target_dir(self, tmp_path):
        target = tmp_path / "nested" / "bot"
        result = runner.invoke(app, ["init", str(target)])
        assert result.exit_code == 0, result.output
        assert (target / "AGENTS.md").exists()

    def test_rerun_refreshes_agents_keeps_claude(self, tmp_path):
        runner.invoke(app, ["init", str(tmp_path)])
        (tmp_path / "AGENTS.md").write_text("stale", encoding="utf-8")
        (tmp_path / "CLAUDE.md").write_text("# my own claude config", encoding="utf-8")
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0, result.output
        # AGENTS.md is pipecat-owned → refreshed; CLAUDE.md is the dev's → preserved.
        assert "pipecat create" in (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
        assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "# my own claude config"

    def test_force_overwrites_claude(self, tmp_path):
        runner.invoke(app, ["init", str(tmp_path)])
        (tmp_path / "CLAUDE.md").write_text("# my own claude config", encoding="utf-8")
        result = runner.invoke(app, ["init", str(tmp_path), "--force"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"

    def test_legacy_scaffolder_flags_redirect(self, tmp_path):
        result = runner.invoke(app, ["init", str(tmp_path), "--name", "x", "--bot-type", "web"])
        assert result.exit_code == 1
        assert "pipecat create" in result.output
        # Redirect must not write a half-initialized project.
        assert not (tmp_path / "AGENTS.md").exists()

    def test_quickstart_redirects_without_creating_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "quickstart"])
        assert result.exit_code == 1
        assert "pipecat create quickstart" in result.output
        assert not (tmp_path / "quickstart").exists()

    def test_target_is_a_file_errors(self, tmp_path):
        afile = tmp_path / "afile"
        afile.write_text("x", encoding="utf-8")
        result = runner.invoke(app, ["init", str(afile)])
        assert result.exit_code == 1
        assert "not a directory" in result.output


class TestBundledGuide:
    """The guide must ship and be release-clean (catches packaging + content regressions)."""

    def test_bundled_files_exist(self):
        assert (AGENT_TEMPLATES / "AGENTS.md").is_file()
        assert (AGENT_TEMPLATES / "CLAUDE.md").is_file()

    def test_agents_content_is_release_clean(self):
        text = (AGENT_TEMPLATES / "AGENTS.md").read_text(encoding="utf-8")
        assert text.strip()
        assert "pipecat create" in text
        # No local-checkout paths or stale command name should ever ship.
        assert "/Users/" not in text
        assert "pipecat init" not in text
        assert "TODO(release)" not in text

    def test_claude_is_agents_import(self):
        assert (AGENT_TEMPLATES / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"
