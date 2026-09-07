#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the `pipecat init quickstart` command."""

from typer.testing import CliRunner

from pipecat.cli.main import app

runner = CliRunner()


class TestQuickstart:
    """Tests for the quickstart preset (scaffolds in-place into ./pipecat-quickstart)."""

    def test_quickstart_generates_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "quickstart"])
        assert result.exit_code == 0, result.output

        project_dir = tmp_path / "pipecat-quickstart"
        assert (project_dir / "server" / "bot.py").exists()
        assert (project_dir / "server" / "pyproject.toml").exists()
        assert (project_dir / ".gitignore").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "server" / "Dockerfile").exists()

    def test_quickstart_fails_if_project_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pipecat-quickstart" / "server").mkdir(parents=True)
        result = runner.invoke(app, ["init", "quickstart"])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_quickstart_output_contains_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "quickstart"])
        assert result.exit_code == 0, result.output
        assert "SmallWebRTC" in result.output
        assert "Daily" in result.output
        assert "Deepgram" in result.output
        assert "OpenAI" in result.output
        assert "Cartesia" in result.output

    def test_quickstart_warns_when_scaffold_flags_passed(self, tmp_path, monkeypatch):
        # quickstart is a fixed preset; combining it with scaffold flags should warn rather
        # than silently ignore them — but still scaffold the canned (web) bot.
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "quickstart", "--bot-type", "telephony"])
        assert result.exit_code == 0, result.output
        assert "fixed preset" in result.output
        assert (tmp_path / "pipecat-quickstart" / "server" / "bot.py").exists()
