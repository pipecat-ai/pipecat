#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the pc init quickstart command."""

from typer.testing import CliRunner

from pipecat.cli.main import app

runner = CliRunner()


class TestQuickstart:
    """Tests for the quickstart subcommand."""

    def test_quickstart_generates_project(self, tmp_path):
        result = runner.invoke(app, ["create", "quickstart", "-o", str(tmp_path)])
        assert result.exit_code == 0, result.output

        project_dir = tmp_path / "pipecat-quickstart"
        assert (project_dir / "server" / "bot.py").exists()
        assert (project_dir / "server" / "pyproject.toml").exists()
        assert (project_dir / ".gitignore").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "server" / "Dockerfile").exists()

    def test_quickstart_fails_if_directory_exists(self, tmp_path):
        (tmp_path / "pipecat-quickstart").mkdir()
        result = runner.invoke(app, ["create", "quickstart", "-o", str(tmp_path)])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_quickstart_output_contains_defaults(self, tmp_path):
        result = runner.invoke(app, ["create", "quickstart", "-o", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "SmallWebRTC" in result.output
        assert "Daily" in result.output
        assert "Deepgram" in result.output
        assert "OpenAI" in result.output
        assert "Cartesia" in result.output
