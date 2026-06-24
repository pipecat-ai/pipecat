#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `pipecat init <target>` in-place scaffolding (e.g. `pipecat init .`)."""

from typer.testing import CliRunner

from pipecat.cli.main import app

runner = CliRunner()

# A complete non-interactive web/cascade invocation, minus the destination.
SERVICE_FLAGS = [
    "--bot-type",
    "web",
    "-t",
    "daily",
    "-m",
    "cascade",
    "--stt",
    "deepgram_stt",
    "--llm",
    "openai_llm",
    "--tts",
    "cartesia_tts",
]


def test_init_in_place_writes_into_target_dir(tmp_path):
    """`pipecat init <dir> ...` scaffolds directly into <dir>, no subfolder."""
    result = runner.invoke(app, ["init", str(tmp_path), "--name", "demo", *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "server" / "bot.py").exists()
    assert (tmp_path / "README.md").exists()
    # Not nested under the project name.
    assert not (tmp_path / "demo").exists()


def test_init_in_place_also_writes_agent_guide(tmp_path):
    """Non-interactive `init` still initializes for agent-led dev: AGENTS.md + CLAUDE.md."""
    result = runner.invoke(app, ["init", str(tmp_path), *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "server" / "bot.py").exists()
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"
    # The scaffold path skips GETTING_STARTED.md — the README is the start-here.
    assert not (tmp_path / "GETTING_STARTED.md").exists()


def test_init_in_place_derives_name_from_dir(tmp_path):
    """Without --name, the project name is derived from the target dir basename."""
    target = tmp_path / "my-derived-bot"
    target.mkdir()
    result = runner.invoke(app, ["init", str(target), *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    pyproject = (target / "server" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "my-derived-bot"' in pyproject


def test_init_in_place_preserves_claude_md(tmp_path):
    """The agent-loop case: an existing CLAUDE.md is untouched."""
    (tmp_path / "CLAUDE.md").write_text("# guidance", encoding="utf-8")
    result = runner.invoke(app, ["init", str(tmp_path), "--name", "demo", *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "# guidance"
    assert (tmp_path / "server" / "bot.py").exists()


def test_init_in_place_over_agent_guide(tmp_path):
    """The full agent loop: scaffold in-place into a dir already holding the guide.

    Mirrors what `pipecat init` leaves behind before the agent re-runs it to scaffold.
    Preserve-by-default: the existing guide files are all kept as-is, and the bot is
    scaffolded alongside them (refresh the guide explicitly with --overwrite-guide).
    """
    (tmp_path / "AGENTS.md").write_text("# my guide", encoding="utf-8")
    (tmp_path / "GETTING_STARTED.md").write_text("# dev", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md", encoding="utf-8")
    result = runner.invoke(app, ["init", str(tmp_path), "--name", "demo", *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "server" / "bot.py").exists()
    # Existing guide files survive untouched.
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "# my guide"
    assert (tmp_path / "GETTING_STARTED.md").read_text(encoding="utf-8") == "# dev"
    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md"


def test_init_in_place_aborts_on_existing_project(tmp_path):
    """Refuses to clobber a directory that already contains a project."""
    (tmp_path / "server").mkdir()
    result = runner.invoke(app, ["init", str(tmp_path), "--name", "demo", *SERVICE_FLAGS])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_invalid_scaffold_flags_write_nothing(tmp_path):
    """An incomplete scaffold invocation fails before writing — no half-initialized dir.

    The config is validated before the guide is written, so a bad invocation leaves the
    directory untouched rather than dropping AGENTS.md/CLAUDE.md and then erroring.
    """
    # --bot-type with no transport/mode/services is invalid.
    result = runner.invoke(app, ["init", str(tmp_path), "--bot-type", "web"])
    assert result.exit_code == 1
    assert "validation failed" in result.output.lower()
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "CLAUDE.md").exists()
    assert not (tmp_path / "server").exists()
