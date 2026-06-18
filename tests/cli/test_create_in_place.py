#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for `pc init <target>` in-place scaffolding (e.g. `pc init .`)."""

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
    """`pc init <dir> --name ...` scaffolds directly into <dir>, no subfolder."""
    result = runner.invoke(app, ["create", str(tmp_path), "--name", "demo", *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "server" / "bot.py").exists()
    assert (tmp_path / "README.md").exists()
    # Not nested under the project name.
    assert not (tmp_path / "demo").exists()


def test_init_in_place_derives_name_from_dir(tmp_path):
    """Without --name, the project name is derived from the target dir basename."""
    target = tmp_path / "my-derived-bot"
    target.mkdir()
    result = runner.invoke(app, ["create", str(target), *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    pyproject = (target / "server" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "my-derived-bot"' in pyproject


def test_init_in_place_preserves_claude_md(tmp_path):
    """The agent-loop case: an existing CLAUDE.md is untouched."""
    (tmp_path / "CLAUDE.md").write_text("# guidance", encoding="utf-8")
    result = runner.invoke(app, ["create", str(tmp_path), "--name", "demo", *SERVICE_FLAGS])
    assert result.exit_code == 0, result.output

    assert (tmp_path / "CLAUDE.md").read_text(encoding="utf-8") == "# guidance"
    assert (tmp_path / "server" / "bot.py").exists()


def test_init_in_place_aborts_on_existing_project(tmp_path):
    """Refuses to clobber a directory that already contains a project."""
    (tmp_path / "server").mkdir()
    result = runner.invoke(app, ["create", str(tmp_path), "--name", "demo", *SERVICE_FLAGS])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_init_target_and_output_are_mutually_exclusive(tmp_path):
    """Passing both a positional target and -o is an error."""
    result = runner.invoke(
        app, ["create", str(tmp_path), "-o", str(tmp_path), "--name", "demo", *SERVICE_FLAGS]
    )
    assert result.exit_code == 1
    assert "not both" in result.output


def test_init_no_positional_still_nests(tmp_path):
    """Non-breaking: without a positional target, the project nests under <name>."""
    result = runner.invoke(
        app, ["create", "--name", "nested-bot", "-o", str(tmp_path), *SERVICE_FLAGS]
    )
    assert result.exit_code == 0, result.output

    assert (tmp_path / "nested-bot" / "server" / "bot.py").exists()


def test_init_quickstart_still_works(tmp_path):
    """Non-breaking: `pc init quickstart -o DIR` still routes to quickstart."""
    result = runner.invoke(app, ["create", "quickstart", "-o", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "pipecat-quickstart" / "server" / "bot.py").exists()
