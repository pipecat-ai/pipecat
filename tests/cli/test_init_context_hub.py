#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Context Hub setup `pipecat init` performs.

The hub ships with the `cli` extra, so there is nothing to install — `init` registers
the MCP server and, when there is no index yet, offers to build one. Two properties are
easy to break without noticing: `init quickstart` is a deliberate short circuit that
must never reach this, and the index question must not return once an index exists.
"""

import io
import subprocess

from typer.testing import CliRunner

from pipecat.cli.main import app

runner = CliRunner()


def _no_hub_setup(monkeypatch) -> list[bool]:
    """Record whether the hub setup ran, without letting it touch anything real."""
    calls: list[bool] = []
    monkeypatch.setattr("pipecat.cli.commands.init._setup_context_hub", lambda: calls.append(True))
    return calls


class TestQuickstartShortCircuit:
    """`pipecat init quickstart` is the fast path and must stay fast.

    It scaffolds a runnable bot from a fixed preset with no questions. Registering an
    MCP server or offering a multi-minute index build there would defeat the point.
    """

    def test_quickstart_never_sets_up_the_hub(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = _no_hub_setup(monkeypatch)
        result = runner.invoke(app, ["init", "quickstart"])
        assert result.exit_code == 0, result.output
        assert calls == []

    def test_quickstart_still_writes_the_agent_guide(self, tmp_path, monkeypatch):
        """The carve-out is about setup cost, not about skipping the guide."""
        monkeypatch.chdir(tmp_path)
        _no_hub_setup(monkeypatch)
        runner.invoke(app, ["init", "quickstart"])
        assert (tmp_path / "pipecat-quickstart" / "AGENTS.md").exists()


class TestCodingAgentPath:
    def test_sets_up_the_hub(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = _no_hub_setup(monkeypatch)
        result = runner.invoke(app, ["init", "."])
        assert result.exit_code == 0, result.output
        assert calls == [True]

    def test_no_context_hub_flag_skips_it(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = _no_hub_setup(monkeypatch)
        result = runner.invoke(app, ["init", ".", "--no-context-hub"])
        assert result.exit_code == 0, result.output
        assert calls == []


class TestIndexPrompt:
    """The index build is the only expensive step, so it is the only question."""

    def _index(self, monkeypatch, metadata):
        monkeypatch.setattr(
            "pipecat.cli.hub_status.read_hub_metadata", lambda: metadata, raising=False
        )

    def test_no_question_once_an_index_exists(self, monkeypatch, capsys):
        """An index cannot exist unless someone ran refresh, so this is the 'been here' flag."""
        from pipecat.cli.commands.init import _offer_context_hub_index

        self._index(monkeypatch, {"last_refresh_at": "2026-08-03T00:00:00+00:00"})
        monkeypatch.setattr("pipecat.cli.commands.init._is_interactive", lambda: True)
        _offer_context_hub_index()
        assert capsys.readouterr().out == ""

    def test_non_interactive_prints_the_command_instead_of_asking(self, monkeypatch, capsys):
        from pipecat.cli.commands.init import _offer_context_hub_index

        self._index(monkeypatch, None)
        monkeypatch.setattr("pipecat.cli.commands.init._is_interactive", lambda: False)
        _offer_context_hub_index()
        assert "pipecat context-hub refresh" in capsys.readouterr().out

    def test_an_index_without_a_completed_refresh_still_prompts(self, monkeypatch, capsys):
        """Metadata can exist before any refresh finished; that is not a built index."""
        from pipecat.cli.commands.init import _offer_context_hub_index

        self._index(monkeypatch, {"metadata_contract_version": "1"})
        monkeypatch.setattr("pipecat.cli.commands.init._is_interactive", lambda: False)
        _offer_context_hub_index()
        assert "pipecat context-hub refresh" in capsys.readouterr().out


class TestIndexBuild:
    """The build shows one line of the hub's stream at a time, so failure must resurface."""

    def _refresh(self, monkeypatch, returncode: int, stderr: str = "") -> list[list[str]]:
        commands: list[list[str]] = []

        class _Process:
            def __init__(self, args, **kwargs):
                commands.append(args)
                self.returncode = returncode
                if stderr:
                    kwargs["stderr"].write(stderr)
                    kwargs["stderr"].flush()

            def poll(self):
                return self.returncode

        monkeypatch.setattr("subprocess.Popen", _Process)
        return commands

    def test_asks_the_hub_for_the_stream_it_echoes(self, monkeypatch):
        """The status line needs INFO; the hub's default is not ours to depend on."""
        from pipecat.cli.commands.init import _build_context_hub_index

        commands = self._refresh(monkeypatch, 0)
        assert _build_context_hub_index() is True
        assert commands[0][-3:] == ["--log-level", "INFO", "refresh"]

    def test_failure_shows_the_error_and_how_to_retry(self, monkeypatch, capsys):
        from pipecat.cli.commands.init import _build_context_hub_index

        self._refresh(monkeypatch, 1, stderr="ERROR could not clone pipecat-ai/pipecat\n")
        assert _build_context_hub_index() is False
        out = capsys.readouterr().out
        assert "could not clone" in out
        assert "pipecat context-hub refresh" in out

    def test_failure_output_is_not_read_as_markup(self, monkeypatch, capsys):
        """Hub errors quote bracketed values, which Rich would otherwise eat."""
        from pipecat.cli.commands.init import _build_context_hub_index

        self._refresh(monkeypatch, 1, stderr="ERROR bad source ['pipecat']\n")
        _build_context_hub_index()
        assert "['pipecat']" in capsys.readouterr().out


class TestStatusLine:
    """What the spinner echoes while the hub works."""

    def test_shows_the_message_without_the_log_furniture(self):
        from pipecat.cli.commands.init import _latest_log_line

        line = (
            "2026-08-03 18:39:34,739 pipecat_context_hub.cli INFO "
            "GitHub ingest (pipecat-ai/pipecat): upserted=1421\n"
        )
        assert _latest_log_line(io.StringIO(line), "")[1].startswith("GitHub ingest")

    def test_an_unfamiliar_line_is_shown_whole(self):
        """A change to the hub's log format must cost legibility, not the status line."""
        from pipecat.cli.commands.init import _latest_log_line

        assert _latest_log_line(io.StringIO("cloning pipecat\n"), "")[1] == "cloning pipecat"

    def test_a_half_written_line_is_carried_to_the_next_read(self):
        from pipecat.cli.commands.init import _latest_log_line

        carried, detail = _latest_log_line(io.StringIO("done\nhalf"), "")
        assert (carried, detail) == ("half", "done")
        assert _latest_log_line(io.StringIO(" written\n"), carried)[1] == "half written"

    def test_long_lines_are_trimmed_to_the_terminal(self):
        """A wrapped status line leaves its overflow on screen when the spinner redraws."""
        from pipecat.cli.commands.init import _index_status, console

        status = _index_status("x" * 500)
        assert status.endswith("…") and len(status) <= console.width

    def test_bracketed_values_are_not_read_as_markup(self):
        from pipecat.cli.commands.init import _index_status

        assert r"\[pipecat]" in _index_status("ingesting [pipecat]")


class TestRegistrationOutcomes:
    """`install` output is captured, so its exit code is all init has to go on.

    Three outcomes look identical from the outside, and reporting only the first leaves
    an editor user believing a registration happened that did not.
    """

    def _install(self, monkeypatch, returncode: int, stderr: str = "") -> None:
        monkeypatch.setattr(
            "pipecat.cli.commands.init._register_context_hub",
            lambda: subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr),
        )
        monkeypatch.setattr("pipecat.cli.commands.init._offer_context_hub_index", lambda: None)

    def test_a_configured_client_is_reported(self, monkeypatch, capsys):
        from pipecat.cli.commands.init import _setup_context_hub

        self._install(monkeypatch, 0)
        _setup_context_hub()
        assert "Registered the Context Hub MCP server" in capsys.readouterr().out

    def test_an_editor_user_is_told_how_to_finish(self, monkeypatch, capsys):
        """Cursor, VS Code, and Zed are configured by hand, so they need the config block."""
        from pipecat.cli.commands.init import _setup_context_hub

        self._install(monkeypatch, 3)
        _setup_context_hub()
        out = capsys.readouterr().out
        assert "pipecat context-hub install" in out
        assert "Registered" not in out

    def test_a_rejected_registration_is_surfaced(self, monkeypatch, capsys):
        """A client CLI that refuses must not be swallowed into silence."""
        from pipecat.cli.commands.init import _setup_context_hub

        self._install(monkeypatch, 1, stderr="claude exited 1: boom")
        _setup_context_hub()
        out = capsys.readouterr().out
        assert "Could not register" in out
        assert "boom" in out


class TestFailureIsNeverFatal:
    """Setting up a side tool must not fail the command that scaffolded the project."""

    def test_a_broken_hub_does_not_fail_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def _explode():
            raise RuntimeError("hub is broken")

        monkeypatch.setattr("pipecat.cli.commands.init._register_context_hub", _explode)
        result = runner.invoke(app, ["init", "."])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "AGENTS.md").exists()
        assert "pipecat context-hub install" in result.output
