#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ``pipecat flows validate``."""

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pipecat.cli.main import app

runner = CliRunner()

FLOW = """
initial_node: start
nodes:
  start:
    role_message: You work for {{ restaurant }}.
    task_messages:
      - role: developer
        content: Greet the caller.
    functions:
      - name: choose_pizza
        transition_to: end
  end:
    task_messages:
      - role: developer
        content: Bye.
    post_actions:
      - type: end_conversation
  orphan:
    task_messages:
      - role: developer
        content: Never reached.
    post_actions:
      - type: end_conversation
"""

TOOLS = '''
from pipecat.flows import FlowManager


async def choose_pizza(flow_manager: FlowManager):
    """User wants pizza."""
    return None, None
'''


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "flow.yaml").write_text(FLOW, encoding="utf-8")
    (tmp_path / "tools.py").write_text(TOOLS, encoding="utf-8")
    return tmp_path


def test_validate_reports_warnings_and_passes(project: Path):
    result = runner.invoke(app, ["flows", "validate", str(project / "flow.yaml")])
    assert result.exit_code == 0, result.output
    assert "warning" in result.output
    assert "orphan" in result.output
    assert result.output.splitlines()[0].endswith(": 0 errors, 1 warning (structure)")
    assert "not checked" not in result.output


def test_strict_fails_on_warnings(project: Path):
    result = runner.invoke(app, ["flows", "validate", str(project / "flow.yaml"), "--strict"])
    assert result.exit_code == 1


def test_validate_with_tools_file(project: Path):
    result = runner.invoke(
        app,
        ["flows", "validate", str(project / "flow.yaml"), "--tools", str(project / "tools.py")],
    )
    assert result.exit_code == 0, result.output
    assert "(structure, tools)" in result.output.splitlines()[0]


def test_missing_tool_is_an_error(project: Path):
    (project / "tools.py").write_text("x = 1\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["flows", "validate", str(project / "flow.yaml"), "--tools", str(project / "tools.py")],
    )
    assert result.exit_code == 1
    assert "references tool 'choose_pizza'" in result.output
    assert result.output.splitlines()[0].endswith(": 1 error, 1 warning (structure, tools)")


def test_variables_checked_when_given(project: Path):
    result = runner.invoke(
        app, ["flows", "validate", str(project / "flow.yaml"), "--var", "other=1"]
    )
    assert result.exit_code == 1
    assert "variable 'restaurant' has no value" in result.output

    result = runner.invoke(
        app, ["flows", "validate", str(project / "flow.yaml"), "--var", "restaurant=Luigi"]
    )
    assert result.exit_code == 0, result.output
    assert "(structure, variables)" in result.output.splitlines()[0]


def test_bad_variable_syntax(project: Path):
    result = runner.invoke(app, ["flows", "validate", str(project / "flow.yaml"), "--var", "nope"])
    assert result.exit_code != 0
    assert "NAME=VALUE" in result.output


def test_json_output(project: Path):
    result = runner.invoke(app, ["flows", "validate", str(project / "flow.yaml"), "--json"])
    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["ok"] is True
    assert report["tools"] == ["choose_pizza"]
    assert report["variables"] == ["restaurant"]
    assert [i["code"] for i in report["issues"]] == ["unreachable_node"]


def test_schema_error_exits_one(project: Path):
    (project / "flow.yaml").write_text("initial_node: nope\nnodes: {}\n", encoding="utf-8")
    result = runner.invoke(app, ["flows", "validate", str(project / "flow.yaml")])
    assert result.exit_code == 1
    assert "error" in result.output


def test_stdin(project: Path):
    result = runner.invoke(app, ["flows", "validate", "-"], input=FLOW)
    assert result.exit_code == 0, result.output
    assert result.output.startswith("stdin: ")


def test_tools_module_by_name(project: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(project)
    monkeypatch.delitem(sys.modules, "tools", raising=False)
    result = runner.invoke(app, ["flows", "validate", "flow.yaml", "--tools", "tools"])
    assert result.exit_code == 0, result.output


def test_fully_checked_run(project: Path):
    result = runner.invoke(
        app,
        [
            "flows",
            "validate",
            str(project / "flow.yaml"),
            "--tools",
            str(project / "tools.py"),
            "--var",
            "restaurant=Luigi",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].endswith("(structure, tools, variables)")


def test_flows_help_lists_validate():
    result = runner.invoke(app, ["flows", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.output
    assert "schema" not in result.output
