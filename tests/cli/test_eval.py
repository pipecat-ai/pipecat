#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pathlib import Path

from pipecat.cli.commands.eval import _build_scenario_runs


def test_multi_scenario_file_expands_fixed_url_runs(tmp_path: Path):
    collection = tmp_path / "collection.yaml"
    collection.write_text(
        "scenarios:\n  - name: first\n    turns: []\n  - name: second\n    turns: []\n"
    )

    runs = _build_scenario_runs([collection], "ws://localhost:7860")

    assert [run.scenario for run in runs] == ["first", "second"]
    assert all(run.scenario_data is not None for run in runs)


def test_malformed_scenario_file_becomes_failed_run(tmp_path: Path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("name: malformed\nturns: [\n")

    runs = _build_scenario_runs([malformed], "ws://localhost:7860")

    assert len(runs) == 1
    assert runs[0].scenario == "malformed"
    assert runs[0].status == "done"
    assert runs[0].error is not None
    assert runs[0].error.startswith("failed to load:")
