#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.utils import startup


def test_setup_file_paths_uses_platform_path_separator(monkeypatch, tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    monkeypatch.setattr(startup.os, "pathsep", ";")
    monkeypatch.setenv("PIPECAT_SETUP_FILES", f"{first};{second}")

    assert startup._setup_file_paths() == [first.resolve(), second.resolve()]
