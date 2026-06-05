#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The import-time version banner is suppressed for the pipecat/pc CLI."""

import sys

import pipecat


def test_banner_suppressed_for_cli_programs(monkeypatch):
    for prog in ("pipecat", "pc", "/usr/local/bin/pipecat", "pipecat.exe"):
        monkeypatch.setattr(sys, "argv", [prog])
        assert pipecat._should_log_version_banner() is False


def test_banner_shown_for_bots_and_library_use(monkeypatch):
    for prog in ("bot.py", "python", "-c", "pytest", "/path/to/myapp"):
        monkeypatch.setattr(sys, "argv", [prog])
        assert pipecat._should_log_version_banner() is True
