#
# Copyright (c) 2025-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pytest configuration for the CLI test suite.

The CLI lives behind the optional ``pipecat-ai[cli]`` extra (typer, rich,
jinja2, questionary, ruff). When that extra is not installed, skip collecting
``tests/cli`` entirely instead of failing collection with ImportErrors, so a
lean ``uv run pytest`` still works. CI installs the ``cli`` extra so these run.
"""

import importlib.util

# Skip the whole directory if the CLI dependencies are not available.
if importlib.util.find_spec("questionary") is None:
    collect_ignore_glob = ["*"]
