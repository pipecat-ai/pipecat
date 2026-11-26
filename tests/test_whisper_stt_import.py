#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for importing the Whisper STT services on Apple Silicon without MLX Whisper."""

import subprocess
import sys

import pytest

pytest.importorskip("faster_whisper")

# Runs in a fresh interpreter so the module is imported from scratch, as on an
# Apple Silicon Mac where only the `whisper` extra (faster-whisper) is installed.
_IMPORT_ON_APPLE_SILICON_WITHOUT_MLX = """
import platform
import sys

platform.system = lambda: "Darwin"
platform.machine = lambda: "arm64"
sys.modules["mlx_whisper"] = None

from pipecat.services.whisper.stt import WhisperSTTService
"""


def test_faster_whisper_service_does_not_need_mlx_whisper():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_ON_APPLE_SILICON_WITHOUT_MLX],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
