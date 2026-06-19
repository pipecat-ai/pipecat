#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Silero VAD."""

import numpy as np
import pytest

from pipecat.audio.vad.silero import SileroOnnxModel


def test_validate_input_rejects_high_dimensional_numpy_audio():
    model = SileroOnnxModel.__new__(SileroOnnxModel)
    model.sample_rates = [8000, 16000]

    audio = np.zeros((1, 1, 512), dtype=np.float32)

    with pytest.raises(ValueError, match="Too many dimensions for input audio chunk 3"):
        model._validate_input(audio, 16000)
