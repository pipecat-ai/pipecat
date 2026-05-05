#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for integer-ratio resampling."""

import numpy as np
import pytest

from pipecat.transports.sip.codecs import resample_down, resample_up


class TestResampleUp:
    def test_8k_to_16k_length(self):
        """160 samples at 8kHz becomes 320 at 16kHz."""
        samples = np.zeros(160, dtype=np.int16)
        result = resample_up(samples, factor=2)
        assert len(result) == 320
        assert result.dtype == np.int16

    def test_empty_input(self):
        """Empty array returns empty array."""
        result = resample_up(np.array([], dtype=np.int16), factor=2)
        assert len(result) == 0

    def test_preserves_dc(self):
        """Constant signal stays constant after upsampling."""
        samples = np.full(160, 1000, dtype=np.int16)
        result = resample_up(samples, factor=2)
        assert np.all(result == 1000)

    def test_sine_wave_roundtrip(self):
        """Upsample then downsample of a sine wave preserves shape."""
        t = np.arange(160, dtype=np.float64)
        sine = (10000 * np.sin(2 * np.pi * t / 160)).astype(np.int16)
        up = resample_up(sine, factor=2)
        down = resample_down(up, factor=2)
        max_err = np.max(np.abs(sine.astype(np.int32) - down[:160].astype(np.int32)))
        assert max_err < 2000

    def test_generic_factor(self):
        """Non-standard factor works."""
        samples = np.arange(80, dtype=np.int16)
        result = resample_up(samples, factor=3)
        assert len(result) == 240


class TestResampleDown:
    def test_16k_to_8k_length(self):
        """320 samples at 16kHz becomes 160 at 8kHz."""
        samples = np.zeros(320, dtype=np.int16)
        result = resample_down(samples, factor=2)
        assert len(result) == 160
        assert result.dtype == np.int16

    def test_24k_to_8k_length(self):
        """480 samples at 24kHz becomes 160 at 8kHz."""
        samples = np.zeros(480, dtype=np.int16)
        result = resample_down(samples, factor=3)
        assert len(result) == 160

    def test_empty_input(self):
        """Empty array returns empty array."""
        result = resample_down(np.array([], dtype=np.int16), factor=2)
        assert len(result) == 0

    def test_preserves_dc(self):
        """Constant signal stays constant after downsampling."""
        samples = np.full(320, 5000, dtype=np.int16)
        result = resample_down(samples, factor=2)
        assert np.allclose(result, 5000, atol=1)
