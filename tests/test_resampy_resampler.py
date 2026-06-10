#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the deprecated resampy resampler."""

import pytest

from pipecat.audio.resamplers.resampy_resampler import ResampyResampler


def test_resampy_resampler_emits_deprecation_warning():
    """Test that instantiating ResampyResampler emits a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="ResampyResampler is deprecated"):
        ResampyResampler()
