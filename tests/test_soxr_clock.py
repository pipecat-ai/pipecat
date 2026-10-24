#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from types import SimpleNamespace

import numpy as np
import pytest

from pipecat.audio.resamplers import soxr_stream_resampler
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler


@pytest.mark.asyncio
@pytest.mark.parametrize("wall_jump,elapsed", [(3600, 0.05), (-3600, 0.3)])
async def test_resampler_idle_reset_uses_elapsed_time(monkeypatch, wall_jump, elapsed):
    clock = {"wall": 10000.0, "elapsed": 10.0}
    monkeypatch.setattr(
        soxr_stream_resampler,
        "time",
        SimpleNamespace(time=lambda: clock["wall"], monotonic=lambda: clock["elapsed"]),
    )
    audio = (np.sin(np.arange(640) * 0.1) * 10000).astype(np.int16).tobytes()
    actual = SOXRStreamAudioResampler(clear_after_secs=0.2)
    reference = SOXRStreamAudioResampler(clear_after_secs=None)
    first = await actual.resample(audio, 16000, 8000)
    expected_first = await reference.resample(audio, 16000, 8000)
    assert first == expected_first
    clock["wall"] += wall_jump
    clock["elapsed"] += elapsed
    if elapsed > 0.2:
        await reference.reset()
    second = await actual.resample(audio, 16000, 8000)
    expected_second = await reference.resample(audio, 16000, 8000)
    assert second + await actual.flush() == expected_second + await reference.flush()
