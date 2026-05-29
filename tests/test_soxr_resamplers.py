#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import pytest

from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
from pipecat.audio.resamplers.soxr_resampler import SOXRAudioResampler
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler


@pytest.mark.asyncio
async def test_soxr_audio_resampler_uses_configured_quality(monkeypatch):
    calls = {}

    def resample(audio_data, in_rate, out_rate, quality):
        calls["quality"] = quality
        calls["in_rate"] = in_rate
        calls["out_rate"] = out_rate
        return audio_data

    monkeypatch.setattr("pipecat.audio.resamplers.soxr_resampler.soxr.resample", resample)

    audio = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    result = await SOXRAudioResampler(quality="HQ").resample(audio, 8000, 16000)

    assert result == audio
    assert calls == {"quality": "HQ", "in_rate": 8000, "out_rate": 16000}


@pytest.mark.asyncio
async def test_soxr_audio_resampler_defaults_to_vhq(monkeypatch):
    calls = {}

    def resample(audio_data, in_rate, out_rate, quality):
        calls["quality"] = quality
        return audio_data

    monkeypatch.setattr("pipecat.audio.resamplers.soxr_resampler.soxr.resample", resample)

    audio = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    await SOXRAudioResampler().resample(audio, 8000, 16000)

    assert calls["quality"] == "VHQ"


@pytest.mark.asyncio
async def test_soxr_stream_audio_resampler_uses_configured_quality(monkeypatch):
    calls = {}

    class ResampleStream:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def resample_chunk(self, audio_data):
            return audio_data

        def clear(self):
            pass

    monkeypatch.setattr(
        "pipecat.audio.resamplers.soxr_stream_resampler.soxr.ResampleStream", ResampleStream
    )

    audio = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    result = await SOXRStreamAudioResampler(quality="QQ").resample(audio, 8000, 16000)

    assert result == audio
    assert calls == {
        "in_rate": 8000,
        "out_rate": 16000,
        "num_channels": 1,
        "quality": "QQ",
        "dtype": "int16",
    }


@pytest.mark.asyncio
async def test_soxr_stream_audio_resampler_defaults_to_vhq(monkeypatch):
    calls = {}

    class ResampleStream:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def resample_chunk(self, audio_data):
            return audio_data

        def clear(self):
            pass

    monkeypatch.setattr(
        "pipecat.audio.resamplers.soxr_stream_resampler.soxr.ResampleStream", ResampleStream
    )

    audio = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    await SOXRStreamAudioResampler().resample(audio, 8000, 16000)

    assert calls["quality"] == "VHQ"


@pytest.mark.asyncio
async def test_rnnoise_default_resampler_quality_is_forwarded(monkeypatch):
    qualities = []

    class RNNoise:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

    class Resampler:
        def __init__(self, quality):
            qualities.append(quality)

    monkeypatch.setattr("pipecat.audio.filters.rnnoise_filter.RNNoise", RNNoise)
    monkeypatch.setattr(
        "pipecat.audio.resamplers.soxr_stream_resampler.SOXRStreamAudioResampler", Resampler
    )

    rnnoise_filter = RNNoiseFilter()
    await rnnoise_filter.start(16000)

    assert qualities == ["QQ", "QQ"]
