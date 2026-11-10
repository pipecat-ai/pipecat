#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for InputAudioResampler."""

import numpy as np
import pytest

from pipecat.audio.resamplers.input_audio_resampler import InputAudioResampler
from pipecat.audio.utils import create_stream_resampler


def _tone(rate: int, seconds: float = 0.1) -> bytes:
    t = np.arange(int(rate * seconds)) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16).tobytes()


async def _fresh(audio: bytes, in_rate: int, out_rate: int) -> bytes:
    return await create_stream_resampler().resample(audio, in_rate, out_rate)


@pytest.mark.asyncio
async def test_audio_at_the_output_rate_is_passed_through():
    audio = _tone(16000)

    assert await InputAudioResampler().resample(audio, 16000, 16000) is audio


@pytest.mark.asyncio
async def test_audio_is_resampled_as_one_stream():
    resampler = InputAudioResampler()
    first, second = _tone(24000), _tone(24000)

    out = await resampler.resample(first, 24000, 16000)
    out += await resampler.resample(second, 24000, 16000)

    assert out == await _fresh(first + second, 24000, 16000)


@pytest.mark.asyncio
async def test_a_rate_change_starts_a_new_stream():
    resampler = InputAudioResampler()
    await resampler.resample(_tone(24000), 24000, 16000)
    audio = _tone(48000)

    assert await resampler.resample(audio, 48000, 16000) == await _fresh(audio, 48000, 16000)


@pytest.mark.asyncio
async def test_reset_starts_a_new_stream():
    resampler = InputAudioResampler()
    await resampler.resample(_tone(24000), 24000, 16000)
    audio = _tone(24000)

    await resampler.reset()

    assert await resampler.resample(audio, 24000, 16000) == await _fresh(audio, 24000, 16000)


@pytest.mark.asyncio
async def test_flush_emits_the_delayed_audio():
    resampler = InputAudioResampler()
    assert await resampler.flush() == b""

    await resampler.resample(_tone(24000), 24000, 16000)

    assert await resampler.flush()
