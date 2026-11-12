#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SincAudioResampler."""

import math

import numpy as np
import pytest

from pipecat.audio.resamplers.sinc_resampler import SincAudioResampler
from pipecat.audio.utils import create_file_resampler

# Down and up, by small ratios and by the large ones 44.1 kHz needs.
RATE_PAIRS = [
    (24000, 8000),
    (24000, 16000),
    (48000, 16000),
    (44100, 16000),
    (8000, 16000),
    (16000, 24000),
    (16000, 44100),
]


def tone(freq: float, rate: int, samples: int) -> np.ndarray:
    return np.sin(2 * np.pi * freq * np.arange(samples) / rate) * 16000


def clip(freq: float, rate: int, samples: int) -> bytes:
    return np.rint(tone(freq, rate, samples)).astype(np.int16).tobytes()


async def resample(audio: bytes, in_rate: int, out_rate: int) -> np.ndarray:
    out = await SincAudioResampler().resample(audio, in_rate, out_rate)
    return np.frombuffer(out, dtype=np.int16)


@pytest.mark.asyncio
@pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
async def test_output_keeps_the_clip_length(in_rate, out_rate):
    samples = in_rate // 10 + 1
    out = await resample(clip(440, in_rate, samples), in_rate, out_rate)
    assert len(out) == math.floor(samples * out_rate / in_rate + 0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
async def test_output_lines_up_with_the_input(in_rate, out_rate):
    out = await resample(clip(1000, in_rate, in_rate), in_rate, out_rate)
    ideal = tone(1000, out_rate, len(out))
    middle = slice(len(out) // 4, 3 * len(out) // 4)
    assert np.max(np.abs(out[middle] - ideal[middle])) <= 3


@pytest.mark.asyncio
async def test_same_rate_returns_the_audio():
    audio = clip(440, 16000, 1600)
    assert await SincAudioResampler().resample(audio, 16000, 16000) is audio


@pytest.mark.asyncio
async def test_clips_do_not_affect_each_other():
    resampler = SincAudioResampler()
    second = clip(880, 24000, 4800)
    await resampler.resample(clip(440, 24000, 4800), 24000, 16000)

    assert await resampler.resample(second, 24000, 16000) == await SincAudioResampler().resample(
        second, 24000, 16000
    )
    assert await resampler.flush() == b""


@pytest.mark.asyncio
async def test_rates_can_change_between_clips():
    resampler = SincAudioResampler()
    audio = clip(440, 48000, 4800)
    await resampler.resample(audio, 24000, 16000)

    out = await resampler.resample(audio, 48000, 16000)
    assert out == await SincAudioResampler().resample(audio, 48000, 16000)
    assert len(out) == 2 * 1600


def test_create_file_resampler_returns_a_sinc_resampler():
    assert isinstance(create_file_resampler(), SincAudioResampler)
