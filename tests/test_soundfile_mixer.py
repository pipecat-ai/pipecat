#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer  # noqa: E402

MIXER_RATE = 16000


def _write_wav(path, sample_rate: int, duration_s: float, channels: int = 1):
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    if channels > 1:
        tone = np.column_stack([tone] * channels)
    sf.write(str(path), tone, sample_rate)
    return tone


def _make_mixer(path):
    return SoundfileMixer(sound_files={"test": str(path)}, default_sound="test")


@pytest.mark.asyncio
async def test_matching_rate_file_loaded_unchanged(tmp_path):
    file_path = tmp_path / "match.wav"
    tone = _write_wav(file_path, MIXER_RATE, 0.5)

    mixer = _make_mixer(file_path)
    await mixer.start(MIXER_RATE)

    assert "test" in mixer._sounds
    assert np.array_equal(mixer._sounds["test"], tone)


@pytest.mark.asyncio
async def test_off_rate_mono_file_is_resampled(tmp_path):
    file_path = tmp_path / "44k.wav"
    _write_wav(file_path, 44100, 0.5)

    mixer = _make_mixer(file_path)
    await mixer.start(MIXER_RATE)

    assert "test" in mixer._sounds
    expected = int(0.5 * MIXER_RATE)
    assert abs(len(mixer._sounds["test"]) - expected) <= MIXER_RATE // 100


@pytest.mark.asyncio
async def test_stereo_off_rate_file_is_downmixed_and_resampled(tmp_path):
    file_path = tmp_path / "48k_stereo.wav"
    _write_wav(file_path, 48000, 0.5, channels=2)

    mixer = _make_mixer(file_path)
    await mixer.start(MIXER_RATE)

    assert "test" in mixer._sounds
    sound = mixer._sounds["test"]
    assert sound.ndim == 1
    expected = int(0.5 * MIXER_RATE)
    assert abs(len(sound) - expected) <= MIXER_RATE // 100


@pytest.mark.asyncio
async def test_mix_uses_resampled_sound(tmp_path):
    file_path = tmp_path / "44k.wav"
    _write_wav(file_path, 44100, 0.5)

    mixer = _make_mixer(file_path)
    await mixer.start(MIXER_RATE)

    silence = b"\x00\x00" * 160
    mixed = await mixer.mix(silence)
    assert len(mixed) == len(silence)
    assert mixed != silence
