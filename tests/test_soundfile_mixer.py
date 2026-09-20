#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import pytest
import soundfile as sf

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer
from pipecat.frames.frames import MixerEnableFrame, MixerUpdateSettingsFrame

SAMPLE_RATE = 16000


def _write_wav(path, samples, sample_rate=SAMPLE_RATE):
    sf.write(str(path), np.asarray(samples, dtype=np.int16), sample_rate, subtype="PCM_16")
    return str(path)


def _pcm(samples):
    return np.asarray(samples, dtype=np.int16).tobytes()


def _samples(audio):
    return np.frombuffer(audio, dtype=np.int16)


async def _started_mixer(sound_files, default_sound, **kwargs):
    mixer = SoundfileMixer(sound_files=sound_files, default_sound=default_sound, **kwargs)
    await mixer.start(SAMPLE_RATE)
    return mixer


@pytest.mark.asyncio
async def test_start_loads_files_matching_sample_rate(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [1, 2, 3])

    mixer = await _started_mixer({"a": path}, "a")

    assert list(mixer._sounds) == ["a"]
    assert _samples(mixer._sounds["a"].tobytes()).tolist() == [1, 2, 3]


@pytest.mark.asyncio
async def test_start_skips_file_with_wrong_sample_rate(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [1, 2, 3], sample_rate=8000)

    mixer = await _started_mixer({"a": path}, "a")

    assert mixer._sounds == {}
    # With nothing loaded, mixing passes the input through untouched.
    audio = _pcm([10, 20, 30])
    assert await mixer.mix(audio) == audio


@pytest.mark.asyncio
async def test_start_ignores_missing_file(tmp_path):
    mixer = await _started_mixer({"a": str(tmp_path / "missing.wav")}, "a")

    assert mixer._sounds == {}


@pytest.mark.asyncio
async def test_mix_adds_sound_scaled_by_volume(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [1000, 1000, 1000])
    mixer = await _started_mixer({"a": path}, "a", volume=0.5)

    mixed = await mixer.mix(_pcm([100, 200, 300]))

    assert _samples(mixed).tolist() == [600, 700, 800]


@pytest.mark.asyncio
async def test_mix_clips_to_int16_range(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [30000, -30000])
    mixer = await _started_mixer({"a": path}, "a", volume=1.0)

    mixed = await mixer.mix(_pcm([30000, -30000]))

    assert _samples(mixed).tolist() == [32767, -32768]


@pytest.mark.asyncio
async def test_mix_consumes_sound_sequentially(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [1, 2, 3, 4])
    mixer = await _started_mixer({"a": path}, "a", volume=1.0)

    first = await mixer.mix(_pcm([0, 0]))
    second = await mixer.mix(_pcm([0, 0]))

    assert _samples(first).tolist() == [1, 2]
    assert _samples(second).tolist() == [3, 4]


@pytest.mark.asyncio
async def test_mix_loops_when_sound_runs_out(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [1, 2, 3, 4])
    mixer = await _started_mixer({"a": path}, "a", volume=1.0, loop=True)

    await mixer.mix(_pcm([0, 0, 0]))
    looped = await mixer.mix(_pcm([0, 0, 0]))

    assert _samples(looped).tolist() == [1, 2, 3]


@pytest.mark.asyncio
async def test_mix_passes_audio_through_when_not_looping(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [1, 2, 3, 4])
    mixer = await _started_mixer({"a": path}, "a", volume=1.0, loop=False)

    await mixer.mix(_pcm([0, 0, 0]))
    audio = _pcm([5, 5, 5])

    assert await mixer.mix(audio) == audio


@pytest.mark.asyncio
async def test_mixer_enable_frame_toggles_mixing(tmp_path):
    path = _write_wav(tmp_path / "a.wav", [1000, 1000])
    mixer = await _started_mixer({"a": path}, "a", volume=1.0)
    audio = _pcm([1, 1])

    await mixer.process_frame(MixerEnableFrame(enable=False))
    assert await mixer.mix(audio) == audio

    await mixer.process_frame(MixerEnableFrame(enable=True))
    assert _samples(await mixer.mix(audio)).tolist() == [1001, 1001]


@pytest.mark.asyncio
async def test_update_settings_changes_sound_and_restarts_it(tmp_path):
    a = _write_wav(tmp_path / "a.wav", [100, 200, 300, 400])
    b = _write_wav(tmp_path / "b.wav", [5, 5, 5, 5])
    mixer = await _started_mixer({"a": a, "b": b}, "a", volume=1.0)

    await mixer.mix(_pcm([0, 0]))
    await mixer.process_frame(MixerUpdateSettingsFrame(settings={"sound": "b"}))
    mixed = await mixer.mix(_pcm([0, 0]))

    assert _samples(mixed).tolist() == [5, 5]


@pytest.mark.asyncio
async def test_update_settings_ignores_unknown_sound(tmp_path):
    a = _write_wav(tmp_path / "a.wav", [100, 200, 300, 400])
    mixer = await _started_mixer({"a": a}, "a", volume=1.0)

    await mixer.process_frame(MixerUpdateSettingsFrame(settings={"sound": "nope"}))
    mixed = await mixer.mix(_pcm([0, 0]))

    assert _samples(mixed).tolist() == [100, 200]


@pytest.mark.asyncio
async def test_update_settings_changes_volume_and_loop(tmp_path):
    a = _write_wav(tmp_path / "a.wav", [1000, 1000, 1000, 1000])
    mixer = await _started_mixer({"a": a}, "a", volume=1.0, loop=True)

    await mixer.process_frame(MixerUpdateSettingsFrame(settings={"volume": 0.5, "loop": False}))

    assert _samples(await mixer.mix(_pcm([0, 0]))).tolist() == [500, 500]
    assert _samples(await mixer.mix(_pcm([0, 0]))).tolist() == [500, 500]
    # The sound is exhausted and looping is off, so audio passes through.
    audio = _pcm([7, 7])
    assert await mixer.mix(audio) == audio
