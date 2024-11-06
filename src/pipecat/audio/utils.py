#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import audioop
import numpy as np
import pyloudnorm as pyln
import resampy


def resample_audio(audio: bytes, original_rate: int, target_rate: int) -> bytes:
    if original_rate == target_rate:
        return audio
    audio_data = np.frombuffer(audio, dtype=np.int16)
    resampled_audio = resampy.resample(audio_data, original_rate, target_rate)
    return resampled_audio.astype(np.int16).tobytes()


def normalize_value(value, min_value, max_value):
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    audio_np = np.frombuffer(audio, dtype=np.int16)
    audio_float = audio_np.astype(np.float64)

    block_size = audio_np.size / sample_rate
    meter = pyln.Meter(sample_rate, block_size=block_size)
    loudness = meter.integrated_loudness(audio_float)

    # Loudness goes from -20 to 80 (more or less), where -20 is quiet and 80 is
    # loud.
    loudness = normalize_value(loudness, -20, 80)

    return loudness


def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    return prev_value + factor * (value - prev_value)


def ulaw_to_pcm(ulaw_bytes: bytes, in_sample_rate: int, out_sample_rate: int):
    # Convert μ-law to PCM
    in_pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)

    # Resample
    out_pcm_bytes = audioop.ratecv(in_pcm_bytes, 2, 1, in_sample_rate, out_sample_rate, None)[0]

    return out_pcm_bytes


def pcm_to_ulaw(pcm_bytes: bytes, in_sample_rate: int, out_sample_rate: int):
    # Resample
    in_pcm_bytes = audioop.ratecv(pcm_bytes, 2, 1, in_sample_rate, out_sample_rate, None)[0]

    # Convert PCM to μ-law
    ulaw_bytes = audioop.lin2ulaw(in_pcm_bytes, 2)

    return ulaw_bytes
