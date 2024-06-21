#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import audioop
import numpy as np
import pyloudnorm as pyln


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


def ulaw_8000_to_pcm_16000(ulaw_8000_bytes):
    # Convert μ-law to PCM
    pcm_8000_bytes = audioop.ulaw2lin(ulaw_8000_bytes, 2)

    # Resample from 8000 Hz to 16000 Hz
    pcm_16000_bytes = audioop.ratecv(pcm_8000_bytes, 2, 1, 8000, 16000, None)[0]

    return pcm_16000_bytes


def pcm_16000_to_ulaw_8000(pcm_16000_bytes):
    # Resample from 16000 Hz to 8000 Hz
    pcm_8000_bytes = audioop.ratecv(pcm_16000_bytes, 2, 1, 16000, 8000, None)[0]

    # Convert PCM to μ-law
    ulaw_8000_bytes = audioop.lin2ulaw(pcm_8000_bytes, 2)

    return ulaw_8000_bytes
