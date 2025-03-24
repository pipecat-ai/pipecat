#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import audioop

import numpy as np
import pyloudnorm as pyln
import soxr

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.resamplers.soxr_resampler import SOXRAudioResampler


def create_default_resampler(**kwargs) -> BaseAudioResampler:
    return SOXRAudioResampler(**kwargs)


def mix_audio(audio1: bytes, audio2: bytes) -> bytes:
    data1 = np.frombuffer(audio1, dtype=np.int16)
    data2 = np.frombuffer(audio2, dtype=np.int16)

    # Max length
    max_length = max(len(data1), len(data2))

    # Zero-pad the arrays to the same length
    padded1 = np.pad(data1, (0, max_length - len(data1)), mode="constant")
    padded2 = np.pad(data2, (0, max_length - len(data2)), mode="constant")

    # Mix the arrays
    mixed_audio = padded1.astype(np.int32) + padded2.astype(np.int32)
    mixed_audio = np.clip(mixed_audio, -32768, 32767).astype(np.int16)

    return mixed_audio.astype(np.int16).tobytes()


def interleave_stereo_audio(left_audio: bytes, right_audio: bytes) -> bytes:
    left = np.frombuffer(left_audio, dtype=np.int16)
    right = np.frombuffer(right_audio, dtype=np.int16)

    min_length = min(len(left), len(right))
    left = left[:min_length]
    right = right[:min_length]

    stereo = np.column_stack((left, right))

    return stereo.astype(np.int16).tobytes()


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


async def ulaw_to_pcm(
    ulaw_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler
):
    # Convert μ-law to PCM
    in_pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)

    # Resample
    out_pcm_bytes = await resampler.resample(in_pcm_bytes, in_rate, out_rate)

    return out_pcm_bytes


async def pcm_to_ulaw(pcm_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler):
    # Resample
    in_pcm_bytes = await resampler.resample(pcm_bytes, in_rate, out_rate)

    # Convert PCM to μ-law
    out_ulaw_bytes = audioop.lin2ulaw(in_pcm_bytes, 2)

    return out_ulaw_bytes


async def alaw_to_pcm(
    alaw_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler
) -> bytes:
    # Convert a-law to PCM
    in_pcm_bytes = audioop.alaw2lin(alaw_bytes, 2)

    # Resample
    out_pcm_bytes = await resampler.resample(in_pcm_bytes, in_rate, out_rate)

    return out_pcm_bytes


async def pcm_to_alaw(pcm_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler):
    # Resample
    in_pcm_bytes = await resampler.resample(pcm_bytes, in_rate, out_rate)

    # Convert PCM to μ-law
    out_alaw_bytes = audioop.lin2alaw(in_pcm_bytes, 2)

    return out_alaw_bytes
