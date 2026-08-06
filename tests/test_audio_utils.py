#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :func:`pipecat.audio.utils.calculate_audio_volume`.

The loudness gate runs on every VAD frame, so its output must stay identical
after the meter/coefficient caching in ``_get_loudness_meter``. These tests pin
``calculate_audio_volume`` to an independent per-call pyloudnorm reference (a
fresh ``Meter`` built on every call, i.e. the pre-caching behavior) across
representative and edge-case frames at both supported sample rates.
"""

import unittest

import numpy as np
import pyloudnorm as pyln

from pipecat.audio.utils import calculate_audio_volume, normalize_value

# Silero VAD frame sizes: 512 samples at 16 kHz and 256 at 8 kHz, both 32 ms.
RATES = {16000: 512, 8000: 256}


def _reference_volume(audio: bytes, sample_rate: int) -> float:
    """Compute the volume with a fresh pyloudnorm Meter per call.

    This mirrors the implementation before meter caching was introduced and is
    used as the ground truth the cached implementation must match exactly.
    """
    audio_np = np.frombuffer(audio, dtype=np.int16)
    audio_float = audio_np.astype(np.float64)
    block_size = audio_np.size / sample_rate
    meter = pyln.Meter(sample_rate, block_size=block_size)
    loudness = meter.integrated_loudness(audio_float)
    return normalize_value(loudness, -20, 80)


def _make_frames(sample_rate: int, num_samples: int) -> dict:
    """Build representative and edge-case single-frame PCM buffers."""
    rng = np.random.default_rng(1234)
    t = np.arange(num_samples) / sample_rate

    spike = np.zeros(num_samples, dtype=np.int16)
    spike[num_samples // 2] = 30000

    speech = 3000 * (
        np.sin(2 * np.pi * 140 * t)
        + 0.5 * np.sin(2 * np.pi * 310 * t)
        + 0.25 * np.sin(2 * np.pi * 900 * t)
    ) + rng.normal(0, 80, num_samples)

    cases = {
        "digital_silence": np.zeros(num_samples, dtype=np.int16),
        "single_sample_spike": spike,
        "full_scale_dc": np.full(num_samples, 32767, dtype=np.int16),
        "full_scale_alternating": np.where(np.arange(num_samples) % 2 == 0, 32767, -32768).astype(
            np.int16
        ),
        "near_silence": np.clip(3 * np.sin(2 * np.pi * 200 * t), -32768, 32767).astype(np.int16),
        "quiet_level": np.clip(
            20 * np.sin(2 * np.pi * 200 * t) + rng.normal(0, 5, num_samples), -32768, 32767
        ).astype(np.int16),
        "dc_offset": np.full(num_samples, 100, dtype=np.int16),
        "speech_like": np.clip(speech, -32768, 32767).astype(np.int16),
    }
    return {name: samples.tobytes() for name, samples in cases.items()}


class TestCalculateAudioVolume(unittest.TestCase):
    """Verify the cached loudness gate is bit-identical to the reference."""

    def test_matches_pyloudnorm_reference(self):
        """calculate_audio_volume must equal a fresh-Meter computation exactly."""
        for sample_rate, num_samples in RATES.items():
            for name, audio in _make_frames(sample_rate, num_samples).items():
                with self.subTest(sample_rate=sample_rate, case=name):
                    self.assertEqual(
                        calculate_audio_volume(audio, sample_rate),
                        _reference_volume(audio, sample_rate),
                    )

    def test_repeated_calls_are_stable(self):
        """The cached meter must not accumulate state across successive frames."""
        for sample_rate, num_samples in RATES.items():
            audio = _make_frames(sample_rate, num_samples)["speech_like"]
            first = calculate_audio_volume(audio, sample_rate)
            for _ in range(5):
                self.assertEqual(calculate_audio_volume(audio, sample_rate), first)

    def test_digital_silence_normalizes_to_zero(self):
        """All-zero audio must normalize to 0.0 (the -inf loudness path)."""
        for sample_rate, num_samples in RATES.items():
            audio = np.zeros(num_samples, dtype=np.int16).tobytes()
            with self.subTest(sample_rate=sample_rate):
                self.assertEqual(calculate_audio_volume(audio, sample_rate), 0.0)


if __name__ == "__main__":
    unittest.main()
