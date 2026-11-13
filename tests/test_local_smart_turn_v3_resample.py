#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that Smart Turn v3 resamples its input to 16 kHz correctly.

Smart Turn v3 is a binary classifier on Whisper log-mel features at 16 kHz. The
analyzer's resampled waveform should stay close in L2 to soxr's very high
quality one, so model predictions on representative inputs are unchanged.

These tests exist to catch regressions if anyone later changes the resampler.
"""

import numpy as np
import pytest

soxr = pytest.importorskip("soxr")
ort = pytest.importorskip("onnxruntime")  # noqa: F841 -- needed by LocalSmartTurnAnalyzerV3

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (  # noqa: E402
    _MODEL_SAMPLE_RATE,
    LocalSmartTurnAnalyzerV3,
)


def _synthesize_48k_speechlike(duration_secs: float = 3.0) -> np.ndarray:
    """Build a deterministic 48 kHz signal with speech-like spectral content.

    Linear chirp 200 Hz -> 4 kHz + a 1 kHz tone + low-amplitude white noise. No
    real speech is needed; the resampler doesn't care, and we just want a
    repeatable signal that exercises the polyphase filter across the audio
    band.
    """
    rng = np.random.default_rng(seed=42)
    sample_rate = 48_000
    n = int(duration_secs * sample_rate)
    t = np.linspace(0, duration_secs, n, endpoint=False, dtype=np.float64)
    chirp = 0.4 * np.sin(2 * np.pi * (200.0 * t + (4000.0 - 200.0) * t**2 / (2 * duration_secs)))
    tone = 0.2 * np.sin(2 * np.pi * 1000.0 * t)
    noise = 0.02 * rng.standard_normal(n)
    return (chirp + tone + noise).astype(np.float32)


def test_resample_48k_to_16k_close_to_vhq():
    """The analyzer's 16 kHz waveform should be near-identical to soxr VHQ's.

    Both line the output up with the input and keep the speech band flat, so the
    difference comes from the top of the band, where the filters roll off
    differently, and from 16-bit rounding. On a band-limited speech-like signal
    that difference is small.
    """
    audio = _synthesize_48k_speechlike()
    analyzer = LocalSmartTurnAnalyzerV3()
    analyzer.set_sample_rate(48_000)
    resampled = analyzer._resample_to_model_rate(audio)
    vhq = soxr.resample(audio, 48_000, _MODEL_SAMPLE_RATE, quality="VHQ")

    assert resampled.shape == vhq.shape
    diff = resampled - vhq
    rms = float(np.sqrt(np.mean(diff**2)))
    peak = float(np.max(np.abs(diff)))

    assert rms < 5e-3, f"resampled vs VHQ RMS error too large: {rms}"
    assert peak < 5e-2, f"resampled vs VHQ peak error too large: {peak}"


def test_resample_16k_fast_path_unchanged():
    """When input is already 16 kHz the resample step must short-circuit."""
    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=_MODEL_SAMPLE_RATE)
    audio = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    out = analyzer._resample_to_model_rate(audio)
    # Fast path returns the same array object (no copy), guaranteeing zero
    # resample cost when transports already deliver 16 kHz.
    assert out is audio


def test_predict_endpoint_on_48k_input_runs():
    """End-to-end smoke: 48 kHz input -> resample -> features -> ONNX.

    We don't assert a specific prediction value (the model can move; the test
    would become noise). We only assert the pipeline returns a well-shaped
    result, which proves the resampler produces a 16 kHz waveform that the
    Whisper feature extractor accepts.
    """
    analyzer = LocalSmartTurnAnalyzerV3()
    analyzer.set_sample_rate(48_000)
    audio = _synthesize_48k_speechlike()
    result = analyzer._predict_endpoint(audio)

    assert set(result.keys()) == {"prediction", "probability"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
