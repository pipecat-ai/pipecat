#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that Smart Turn v3 still resamples correctly after dropping the soxr
quality preset from ``VHQ`` to ``HQ``.

Smart Turn v3 is a binary classifier on Whisper log-mel features at 16 kHz.
``VHQ`` (~26-tap polyphase) vs ``HQ`` (~16-tap) is well below the noise floor
of the mel feature representation, so model predictions should be unchanged
on representative inputs and the resampled waveforms should be close in L2.

These tests exist to catch regressions if anyone later changes the resampler
or its quality preset.
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
    chirp = 0.4 * np.sin(2 * np.pi * (200.0 + (4000.0 - 200.0) * t / duration_secs) * t)
    tone = 0.2 * np.sin(2 * np.pi * 1000.0 * t)
    noise = 0.02 * rng.standard_normal(n)
    return (chirp + tone + noise).astype(np.float32)


def test_resample_48k_to_16k_close_to_vhq():
    """Dropping to HQ should produce a near-identical 16 kHz waveform to VHQ.

    The two presets differ in filter length, not in cutoff or interpolation
    semantics, so the output difference is dominated by the slightly different
    stop-band attenuation of HQ. On a band-limited speech-like signal that
    difference is small.
    """
    audio = _synthesize_48k_speechlike()
    vhq = soxr.resample(audio, 48_000, _MODEL_SAMPLE_RATE, quality="VHQ")
    hq = soxr.resample(audio, 48_000, _MODEL_SAMPLE_RATE, quality="HQ")

    assert vhq.shape == hq.shape
    diff = vhq - hq
    rms = float(np.sqrt(np.mean(diff**2)))
    peak = float(np.max(np.abs(diff)))

    # Empirically the RMS error between HQ and VHQ on a speech-band signal sits
    # around 1e-4; the peak sits around 1e-3. The bounds below are generous
    # enough to absorb soxr version-to-version drift without becoming a flake.
    assert rms < 5e-3, f"resampled HQ vs VHQ RMS error too large: {rms}"
    assert peak < 5e-2, f"resampled HQ vs VHQ peak error too large: {peak}"


def test_resample_16k_fast_path_unchanged():
    """When input is already 16 kHz the resample step must short-circuit."""
    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=_MODEL_SAMPLE_RATE)
    audio = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    out = analyzer._resample_to_model_rate(audio)
    # Fast path returns the same array object (no copy), guaranteeing zero
    # resample cost when transports already deliver 16 kHz.
    assert out is audio


def test_predict_endpoint_on_48k_input_runs():
    """End-to-end smoke: 48 kHz input -> resample (HQ) -> features -> ONNX.

    We don't assert a specific prediction value (the model can move; the test
    would become noise). We only assert the pipeline returns a well-shaped
    result, which proves the HQ resampler still produces a 16 kHz waveform
    that the Whisper feature extractor accepts.
    """
    analyzer = LocalSmartTurnAnalyzerV3(sample_rate=48_000)
    audio = _synthesize_48k_speechlike()
    result = analyzer._predict_endpoint(audio)

    assert set(result.keys()) == {"prediction", "probability"}
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
