#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Numerical-equivalence tests for the vendored Whisper log-mel extractor."""

import numpy as np
import pytest

from pipecat.audio.turn.smart_turn._whisper_features import compute_whisper_log_mel_features

transformers = pytest.importorskip("transformers")


_N_SAMPLES = 128_000  # 8 seconds at 16 kHz


@pytest.fixture(scope="module")
def reference_extractor():
    return transformers.WhisperFeatureExtractor(chunk_length=8)


def _silence():
    return np.zeros(_N_SAMPLES, dtype=np.float32)


def _noise():
    rng = np.random.default_rng(0xC0FFEE)
    return (rng.standard_normal(_N_SAMPLES) * 0.1).astype(np.float32)


def _sweep():
    t = np.linspace(0, 8, _N_SAMPLES, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * (200 + (4000 - 200) * t / 8) * t)).astype(np.float32)


def _partial():
    rng = np.random.default_rng(0xC0FFEE)
    return np.concatenate(
        [
            np.zeros(120_000, dtype=np.float32),
            (rng.standard_normal(8_000) * 0.1).astype(np.float32),
        ]
    )


FIXTURES = [
    pytest.param(_silence, id="silence"),
    pytest.param(_noise, id="noise"),
    pytest.param(_sweep, id="sweep"),
    pytest.param(_partial, id="partial"),
]


@pytest.mark.parametrize("audio_factory", FIXTURES)
@pytest.mark.parametrize("do_normalize", [True, False])
def test_matches_transformers_numpy_path(reference_extractor, audio_factory, do_normalize):
    """The vendored implementation matches transformers' numpy code path tightly."""
    audio = audio_factory()
    got = compute_whisper_log_mel_features(audio, do_normalize=do_normalize)

    # Replicate the numpy code path: pad-or-truncate (float32), optional normalize, then
    # call _np_extract_fbank_features directly. This is apples-to-apples; both
    # implementations are pure numpy.
    x = audio.astype(np.float32)
    if x.size < _N_SAMPLES:
        x = np.pad(x, (0, _N_SAMPLES - x.size), mode="constant")
    elif x.size > _N_SAMPLES:
        x = x[:_N_SAMPLES]
    if do_normalize:
        x = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)

    expected = reference_extractor._np_extract_fbank_features(x[np.newaxis, :], device="cpu")[0]

    assert got.shape == expected.shape == (80, 800)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("audio_factory", FIXTURES)
@pytest.mark.parametrize("do_normalize", [True, False])
def test_matches_transformers_public_call(reference_extractor, audio_factory, do_normalize):
    """The vendored implementation matches the public WhisperFeatureExtractor call.

    The public ``__call__`` dispatches to the torch path when torch is installed,
    which diverges from the numpy path by up to ~1e-4 on high-amplitude inputs
    purely due to ``torch.stft`` vs ``numpy.fft.rfft`` rounding (the divergence
    is internal to transformers, not introduced by this port). A looser tolerance
    is required here than in the numpy-vs-numpy test above.
    """
    audio = audio_factory()
    got = compute_whisper_log_mel_features(audio, do_normalize=do_normalize)
    out = reference_extractor(
        audio,
        sampling_rate=16_000,
        return_tensors="np",
        padding="max_length",
        max_length=_N_SAMPLES,
        truncation=True,
        do_normalize=do_normalize,
    )
    expected = out.input_features.squeeze(0).astype(np.float32)

    assert got.shape == expected.shape == (80, 800)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
