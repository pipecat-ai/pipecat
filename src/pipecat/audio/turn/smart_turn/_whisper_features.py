#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Numpy-only Whisper-style log-mel feature extraction for Smart Turn v3.

Vendored from `transformers.WhisperFeatureExtractor` so Smart Turn v3 can
compute features without importing the `transformers` package. The math
mirrors `transformers.models.whisper.feature_extraction_whisper` and
`transformers.audio_utils` (Apache-2.0 licensed); see those modules in the
upstream HuggingFace repository for the reference implementation.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 80
_SAMPLING_RATE = 16000
_MEL_FLOOR = 1e-10
_NORM_VARIANCE_EPS = 1e-7


def _hertz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    """Convert frequencies in hertz to mels using the Slaney scale."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    return mels


def _mel_to_hertz_slaney(mels: np.ndarray) -> np.ndarray:
    """Convert frequencies in mels to hertz using the Slaney scale."""
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _build_mel_filterbank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    """Build a Slaney-normalized triangular mel filterbank.

    Returns a matrix of shape ``(num_frequency_bins, num_mel_filters)`` that
    projects a power spectrogram onto the mel scale.
    """
    mel_min = float(_hertz_to_mel_slaney(np.array([min_frequency], dtype=np.float64))[0])
    mel_max = float(_hertz_to_mel_slaney(np.array([max_frequency], dtype=np.float64))[0])
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    # Slaney area normalization.
    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters


def _periodic_hann_window(window_length: int) -> np.ndarray:
    """Return a length-``window_length`` periodic Hann window (matches torch.hann_window)."""
    return np.hanning(window_length + 1)[:-1]


# Precomputed constants for the Whisper feature extractor defaults used by Smart Turn v3.
_HANN_WINDOW = _periodic_hann_window(_N_FFT)
_MEL_FILTERS = _build_mel_filterbank(
    num_frequency_bins=_N_FFT // 2 + 1,
    num_mel_filters=_N_MELS,
    min_frequency=0.0,
    max_frequency=_SAMPLING_RATE / 2.0,
    sampling_rate=_SAMPLING_RATE,
)


def _power_spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """Compute a centered power spectrogram (reflect-padded, real-FFT, |.|^2).

    Returns shape ``(num_frequency_bins, num_frames)`` in float64 to match the
    reference implementation, which promotes to float64 internally.
    """
    pad = frame_length // 2
    padded = np.pad(waveform.astype(np.float64), (pad, pad), mode="reflect")
    win = window.astype(np.float64)

    # Zero-copy view of all length-`frame_length` windows in `padded`, then
    # decimate by `hop_length` to land on the analysis frames the loop walked.
    # Shape: (num_frames, frame_length). Bit-identical to the per-frame slices
    # the previous Python loop produced.
    windows = sliding_window_view(padded, frame_length)[::hop_length]
    # One batched rFFT replaces ~800 Python-level calls into numpy. Output
    # dtype is complex128, same as the loop's `np.empty(..., complex128)`.
    spec = np.fft.rfft(windows * win, axis=-1)

    return (np.abs(spec) ** 2).T  # (num_frequency_bins, num_frames)


def compute_whisper_log_mel_features(
    audio: np.ndarray,
    *,
    do_normalize: bool = True,
) -> np.ndarray:
    """Compute Whisper-style log-mel features for Smart Turn v3.

    Replicates the output of
    ``transformers.WhisperFeatureExtractor(chunk_length=8)(audio, sampling_rate=16000,
    padding="max_length", max_length=128000, truncation=True, do_normalize=...,
    return_tensors="np").input_features.squeeze(0)``.

    Assumes the caller has already padded or truncated the input to exactly
    ``128000`` samples (8 seconds at 16 kHz); shorter inputs are zero-padded
    at the end, longer inputs are truncated to the leading window.

    Args:
        audio: One-dimensional float audio at 16 kHz. Any numeric dtype is
            accepted; the function casts to float64 internally and returns
            float32.
        do_normalize: If True, apply zero-mean unit-variance normalization to
            the waveform before computing the spectrogram. Matches the
            transformers ``do_normalize=True`` behavior.

    Returns:
        Float32 ndarray of shape ``(80, 800)`` containing log-mel features in
        roughly the ``[-1, 1]`` range.
    """
    if audio.ndim != 1:
        raise ValueError(f"Expected 1-D audio, got shape {audio.shape}")

    x = np.asarray(audio, dtype=np.float32)
    n_samples = _SAMPLING_RATE * 8  # 128000
    if x.size < n_samples:
        x = np.pad(x, (0, n_samples - x.size), mode="constant")
    elif x.size > n_samples:
        x = x[:n_samples]

    # Zero-mean unit-variance waveform normalization happens BEFORE the
    # spectrogram, mirroring transformers' WhisperFeatureExtractor.__call__.
    # Done in float32 to match the reference's precision (it normalizes the
    # padded float32 buffer before the spectrogram promotes to float64).
    if do_normalize:
        x = (x - x.mean()) / np.sqrt(x.var() + _NORM_VARIANCE_EPS)

    magnitudes = _power_spectrogram(x, _HANN_WINDOW, _N_FFT, _HOP_LENGTH)
    mel_spec = np.maximum(_MEL_FLOOR, _MEL_FILTERS.T @ magnitudes)
    log_spec = np.log10(mel_spec)
    log_spec = log_spec[:, :-1]  # drop the trailing frame, matching the reference.
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)
