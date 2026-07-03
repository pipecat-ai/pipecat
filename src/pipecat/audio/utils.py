#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio utility functions for Pipecat.

This module provides common audio processing utilities including mixing,
format conversion, volume calculation, and codec transformations for
various audio formats used in Pipecat pipelines.
"""

import audioop

import numpy as np
import pyloudnorm as pyln

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.resamplers.soxr_resampler import SOXRAudioResampler
from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler

# Normal speech usually results in many samples between ±500 to ±5000, depending on loudness and mic gain.
# So we are using a threshold that is well below what real speech produces.
SPEAKING_THRESHOLD = 20


def create_file_resampler(**kwargs) -> BaseAudioResampler:
    """Create an audio resampler instance for batch processing of complete audio files.

    Args:
        **kwargs: Additional keyword arguments passed to the resampler constructor.

    Returns:
        A configured SOXRAudioResampler instance.
    """
    return SOXRAudioResampler(**kwargs)


def create_stream_resampler(**kwargs) -> BaseAudioResampler:
    """Create a stream audio resampler instance.

    Args:
        **kwargs: Additional keyword arguments passed to the resampler constructor.

    Returns:
        A configured SOXRStreamAudioResampler instance.
    """
    return SOXRStreamAudioResampler(**kwargs)


def mix_audio(audio1: bytes, audio2: bytes) -> bytes:
    """Mix two audio streams together by adding their samples.

    Both audio streams are assumed to be 16-bit signed integer PCM data.
    If the streams have different lengths, the shorter one is zero-padded
    to match the longer stream.

    Args:
        audio1: First audio stream as raw bytes (16-bit signed integers).
        audio2: Second audio stream as raw bytes (16-bit signed integers).

    Returns:
        Mixed audio data as raw bytes with samples clipped to 16-bit range.
    """
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
    """Interleave left and right mono audio channels into stereo audio.

    Takes two mono audio streams and combines them into a single stereo
    stream by interleaving the samples (L, R, L, R, ...). If the channels
    have different lengths, both are truncated to the shorter length.

    Args:
        left_audio: Left channel audio as raw bytes (16-bit signed integers).
        right_audio: Right channel audio as raw bytes (16-bit signed integers).

    Returns:
        Interleaved stereo audio data as raw bytes.
    """
    left = np.frombuffer(left_audio, dtype=np.int16)
    right = np.frombuffer(right_audio, dtype=np.int16)

    min_length = min(len(left), len(right))
    left = left[:min_length]
    right = right[:min_length]

    stereo = np.column_stack((left, right))

    return stereo.astype(np.int16).tobytes()


def normalize_value(value, min_value, max_value):
    """Normalize a value to the range [0, 1] and clamp it to bounds.

    Args:
        value: The value to normalize.
        min_value: The minimum value of the input range.
        max_value: The maximum value of the input range.

    Returns:
        Normalized value clamped to the range [0, 1].
    """
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    """Calculate the loudness level of audio data using EBU R128 standard.

    Uses the pyloudnorm library to calculate integrated loudness according
    to the EBU R128 recommendation, then normalizes the result to [0, 1].

    Args:
        audio: Audio data as raw bytes (16-bit signed integers).
        sample_rate: Sample rate of the audio in Hz.

    Returns:
        Normalized loudness value between 0 (quiet) and 1 (loud).
    """
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
    """Apply exponential smoothing to a value.

    Exponential smoothing is used to reduce noise in time-series data by
    giving more weight to recent values while still considering historical data.

    Args:
        value: The new value to incorporate.
        prev_value: The previous smoothed value.
        factor: Smoothing factor between 0 and 1. Higher values give more
                weight to the new value.

    Returns:
        The exponentially smoothed value.
    """
    return prev_value + factor * (value - prev_value)


async def ulaw_to_pcm(
    ulaw_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler
):
    """Convert μ-law encoded audio to PCM and optionally resample.

    Args:
        ulaw_bytes: μ-law encoded audio data as raw bytes.
        in_rate: Original sample rate of the μ-law audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        PCM audio data as raw bytes at the specified output rate.
    """
    # Convert μ-law to PCM
    in_pcm_bytes = audioop.ulaw2lin(ulaw_bytes, 2)

    # Resample
    out_pcm_bytes = await resampler.resample(in_pcm_bytes, in_rate, out_rate)

    return out_pcm_bytes


async def pcm_to_ulaw(pcm_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler):
    """Convert PCM audio to μ-law encoding and optionally resample.

    Args:
        pcm_bytes: PCM audio data as raw bytes (16-bit signed integers).
        in_rate: Original sample rate of the PCM audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        μ-law encoded audio data as raw bytes at the specified output rate.
    """
    # Resample
    in_pcm_bytes = await resampler.resample(pcm_bytes, in_rate, out_rate)

    # Convert PCM to μ-law
    out_ulaw_bytes = audioop.lin2ulaw(in_pcm_bytes, 2)

    return out_ulaw_bytes


async def alaw_to_pcm(
    alaw_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler
) -> bytes:
    """Convert A-law encoded audio to PCM and optionally resample.

    Args:
        alaw_bytes: A-law encoded audio data as raw bytes.
        in_rate: Original sample rate of the A-law audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        PCM audio data as raw bytes at the specified output rate.
    """
    # Convert a-law to PCM
    in_pcm_bytes = audioop.alaw2lin(alaw_bytes, 2)

    # Resample
    out_pcm_bytes = await resampler.resample(in_pcm_bytes, in_rate, out_rate)

    return out_pcm_bytes


async def pcm_to_alaw(pcm_bytes: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler):
    """Convert PCM audio to A-law encoding and optionally resample.

    Args:
        pcm_bytes: PCM audio data as raw bytes (16-bit signed integers).
        in_rate: Original sample rate of the PCM audio in Hz.
        out_rate: Desired output sample rate in Hz.
        resampler: Audio resampler instance for rate conversion.

    Returns:
        A-law encoded audio data as raw bytes at the specified output rate.
    """
    # Resample
    in_pcm_bytes = await resampler.resample(pcm_bytes, in_rate, out_rate)

    # Convert PCM to μ-law
    out_alaw_bytes = audioop.lin2alaw(in_pcm_bytes, 2)

    return out_alaw_bytes


def is_silence(pcm_bytes: bytes) -> bool:
    """Determine if an audio sample contains silence by checking amplitude levels.

    This function analyzes raw PCM audio data to detect silence by comparing
    the maximum absolute amplitude against a predefined threshold. The audio
    is expected to be clean speech or complete silence without background noise.

    Args:
        pcm_bytes: Raw PCM audio data as bytes (16-bit signed integers).

    Returns:
        bool: True if the audio sample is considered silence (below threshold),
              False otherwise.

    Note:
        Normal speech typically produces amplitude values between ±500 to ±5000,
        depending on factors like loudness and microphone gain. The threshold
        (SPEAKING_THRESHOLD) is set well below typical speech levels to
        reliably detect silence vs. speech.
    """
    # Convert raw audio bytes to a NumPy array of int16 samples
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)

    # Check the maximum absolute amplitude in the frame
    max_value = np.abs(audio_data).max()

    # If max value is lower than SPEAKING_THRESHOLD, consider it as silence
    return max_value <= SPEAKING_THRESHOLD


def detect_speech_onset(
    pcm_bytes: bytes,
    sample_rate: int,
    num_channels: int = 1,
    *,
    frame_ms: float = 10.0,
    hop_ms: float = 1.0,
    threshold_db: float = -40.0,
    min_voiced_ms: float = 50.0,
) -> int | None:
    """Detect the first sample of sustained audible speech in PCM audio.

    Measures short-time RMS energy: the signal is downmixed to mono, scanned
    with a ``frame_ms`` window hopping every ``hop_ms``, and each window's RMS
    is computed with its mean removed (a DC offset carries no energy). Onset is
    the start of the first run of windows whose RMS stays above ``threshold_db``
    for at least ``min_voiced_ms``.

    Working on energy rather than per-sample amplitude rejects noise-floor blips
    (a lone loud sample averages out over the window), and the minimum-duration
    requirement rejects brief transients. ``hop_ms`` sets the onset resolution.

    Operates on a growing buffer as audio streams in: returns None until enough
    audio has arrived to confirm an onset, at which point the caller stops
    feeding it.

    Args:
        pcm_bytes: Raw PCM audio data (16-bit signed integers).
        sample_rate: Sample rate of the audio in Hz.
        num_channels: Number of interleaved audio channels (downmixed to mono).
        frame_ms: Analysis window length in milliseconds.
        hop_ms: Step between windows in milliseconds (the onset resolution).
        threshold_db: Energy gate in dBFS; windows quieter than this are silence.
            Sits above typical TTS noise-floor padding and below voiced onset.
        min_voiced_ms: Minimum duration the energy must stay above the gate for
            an onset to count.

    Returns:
        The per-channel sample index of speech onset, or None if no sustained
        onset is present yet.
    """
    if sample_rate <= 0:
        return None

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size == 0:
        return None

    # Downmix to a mono signal normalized to [-1, 1], keeping the sign so that
    # mean-removed RMS measures energy correctly.
    if num_channels > 1:
        usable = samples.size - (samples.size % num_channels)
        mono = samples[:usable].reshape(-1, num_channels).mean(axis=1).astype(np.float32)
    else:
        mono = samples.astype(np.float32)
    mono /= 32768.0

    n = mono.size
    frame = max(1, round(sample_rate * frame_ms / 1000))
    hop = max(1, round(sample_rate * hop_ms / 1000))
    if n < frame:
        # Not enough audio for even one window; wait for more.
        return None

    gate = 10.0 ** (threshold_db / 20.0)  # normalized RMS gate (-40 dBFS = 0.01)

    # Edge-pad so each window stays centered on its hop position and a constant
    # start isn't read as a step (which would carry energy).
    pad = frame // 2
    padded = np.pad(mono, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    rms = np.sqrt(np.mean((windows - windows.mean(axis=1, keepdims=True)) ** 2, axis=1))

    active = rms > gate
    if not active.any():
        return None

    # Onset is the start of the first run of active windows lasting at least
    # min_voiced_ms. A window spreads a lone sample across ~frame_ms, so
    # min_voiced_ms must exceed frame_ms to tell a real onset from a blip.
    min_windows = max(1, round(min_voiced_ms / hop_ms))
    run = 0
    for i, is_active in enumerate(active):
        if is_active:
            run += 1
            if run >= min_windows:
                return int((i - min_windows + 1) * hop)
        else:
            run = 0

    return None
