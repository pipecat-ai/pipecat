#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Audio utilities for calculating chunk sizes and durations."""

from enum import Enum


class AudioEncoding(str, Enum):
    """Supported audio encoding formats."""

    PCM = "pcm"
    ULAW = "ulaw"
    ALAW = "alaw"


def get_audio_encoding(metadata: dict) -> AudioEncoding:
    """Extract audio encoding from frame metadata.

    Args:
        metadata: Frame metadata dictionary

    Returns:
        AudioEncoding enum value, defaults to PCM if not specified
    """
    encoding = metadata.get("audio_encoding", "").lower()
    if encoding == "ulaw":
        return AudioEncoding.ULAW
    elif encoding == "alaw":
        return AudioEncoding.ALAW
    return AudioEncoding.PCM


def calculate_audio_bytes_per_sample(encoding: AudioEncoding) -> int:
    """Calculate bytes per sample for different audio encodings.

    Args:
        encoding: Audio encoding format

    Returns:
        Number of bytes per sample
    """
    if encoding == AudioEncoding.PCM:
        return 2  # 16-bit PCM
    elif encoding in (AudioEncoding.ULAW, AudioEncoding.ALAW):
        return 1  # 8-bit compressed
    raise ValueError(f"Unknown audio encoding: {encoding}")


def calculate_chunk_size_bytes(
    sample_rate: int,
    duration_ms: int,
    num_channels: int = 1,
    encoding: AudioEncoding = AudioEncoding.PCM,
) -> int:
    """Calculate the number of bytes needed for a specific duration of audio.

    This function calculates the exact number of bytes required to represent
    a given duration of audio, taking into account the sample rate, number of
    channels, and encoding format.

    Args:
        sample_rate: Audio sample rate in Hz (e.g., 8000, 16000, 44100)
        duration_ms: Duration in milliseconds
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
        encoding: Audio encoding format (PCM, μ-law, A-law)

    Returns:
        Number of bytes needed for the specified duration

    Examples:
        >>> calculate_chunk_size_bytes(8000, 20, 1, AudioEncoding.PCM)
        320  # 20ms of 8kHz mono PCM

        >>> calculate_chunk_size_bytes(8000, 20, 1, AudioEncoding.ULAW)
        160  # 20ms of 8kHz mono μ-law
    """
    samples_per_ms = sample_rate / 1000
    total_samples = int(samples_per_ms * duration_ms)
    bytes_per_sample = calculate_audio_bytes_per_sample(encoding)

    return total_samples * num_channels * bytes_per_sample


def calculate_duration_ms(
    byte_count: int,
    sample_rate: int,
    num_channels: int = 1,
    encoding: AudioEncoding = AudioEncoding.PCM,
) -> float:
    """Calculate the duration in milliseconds for a given number of audio bytes.

    Args:
        byte_count: Number of bytes of audio data
        sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels
        encoding: Audio encoding format

    Returns:
        Duration in milliseconds

    Examples:
        >>> calculate_duration_ms(320, 8000, 1, AudioEncoding.PCM)
        20.0  # 320 bytes of 8kHz mono PCM = 20ms

        >>> calculate_duration_ms(160, 8000, 1, AudioEncoding.ULAW)
        20.0  # 160 bytes of 8kHz mono μ-law = 20ms
    """
    bytes_per_sample = calculate_audio_bytes_per_sample(encoding)
    total_samples = byte_count / (num_channels * bytes_per_sample)
    duration_seconds = total_samples / sample_rate

    return duration_seconds * 1000  # Convert to milliseconds
