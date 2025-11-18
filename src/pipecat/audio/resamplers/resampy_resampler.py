#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Resampy-based audio resampler implementation.

This module provides an audio resampler that uses the resampy library
for high-quality audio sample rate conversion.
"""

import numpy as np
import resampy

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler


class ResampyResampler(BaseAudioResampler):
    """Audio resampler implementation using the resampy library.

    This resampler uses the resampy library's Kaiser windowing filter
    for high-quality audio resampling with good performance characteristics.
    """

    def __init__(self, **kwargs):
        """Initialize the resampy resampler.

        Args:
            **kwargs: Additional keyword arguments (currently unused).
        """
        pass

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample audio data using resampy library.

        Args:
            audio: Input audio data as raw bytes (16-bit signed integers).
            in_rate: Original sample rate in Hz.
            out_rate: Target sample rate in Hz.

        Returns:
            Resampled audio data as raw bytes (16-bit signed integers).
        """
        if in_rate == out_rate:
            return audio
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = resampy.resample(audio_data, in_rate, out_rate, filter="kaiser_fast")
        result = resampled_audio.astype(np.int16).tobytes()
        return result
