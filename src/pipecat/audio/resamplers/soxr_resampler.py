#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SoX-based audio resampler implementation.

This module provides an audio resampler that uses the SoX resampler library
for very high-quality audio sample rate conversion.

When to use the SOXRAudioResampler:
1. For batch processing of complete audio files
2. When you have all the audio data available at once

"""

import numpy as np
import soxr

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler


class SOXRAudioResampler(BaseAudioResampler):
    """Audio resampler implementation using the SoX resampler library.

    This resampler uses the SoX resampler library configured for very high
    quality (VHQ) resampling, providing excellent audio quality at the cost
    of additional computational overhead.
    """

    def __init__(self, **kwargs):
        """Initialize the SoX audio resampler.

        Args:
            **kwargs: Additional keyword arguments (currently unused).
        """
        pass

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample audio data using SoX resampler library.

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
        resampled_audio = soxr.resample(audio_data, in_rate, out_rate, quality="VHQ")
        result = resampled_audio.astype(np.int16).tobytes()
        return result
