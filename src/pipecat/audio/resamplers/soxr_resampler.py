#
# Copyright (c) 2024-2026, Daily
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
from loguru import logger

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler, SoxrQuality
from pipecat.utils.deprecation import deprecated

try:
    import soxr
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use SOXRAudioResampler, you need to `uv add "pipecat-ai[soxr]"`.')
    raise ImportError(f"Missing module: {e}") from e


@deprecated(
    "`SOXRAudioResampler` is deprecated since 1.16.0 and will be removed in 2.0.0. Use "
    "`SincAudioResampler` instead."
)
class SOXRAudioResampler(BaseAudioResampler):
    """Audio resampler implementation using the SoX resampler library.

    This resampler uses the SoX resampler library configured for very high
    quality (VHQ) resampling by default, providing excellent audio quality at
    the cost of additional computational overhead.

    .. deprecated:: 1.16.0
        Use :class:`~pipecat.audio.resamplers.sinc_resampler.SincAudioResampler` instead.
        Will be removed in 2.0.0.
    """

    def __init__(self, *, quality: SoxrQuality = "VHQ", **kwargs):
        """Initialize the SoX audio resampler.

        Args:
            quality: SOXR quality preset. Higher quality means higher CPU cost.
                One of "VHQ" (default, very high quality), "HQ", "MQ", "LQ",
                or "QQ" (quick, lowest latency).
            **kwargs: Reserved for forward compatibility; currently ignored.
        """
        self._quality = quality

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
        resampled_audio = soxr.resample(audio_data, in_rate, out_rate, quality=self._quality)
        result = resampled_audio.astype(np.int16).tobytes()
        return result
