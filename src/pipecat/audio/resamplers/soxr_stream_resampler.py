#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SoX-based audio resampler stream implementation.

This module provides an audio resampler that uses the SoX ResampleStream library
for very high quality audio sample rate conversion.

When to use the SOXRStreamAudioResampler:
1. For real-time processing scenarios
2. When dealing with very long audio signals
3. When processing audio in chunks or streams
4. When you need to reuse the same resampler configuration multiple times, as it saves initialization overhead

"""

import time

import numpy as np
import soxr

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler

CLEAR_STREAM_AFTER_SECS = 0.2


class SOXRStreamAudioResampler(BaseAudioResampler):
    """Audio resampler implementation using the SoX ResampleStream library.

    This resampler uses the SoX ResampleStream library configured for very high
    quality (VHQ) resampling, providing excellent audio quality at the cost
    of additional computational overhead.
    It keeps an internal history which avoids clicks at chunk boundaries.

    Notes:
        - Only supports mono audio (1 channel).
        - Input must be 16-bit signed PCM audio as raw bytes.
    """

    def __init__(self, **kwargs):
        """Initialize the resampler.

        Args:
            **kwargs: Additional keyword arguments (currently unused).
        """
        self._in_rate: float | None = None
        self._out_rate: float | None = None
        self._last_resample_time: float = 0
        self._soxr_stream: soxr.ResampleStream | None = None

    def _initialize(self, in_rate: float, out_rate: float):
        self._in_rate = in_rate
        self._out_rate = out_rate
        self._last_resample_time = time.time()
        self._soxr_stream = soxr.ResampleStream(
            in_rate=in_rate, out_rate=out_rate, num_channels=1, quality="VHQ", dtype="int16"
        )

    def _maybe_clear_internal_state(self):
        current_time = time.time()
        time_since_last_resample = current_time - self._last_resample_time
        # If more than CLEAR_STREAM_AFTER_SECS seconds have passed, clear the resampler state
        if time_since_last_resample > CLEAR_STREAM_AFTER_SECS:
            if self._soxr_stream:
                self._soxr_stream.clear()
        self._last_resample_time = current_time

    def _maybe_initialize_sox_stream(self, in_rate: int, out_rate: int):
        if self._soxr_stream is None:
            self._initialize(in_rate, out_rate)
        else:
            self._maybe_clear_internal_state()

        if self._in_rate != in_rate or self._out_rate != out_rate:
            raise ValueError(
                f"SOXRStreamAudioResampler cannot be reused with different sample rates: "
                f"expected {self._in_rate}->{self._out_rate}, got {in_rate}->{out_rate}"
            )

    async def resample(self, audio: bytes, in_rate: int, out_rate: int) -> bytes:
        """Resample audio data using soxr.ResampleStream resampler library.

        Args:
            audio: Input audio data as raw bytes (16-bit signed integers).
            in_rate: Original sample rate in Hz.
            out_rate: Target sample rate in Hz.

        Returns:
            Resampled audio data as raw bytes (16-bit signed integers).
        """
        if in_rate == out_rate:
            return audio

        self._maybe_initialize_sox_stream(in_rate, out_rate)
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = self._soxr_stream.resample_chunk(audio_data)
        result = resampled_audio.astype(np.int16).tobytes()
        return result
