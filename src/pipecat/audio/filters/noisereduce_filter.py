#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Noisereduce audio filter for Pipecat.

This module provides an audio filter implementation using the noisereduce
library to reduce background noise in audio streams through spectral
gating algorithms.
"""

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import noisereduce as nr
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the noisereduce filter, you need to `pip install pipecat-ai[noisereduce]`."
    )
    raise Exception(f"Missing module: {e}")


class NoisereduceFilter(BaseAudioFilter):
    """Audio filter using the noisereduce library for noise suppression.

    Applies spectral gating noise reduction algorithms to suppress background
    noise in audio streams. Uses the noisereduce library's default noise
    reduction parameters.
    """

    def __init__(self) -> None:
        """Initialize the noisereduce filter."""
        self._filtering = True
        self._sample_rate = 0

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate

    async def stop(self):
        """Clean up the filter when stopping."""
        pass

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply noise reduction to audio data using spectral gating.

        Converts audio to float32, applies noisereduce processing, and returns
        the filtered audio clipped to int16 range.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Noise-reduced audio data as bytes.
        """
        if not self._filtering:
            return audio

        data = np.frombuffer(audio, dtype=np.int16)

        # Add a small epsilon to avoid division by zero.
        epsilon = 1e-10
        data = data.astype(np.float32) + epsilon

        # Noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=self._sample_rate)
        audio = np.clip(reduced_noise, -32768, 32767).astype(np.int16).tobytes()

        return audio
