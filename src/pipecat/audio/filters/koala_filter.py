#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Koala noise suppression audio filter for Pipecat.

This module provides an audio filter implementation using PicoVoice's Koala
Noise Suppression engine to reduce background noise in audio streams.
"""

from typing import Sequence

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import pvkoala
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the Koala filter, you need to `pip install pipecat-ai[koala]`.")
    raise Exception(f"Missing module: {e}")


class KoalaFilter(BaseAudioFilter):
    """Audio filter using Koala Noise Suppression from PicoVoice.

    Provides real-time noise suppression for audio streams using PicoVoice's
    Koala engine. The filter buffers audio data to match Koala's required
    frame length and processes it in chunks.
    """

    def __init__(self, *, access_key: str) -> None:
        """Initialize the Koala noise suppression filter.

        Args:
            access_key: PicoVoice access key for Koala engine authentication.
        """
        self._access_key = access_key

        self._filtering = True
        self._sample_rate = 0
        self._koala = pvkoala.create(access_key=f"{self._access_key}")
        self._koala_ready = True
        self._audio_buffer = bytearray()

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        if self._sample_rate != self._koala.sample_rate:
            logger.warning(
                f"Koala filter needs sample rate {self._koala.sample_rate} (got {self._sample_rate})"
            )
            self._koala_ready = False

    async def stop(self):
        """Clean up the Koala engine when stopping."""
        self._koala.reset()

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply Koala noise suppression to audio data.

        Buffers incoming audio and processes it in chunks that match Koala's
        required frame length. Returns filtered audio data.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Noise-suppressed audio data as bytes.
        """
        if not self._koala_ready or not self._filtering:
            return audio

        self._audio_buffer.extend(audio)

        filtered_data: Sequence[int] = []

        num_frames = len(self._audio_buffer) // 2
        while num_frames >= self._koala.frame_length:
            # Grab the number of frames required by Koala.
            num_bytes = self._koala.frame_length * 2
            audio = bytes(self._audio_buffer[:num_bytes])
            # Process audio
            data = np.frombuffer(audio, dtype=np.int16).tolist()
            filtered_data += self._koala.process(data)
            # Adjust audio buffer and check again
            self._audio_buffer = self._audio_buffer[num_bytes:]
            num_frames = len(self._audio_buffer) // 2

        filtered = np.array(filtered_data, dtype=np.int16).tobytes()

        return filtered
