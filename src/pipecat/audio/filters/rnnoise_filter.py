#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RNNoise noise suppression audio filter for Pipecat.

This module provides an audio filter implementation using RNNoise, a recurrent
neural network for audio noise reduction, via the pyrnnoise library.
"""

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    from pyrnnoise import RNNoise
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the RNNoise filter, you need to `pip install pipecat-ai[rnnoise]`."
    )
    raise Exception(f"Missing module: {e}")


class RNNoiseFilter(BaseAudioFilter):
    """Audio filter using RNNoise for noise suppression.

    Provides real-time noise suppression for audio streams using RNNoise, a
    recurrent neural network for audio noise reduction. The filter buffers audio
    data to match RNNoise's required frame length (480 samples at 48kHz) and
    processes it in chunks.
    """

    def __init__(self) -> None:
        """Initialize the RNNoise noise suppression filter."""
        self._filtering = True
        self._sample_rate = 0
        self._rnnoise = None
        self._rnnoise_ready = False

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        if self._sample_rate != 48000:
            logger.warning(f"RNNoise filter needs sample rate 48000 Hz (got {self._sample_rate})")
            self._rnnoise_ready = False
        else:
            self._rnnoise = RNNoise(sample_rate=self._sample_rate)
            self._rnnoise_ready = True

    async def stop(self):
        """Clean up the RNNoise engine when stopping."""
        self._rnnoise = None
        self._rnnoise_ready = False

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply RNNoise noise suppression to audio data.

        Buffers incoming audio and processes it in chunks that match RNNoise's
        required frame length (480 samples at 48kHz). Returns filtered audio data.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Noise-suppressed audio data as bytes.
        """
        if not self._rnnoise_ready or not self._filtering:
            return audio

        # Convert bytes to numpy array (int16)
        audio_samples = np.frombuffer(audio, dtype=np.int16)

        # Process chunk through RNNoise
        # process_chunk handles buffering internally and yields (speech_prob, denoised_frame)
        # denoised_frame is in float32 format normalized to [-1.0, 1.0]
        filtered_frames = []
        for speech_prob, denoised_frame in self._rnnoise.process_chunk(audio_samples, last=False):
            # Convert denoised_frame from float32 [-1.0, 1.0] to int16
            denoised_int16 = (denoised_frame * 32767).astype(np.int16)
            # Handle mono audio (squeeze channel dimension if present)
            if len(denoised_int16.shape) > 1 and denoised_int16.shape[1] == 1:
                denoised_int16 = denoised_int16[:, 0]
            filtered_frames.append(denoised_int16)

        # Combine all processed frames
        if filtered_frames:
            filtered_audio = np.concatenate(filtered_frames)
            return filtered_audio.tobytes()

        # No frames processed yet (buffering)
        return b""
