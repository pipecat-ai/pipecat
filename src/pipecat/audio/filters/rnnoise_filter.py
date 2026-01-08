#
# Copyright (c) 2024-2026, Daily
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
    RNNoise = None
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the RNNoise filter, you need to `pip install pipecat-ai[rnnoise]`."
    )


class RNNoiseFilter(BaseAudioFilter):
    """Audio filter using RNNoise for noise suppression.

    Provides real-time noise suppression for audio streams using RNNoise, a
    recurrent neural network for audio noise reduction. The filter buffers audio
    data to match RNNoise's required frame length (480 samples at 48kHz) and
    processes it in chunks.
    """

    def __init__(self, resampler_quality: str = "QQ") -> None:
        """Initialize the RNNoise noise suppression filter.

        Args:
            resampler_quality: Quality of the resampler if resampling is needed.
                               One of "VHQ", "HQ", "MQ", "LQ", "QQ". Defaults to "QQ"
                               (Quick) for lowest latency.
        """
        self._filtering = True
        self._sample_rate = 0
        self._rnnoise = None
        self._rnnoise_ready = False
        self._resampler_in = None
        self._resampler_out = None
        self._resampler_quality = resampler_quality

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate

        try:
            # RNNoise always requires 48kHz
            self._rnnoise = RNNoise(sample_rate=48000)
            self._rnnoise_ready = True
        except Exception as e:
            logger.error(f"Failed to initialize RNNoise: {e}")
            self._rnnoise_ready = False
            return

        if self._sample_rate != 48000:
            logger.info(f"RNNoise filter enabling resampling: {self._sample_rate} <-> 48000")
            try:
                from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler

                self._resampler_in = SOXRStreamAudioResampler(quality=self._resampler_quality)
                self._resampler_out = SOXRStreamAudioResampler(quality=self._resampler_quality)
            except ImportError as e:
                logger.error(f"Could not import SOXRStreamAudioResampler for resampling: {e}")
                self._rnnoise_ready = False

    async def stop(self):
        """Clean up the RNNoise engine when stopping."""
        self._rnnoise = None
        self._rnnoise_ready = False
        self._resampler_in = None
        self._resampler_out = None

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

        # Resample input if needed
        in_audio = audio
        if self._sample_rate != 48000 and self._resampler_in:
            in_audio = await self._resampler_in.resample(audio, self._sample_rate, 48000)

        # If audio is empty, return empty bytes (no point in noise cancellation)
        if len(in_audio) == 0:
            return b""

        # Convert bytes to numpy array (int16)
        audio_samples = np.frombuffer(in_audio, dtype=np.int16)

        # Process chunk through RNNoise
        # denoise_chunk handles buffering internally and yields (speech_prob, denoised_frame)
        # denoised_frame is in float32 format normalized to [-1.0, 1.0]
        filtered_frames = []
        for speech_prob, denoised_frame in self._rnnoise.denoise_chunk(audio_samples):
            # Check if output is float (needs scaling) or int16 (ready)
            if np.issubdtype(denoised_frame.dtype, np.floating):
                denoised_int16 = (denoised_frame * 32767).astype(np.int16)
            else:
                denoised_int16 = denoised_frame.astype(np.int16)

            # Handle shape (pyrnnoise returns (channels, samples), e.g. (1, 480))
            # We want flat array for mono
            if denoised_int16.ndim > 1:
                denoised_int16 = denoised_int16.squeeze()

            filtered_frames.append(denoised_int16)

        # Combine all processed frames
        if filtered_frames:
            filtered_audio = np.concatenate(filtered_frames).tobytes()

            # Resample output if needed
            if self._sample_rate != 48000 and self._resampler_out:
                return await self._resampler_out.resample(filtered_audio, 48000, self._sample_rate)

            return filtered_audio

        # No frames processed yet (buffering)
        return b""
