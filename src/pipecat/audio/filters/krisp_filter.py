#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp noise reduction audio filter for Pipecat.

This module provides an audio filter implementation using Krisp's noise
reduction technology to suppress background noise in audio streams.
"""

import os

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    from pipecat_ai_krisp.audio.krisp_processor import KrispAudioProcessor
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the Krisp filter, you need to `pip install pipecat-ai[krisp]`.")
    raise Exception(f"Missing module: {e}")


class KrispProcessorManager:
    """Singleton manager for KrispAudioProcessor instances.

    Ensures that only one KrispAudioProcessor instance exists for the entire
    program.
    """

    _krisp_instance = None

    @classmethod
    def get_processor(cls, sample_rate: int, sample_type: str, channels: int, model_path: str):
        """Get or create a KrispAudioProcessor instance.

        Args:
            sample_rate: Audio sample rate in Hz.
            sample_type: Audio sample type (e.g., "PCM_16").
            channels: Number of audio channels.
            model_path: Path to the Krisp model file.

        Returns:
            Shared KrispAudioProcessor instance.
        """
        if cls._krisp_instance is None:
            cls._krisp_instance = KrispAudioProcessor(
                sample_rate, sample_type, channels, model_path
            )
        return cls._krisp_instance


class KrispFilter(BaseAudioFilter):
    """Audio filter using Krisp noise reduction technology.

    Provides real-time noise reduction for audio streams using Krisp's
    proprietary noise suppression algorithms. Requires a Krisp model file
    for operation.
    """

    def __init__(
        self, sample_type: str = "PCM_16", channels: int = 1, model_path: str = None
    ) -> None:
        """Initialize the Krisp noise reduction filter.

        Args:
            sample_type: The audio sample format. Defaults to "PCM_16".
            channels: Number of audio channels. Defaults to 1.
            model_path: Path to the Krisp model file. If None, uses KRISP_MODEL_PATH
                environment variable.

        Raises:
            ValueError: If model_path is not provided and KRISP_MODEL_PATH is not set.
        """
        super().__init__()

        # Set model path, checking environment if not specified
        self._model_path = model_path or os.getenv("KRISP_MODEL_PATH")
        if not self._model_path:
            logger.error(
                "Model path for KrispAudioProcessor is not provided and KRISP_MODEL_PATH is not set."
            )
            raise ValueError("Model path for KrispAudioProcessor must be provided.")

        self._sample_type = sample_type
        self._channels = channels
        self._sample_rate = 0
        self._filtering = True
        self._krisp_processor = None

    async def start(self, sample_rate: int):
        """Initialize the Krisp processor with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        self._krisp_processor = KrispProcessorManager.get_processor(
            self._sample_rate, self._sample_type, self._channels, self._model_path
        )

    async def stop(self):
        """Clean up the Krisp processor when stopping."""
        self._krisp_processor = None

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply Krisp noise reduction to audio data.

        Converts audio to float32, applies Krisp noise reduction processing,
        and returns the filtered audio clipped to int16 range.

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

        # Process the audio chunk to reduce noise
        reduced_noise = self._krisp_processor.process(data)

        # Clip and set processed audio back to frame
        audio = np.clip(reduced_noise, -32768, 32767).astype(np.int16).tobytes()

        return audio
