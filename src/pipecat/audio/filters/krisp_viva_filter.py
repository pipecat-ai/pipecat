#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp noise reduction audio filter for Pipecat.

This module provides an audio filter implementation using Krisp VIVA SDK.
"""

import os

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the Krisp filter, you need to install krisp_audio.")
    raise Exception(f"Missing module: {e}")


def _log_callback(log_message, log_level):
    logger.info(f"[{log_level}] {log_message}")


class KrispVivaFilter(BaseAudioFilter):
    """Audio filter using the Krisp VIVA SDK.

    Provides real-time noise reduction for audio streams using Krisp's
    proprietary noise suppression algorithms. This filter requires a
    valid Krisp model file to operate.

    Supported sample rates:
        - 8000 Hz
        - 16000 Hz
        - 24000 Hz
        - 32000 Hz
        - 44100 Hz
        - 48000 Hz
    """

    # Initialize Krisp Audio SDK globally
    krisp_audio.globalInit("", _log_callback, krisp_audio.LogLevel.Off)
    SDK_VERSION = krisp_audio.getVersion()
    logger.debug(
        f"Krisp Audio Python SDK Version: {SDK_VERSION.major}."
        f"{SDK_VERSION.minor}.{SDK_VERSION.patch}"
    )

    SAMPLE_RATES = {
        8000: krisp_audio.SamplingRate.Sr8000Hz,
        16000: krisp_audio.SamplingRate.Sr16000Hz,
        24000: krisp_audio.SamplingRate.Sr24000Hz,
        32000: krisp_audio.SamplingRate.Sr32000Hz,
        44100: krisp_audio.SamplingRate.Sr44100Hz,
        48000: krisp_audio.SamplingRate.Sr48000Hz,
    }

    FRAME_SIZE_MS = 10  # Krisp requires audio frames of 10ms duration for processing.

    def __init__(self, model_path: str = None, noise_suppression_level: int = 100) -> None:
        """Initialize the Krisp noise reduction filter.

        Args:
            model_path: Path to the Krisp model file (.kef extension).
                If None, uses KRISP_VIVA_MODEL_PATH environment variable.
            noise_suppression_level: Noise suppression level.

        Raises:
            ValueError: If model_path is not provided and KRISP_VIVA_MODEL_PATH is not set.
            Exception: If model file doesn't have .kef extension.
            FileNotFoundError: If model file doesn't exist.
        """
        super().__init__()

        # Set model path, checking environment if not specified
        self._model_path = model_path or os.getenv("KRISP_VIVA_MODEL_PATH")
        if not self._model_path:
            logger.error("Model path is not provided and KRISP_VIVA_MODEL_PATH is not set.")
            raise ValueError("Model path for KrispAudioProcessor must be provided.")

        if not self._model_path.endswith(".kef"):
            raise Exception("Model is expected with .kef extension")

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        self._filtering = True
        self._session = None
        self._samples_per_frame = None
        self._noise_suppression_level = noise_suppression_level

        # Audio buffer to accumulate samples for complete frames
        self._audio_buffer = bytearray()

    def _int_to_sample_rate(self, sample_rate):
        """Convert integer sample rate to krisp_audio SamplingRate enum.

        Args:
            sample_rate: Sample rate as integer

        Returns:
            krisp_audio.SamplingRate enum value

        Raises:
            ValueError: If sample rate is not supported
        """
        if sample_rate not in self.SAMPLE_RATES:
            raise ValueError("Unsupported sample rate")
        return self.SAMPLE_RATES[sample_rate]

    async def start(self, sample_rate: int):
        """Initialize the Krisp processor with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        model_info = krisp_audio.ModelInfo()
        model_info.path = self._model_path

        nc_cfg = krisp_audio.NcSessionConfig()
        nc_cfg.inputSampleRate = self._int_to_sample_rate(sample_rate)
        nc_cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd10ms
        nc_cfg.outputSampleRate = nc_cfg.inputSampleRate
        nc_cfg.modelInfo = model_info

        self._samples_per_frame = int((sample_rate * self.FRAME_SIZE_MS) / 1000)
        self._session = krisp_audio.NcInt16.create(nc_cfg)

    async def stop(self):
        """Clean up the Krisp processor when stopping."""
        self._session = None

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply Krisp noise reduction to audio data.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Noise-reduced audio data as bytes.
        """
        if not self._filtering:
            return audio

        # Add incoming audio to our buffer
        self._audio_buffer.extend(audio)

        # Calculate how many complete frames we can process
        total_samples = len(self._audio_buffer) // 2  # 2 bytes per int16 sample
        num_complete_frames = total_samples // self._samples_per_frame

        if num_complete_frames == 0:
            # Not enough samples for a complete frame yet, return empty
            return b""

        # Calculate how many bytes we need for complete frames
        complete_samples_count = num_complete_frames * self._samples_per_frame
        bytes_to_process = complete_samples_count * 2  # 2 bytes per sample

        # Extract the bytes we can process
        audio_to_process = bytes(self._audio_buffer[:bytes_to_process])

        # Remove processed bytes from buffer, keep the remainder
        self._audio_buffer = self._audio_buffer[bytes_to_process:]

        # Process the complete frames
        samples = np.frombuffer(audio_to_process, dtype=np.int16)
        frames = samples.reshape(-1, self._samples_per_frame)
        processed_samples = np.empty_like(samples)

        for i, frame in enumerate(frames):
            cleaned_frame = self._session.process(frame, self._noise_suppression_level)
            processed_samples[i * self._samples_per_frame : (i + 1) * self._samples_per_frame] = (
                cleaned_frame
            )

        return processed_samples.tobytes()
