#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp noise reduction audio filter for Pipecat.

This module provides an audio filter implementation using Krisp VIVA SDK.
"""

import asyncio
import os

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.krisp_instance import KrispVivaSDKManager, int_to_krisp_sample_rate
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use KrispVivaFilter, you need to install krisp_audio.")
    raise Exception(f"Missing module: {e}")


class KrispVivaFilter(BaseAudioFilter):
    """Audio filter using the Krisp VIVA SDK.

    Provides real-time noise reduction for audio streams using Krisp's
    proprietary noise suppression algorithms. This filter requires a
    valid Krisp model file to operate.
    """

    # TODO: Krisp supports different frame sizes, 10ms, 15ms, 20ms, 30ms, 32ms
    # We should add a configuration option for this.
    FRAME_SIZE_MS = 10

    class State:
        """Filter state enumeration for managing filter lifecycle."""

        STOPPING = "STOPPING"
        STARTING = "STARTING"
        STARTED = "STARTED"
        STOPPED = "STOPPED"

    def __init__(self, model_path: str = None, noise_suppression_level: int = 100) -> None:
        """Initialize the Krisp noise reduction filter.

        Args:
            model_path: Path to the Krisp model file (.kef extension).
                If None, uses KRISP_VIVA_FILTER_MODEL_PATH environment variable.
            noise_suppression_level: Noise suppression level.

        Raises:
            ValueError: If model_path is not provided and KRISP_VIVA_FILTER_MODEL_PATH is not set.
            Exception: If model file doesn't have .kef extension.
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If Krisp SDK initialization fails.
        """
        super().__init__()

        # Acquire SDK reference (will initialize on first call)
        try:
            KrispVivaSDKManager.acquire()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Krisp SDK: {e}")

        try:
            # Set model path, checking environment if not specified
            self._model_path = model_path or os.getenv("KRISP_VIVA_FILTER_MODEL_PATH")
            if not self._model_path:
                logger.error(
                    "Model path is not provided and KRISP_VIVA_FILTER_MODEL_PATH is not set."
                )
                raise ValueError("Model path for KrispAudioProcessor must be provided.")

            if not self._model_path.endswith(".kef"):
                raise Exception("Model is expected with .kef extension")

            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            self._session = None
            self._preload_session = None
            self._samples_per_frame = None
            self._noise_suppression_level = noise_suppression_level
            self._audio_buffer = bytearray()
            self._sdk_acquired = True
            self._state = self.State.STOPPED

            # Adding the model preload mechanism with default sample rate
            # improves the latency of the session creation in the start() method.
            self._preload_model()

        except Exception:
            # If initialization fails, release the SDK reference
            KrispVivaSDKManager.release()
            raise

    def __del__(self):
        """Release SDK reference when filter is destroyed."""
        if hasattr(self, "_sdk_acquired") and self._sdk_acquired:
            try:
                # Clean up session first
                if hasattr(self, "_session") and self._session is not None:
                    self._session = None

                if hasattr(self, "_preload_session") and self._preload_session is not None:
                    self._preload_session = None

                KrispVivaSDKManager.release()
                self._sdk_acquired = False
            except Exception as e:
                logger.error(f"Error in __del__: {e}", exc_info=True)

    def _create_session(self, sample_rate: int):
        """Preload the model with a specific sample rate.

        Args:
            sample_rate: Sample rate for preloading

        Raises:
            Exception: If preloading fails
        """
        try:
            model_info = krisp_audio.ModelInfo()
            model_info.path = self._model_path

            nc_cfg = krisp_audio.NcSessionConfig()
            nc_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
            nc_cfg.inputFrameDuration = krisp_audio.FrameDuration.Fd10ms
            nc_cfg.outputSampleRate = nc_cfg.inputSampleRate
            nc_cfg.modelInfo = model_info

            self._samples_per_frame = int((sample_rate * self.FRAME_SIZE_MS) / 1000)
            self._current_sample_rate = sample_rate
            session = krisp_audio.NcInt16.create(nc_cfg)
            return session
        except Exception as e:
            logger.error(f"Failed to create Krisp session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create Krisp processing session: {e}") from e

    def _preload_model(self):
        """Preload the model with a specific sample rate."""
        try:
            self._preload_session = self._create_session(16000)
        except Exception as e:
            logger.error(f"Failed to preload Krisp model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to preload Krisp model: {e}") from e

    async def start(self, sample_rate: int):
        """Initialize the Krisp processor with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        try:
            self._session = self._create_session(sample_rate)
            self._state = self.State.STARTED
        except Exception as e:
            logger.error(f"Failed to start Krisp session: {e}", exc_info=True)
            self._session = None
            raise RuntimeError(f"Failed to create Krisp processing session: {e}") from e

    async def stop(self):
        """Clean up the Krisp processor when stopping."""
        self._session = None
        self._state = self.State.STOPPED
        self._audio_buffer.clear()

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            if frame.enable == True:
                if not self._state == self.State.STARTED:
                    self._state = self.State.STARTING

            elif frame.enable == False:
                if not self._state == self.State.STOPPED:
                    self._state = self.State.STOPPING

    async def filter(self, audio: bytes) -> bytes:
        """Apply Krisp noise reduction to audio data.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Noise-reduced audio data as bytes.
        """
        # Handle state transitions and return unprocessed audio when not actively filtering
        if self._state == self.State.STOPPED:
            return audio

        if self._state == self.State.STOPPING:

            async def stop():
                if self._state == self.State.STOPPING:
                    self._state = self.State.STOPPED
                    self._session = None
                    self._audio_buffer.clear()
                    self._session = self._create_session(self._current_sample_rate)

            asyncio.create_task(stop())
            return audio
        elif self._state == self.State.STARTING:
            if self._session is None:
                logger.debug("Session is not initialized but state is starting")
                return audio
            self._state = self.State.STARTED
        elif self._state == self.State.STOPPED:
            return audio

        try:
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
                processed_samples[
                    i * self._samples_per_frame : (i + 1) * self._samples_per_frame
                ] = cleaned_frame

            return processed_samples.tobytes()

        except Exception as e:
            logger.error(f"Error during Krisp filtering: {e}", exc_info=True)
            return audio
