#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp Voice Activity Detection (VAD) implementation for Pipecat.

This module provides a VAD analyzer based on the Krisp VIVA SDK,
which can detect voice activity in audio streams with high accuracy.
Supports 8kHz, 16kHz, 32kHz, 44.1kHz and 48kHz sample rates.
"""

import os
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.krisp_instance import (
    KrispVivaSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
)
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use KrispVivaVADAnalyzer, you need to install krisp_audio.")
    raise Exception(f"Missing module: {e}")


class KrispVivaVadAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using the Krisp VIVA SDK."""

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        frame_duration: int = 10,
        sample_rate: Optional[int] = None,
        params: Optional[VADParams] = None,
    ):
        """Initialize the Krisp VIVA VAD analyzer.

        Args:
            model_path: Path to the Krisp model file (.kef extension).
                If None, uses KRISP_VIVA_VAD_MODEL_PATH environment variable.
            frame_duration: Frame duration in milliseconds (default: 10ms).
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, 44100 or 48000 Hz).
                If None, will be set later.
            params: VAD parameters for detection configuration.

        Raises:
            ValueError: If model_path is not provided and KRISP_VIVA_VAD_MODEL_PATH is not set.
            Exception: If model file doesn't have .kef extension.
            FileNotFoundError: If model file doesn't exist.
        """
        super().__init__(sample_rate=sample_rate, params=params)

        logger.debug("Loading Krisp VIVA VAD model...")

        try:
            # Set model path, checking environment if not specified
            if model_path:
                self._model_path = model_path
            else:
                self._model_path = os.getenv("KRISP_VIVA_VAD_MODEL_PATH")
                if not self._model_path:
                    logger.error(
                        "Model path is not provided and KRISP_VIVA_VAD_MODEL_PATH is not set."
                    )
                    raise ValueError("Model path for KrispVivaVADAnalyzer must be provided.")

            if not self._model_path.endswith(".kef"):
                raise Exception("Model is expected with .kef extension")

            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            self._session = None
            self._frame_duration_ms = frame_duration
            self._samples_per_frame = None
            # Calculate samples per frame if sample_rate is provided
            if sample_rate is not None:
                self._samples_per_frame = int((sample_rate * frame_duration) / 1000)

            # Acquire SDK reference (will initialize on first call)
            KrispVivaSDKManager.acquire()

            logger.debug("Loaded Krisp VIVA VAD")

        except Exception:
            # If initialization fails, release the SDK reference
            KrispVivaSDKManager.release()
            raise

    def _create_session(self, sample_rate: int, frame_duration: int):
        """Create a Krisp VAD session with a specific sample rate.

        Args:
            sample_rate: Sample rate for the session
            frame_duration: Frame duration in milliseconds

        Returns:
            Krisp VAD session instance

        Raises:
            RuntimeError: If session creation fails
        """
        try:
            model_info = krisp_audio.ModelInfo()
            model_info.path = self._model_path

            vad_cfg = krisp_audio.VadSessionConfig()
            vad_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
            vad_cfg.inputFrameDuration = int_to_krisp_frame_duration(frame_duration)
            vad_cfg.modelInfo = model_info

            self._samples_per_frame = int((sample_rate * frame_duration) / 1000)
            session = krisp_audio.VadFloat.create(vad_cfg)
            return session
        except Exception as e:
            logger.error(f"Failed to create Krisp VAD session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create Krisp VAD session: {e}") from e

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate (must be 8000, 16000, 32000 or 48000 Hz).

        Raises:
            ValueError: If sample rate is not 8000, 16000, 32000 or 48000 Hz.
            RuntimeError: If VAD session creation fails.
        """
        if (
            sample_rate != 48000
            and sample_rate != 44100
            and sample_rate != 32000
            and sample_rate != 16000
            and sample_rate != 8000
        ):
            raise ValueError(
                f"Krisp VIVA VAD sample rate needs to be 8000, 16000, 32000, 44100 or 48000 (sample rate: {sample_rate})"
            )

        # Create or recreate session with new sample rate
        try:
            self._session = self._create_session(sample_rate, self._frame_duration_ms)
        except Exception as e:
            logger.error(f"Failed to set sample rate: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create Krisp VAD session: {e}") from e

        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            Number of frames (samples) needed for VAD processing based on
            current sample rate and frame duration.
        """
        # If already calculated from session creation, return it
        if self._samples_per_frame is not None:
            return self._samples_per_frame

        # Calculate from current sample rate if available
        if self.sample_rate > 0:
            return int((self.sample_rate * self._frame_duration_ms) / 1000)

        # Fallback: calculate from initial sample rate if provided
        if self._init_sample_rate is not None:
            return int((self._init_sample_rate * self._frame_duration_ms) / 1000)

        # Default fallback: assume 16kHz @ 10ms = 160 samples
        return int((16000 * self._frame_duration_ms) / 1000)

    def voice_confidence(self, buffer) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze (bytes, int16 format).

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        if self._session is None:
            logger.warning("VAD session not initialized. Cannot process audio.")
            return 0.0

        try:
            # Convert bytes buffer to float32 numpy array
            # Buffer is int16 (2 bytes per sample), need to convert to float32
            audio_int16 = np.frombuffer(buffer, dtype=np.int16)
            # Normalize to [-1.0, 1.0] range
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Process through VAD session
            voice_probability = self._session.process(audio_float32)

            return voice_probability

        except Exception as e:
            logger.error(f"Error analyzing audio with Krisp VIVA VAD: {e}", exc_info=True)
            return 0.0

    async def cleanup(self):
        """Cleanup analyzer resources."""
        try:
            self._session = None
            KrispVivaSDKManager.release()
        except Exception:
            # Ignore errors during cleanup
            pass
