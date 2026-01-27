#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Krisp turn analyzer for end-of-turn detection using Krisp VIVA SDK.

This module provides a turn analyzer implementation using Krisp's turn detection
(Tt) API to determine when a user has finished speaking in a conversation.

Note: This analyzer uses a different model than KrispVivaFilter. The model path
can be specified via the KRISP_VIVA_TURN_MODEL_PATH environment variable or
passed directly to the constructor.
"""

import os
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from pipecat.audio.krisp_instance import (
    KrispVivaSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
)
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.metrics.metrics import MetricsData

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use KrispVivaTurn, you need to install krisp_audio.")
    raise Exception(f"Missing module: {e}")


class KrispTurnParams(BaseTurnParams):
    """Configuration parameters for Krisp turn analysis.

    Parameters:
        threshold: Probability threshold for turn completion (0.0 to 1.0).
            Higher values require more confidence before marking turn as complete.
        frame_duration_ms: Frame duration in milliseconds for turn detection.
            Supported values: 10, 15, 20, 30, 32.
    """

    threshold: float = 0.5
    frame_duration_ms: int = 20


class KrispVivaTurn(BaseTurnAnalyzer):
    """Turn analyzer using Krisp VIVA SDK for end-of-turn detection.

    Uses Krisp's turn detection (Tt) API to determine when a user has finished
    speaking. This analyzer requires a valid Krisp model file to operate.
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        params: Optional[KrispTurnParams] = None,
    ) -> None:
        """Initialize the Krisp turn analyzer.

        Args:
            model_path: Path to the Krisp turn detection model file (.kef extension).
                If None, uses KRISP_VIVA_TURN_MODEL_PATH environment variable.
            sample_rate: Optional initial sample rate for audio processing.
                If provided, this will be used as the fixed sample rate.
            params: Configuration parameters for turn analysis behavior.

        Raises:
            ValueError: If model_path is not provided and KRISP_VIVA_TURN_MODEL_PATH is not set.
            Exception: If model file doesn't have .kef extension.
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If Krisp SDK initialization fails.
        """
        super().__init__(sample_rate=sample_rate)

        # Acquire SDK reference (will initialize on first call)
        try:
            KrispVivaSDKManager.acquire()
            self._sdk_acquired = True
        except Exception as e:
            self._sdk_acquired = False
            raise RuntimeError(f"Failed to initialize Krisp SDK: {e}")

        try:
            # Set model path, checking environment if not specified
            self._model_path = model_path or os.getenv("KRISP_VIVA_TURN_MODEL_PATH")
            if not self._model_path:
                logger.error(
                    "Model path is not provided and KRISP_VIVA_TURN_MODEL_PATH is not set."
                )
                raise ValueError("Model path for KrispVivaTurn must be provided.")

            if not self._model_path.endswith(".kef"):
                raise Exception("Model is expected with .kef extension")

            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            self._params = params or KrispTurnParams()
            self._tt_session = None
            self._preload_tt_session = None
            self._samples_per_frame = None
            self._audio_buffer = bytearray()

            # State tracking
            self._speech_triggered = False
            self._last_probability = None
            self._frame_probabilities = []
            self._last_state = EndOfTurnState.INCOMPLETE

            # Create session with provided sample rate or default to 16000 Hz
            # This preloads the model to improve latency when set_sample_rate is called later
            preload_sample_rate = sample_rate if sample_rate else 16000
            try:
                self._preload_tt_session = self._create_tt_session(preload_sample_rate)
            except Exception as e:
                logger.error(f"Failed to create turn detection session: {e}", exc_info=True)
                self._preload_tt_session = None
                raise RuntimeError(f"Failed to create turn detection session: {e}") from e

        except Exception:
            # If initialization fails, release the SDK reference
            if self._sdk_acquired:
                KrispVivaSDKManager.release()
                self._sdk_acquired = False
            raise

    async def cleanup(self):
        """Release SDK reference when analyzer is destroyed."""
        if self._sdk_acquired:
            try:
                # Clean up session first
                if hasattr(self, "_tt_session") and self._tt_session is not None:
                    self._tt_session = None
                if hasattr(self, "_preload_tt_session") and self._preload_tt_session is not None:
                    self._preload_tt_session = None

                KrispVivaSDKManager.release()
                self._sdk_acquired = False
            except Exception as e:
                logger.error(f"Error in __del__: {e}", exc_info=True)

    def _create_tt_session(self, sample_rate: int):
        """Create a turn detection session with the specified sample rate.

        Args:
            sample_rate: Sample rate for the session

        Returns:
            krisp_audio.TtFloat instance

        Raises:
            ValueError: If sample rate or frame duration is not supported
            RuntimeError: If session creation fails
        """
        try:
            model_info = krisp_audio.ModelInfo()
            model_info.path = self._model_path

            tt_cfg = krisp_audio.TtSessionConfig()
            tt_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
            tt_cfg.inputFrameDuration = int_to_krisp_frame_duration(self._params.frame_duration_ms)
            tt_cfg.modelInfo = model_info

            # Calculate samples per frame for this sample rate
            self._samples_per_frame = int((sample_rate * self._params.frame_duration_ms) / 1000)

            tt_instance = krisp_audio.TtFloat.create(tt_cfg)
            return tt_instance
        except Exception as e:
            logger.error(f"Failed to create Krisp turn detection session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create Krisp turn detection session: {e}") from e

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate and create/update the turn detection session.

        Args:
            sample_rate: The sample rate to set.
        """
        if self._sample_rate == sample_rate:
            return

        super().set_sample_rate(sample_rate)
        # Create session when sample rate is set
        try:
            self._tt_session = self._create_tt_session(self._sample_rate)
            self.clear()
        except Exception as e:
            logger.error(f"Failed to create turn detection session: {e}", exc_info=True)
            self._tt_session = None

    @property
    def frame_probabilities(self) -> list:
        """Get all probabilities from the last append_audio call.

        Returns:
            List of probability values for each frame processed in the last append_audio call.
        """
        return self._frame_probabilities

    @property
    def last_probability(self) -> Optional[float]:
        """Get the last turn probability value computed.

        Returns:
            Last probability value, or None if no frames have been processed yet.
        """
        return self._last_probability

    @property
    def speech_triggered(self) -> bool:
        """Check if speech has been detected and triggered analysis.

        Returns:
            True if speech has been detected and turn analysis is active.
        """
        return self._speech_triggered

    @property
    def params(self) -> KrispTurnParams:
        """Get the current turn analyzer parameters.

        Returns:
            Current turn analyzer configuration parameters.
        """
        return self._params

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Append audio data for turn analysis.

        Args:
            buffer: Raw audio data bytes to append for analysis.
            is_speech: Whether the audio buffer contains detected speech.

        Returns:
            Current end-of-turn state after processing the audio.
        """
        if self._tt_session is None:
            logger.warning("Turn detection session not initialized, returning INCOMPLETE")
            self._last_state = EndOfTurnState.INCOMPLETE
            return EndOfTurnState.INCOMPLETE

        if self._samples_per_frame is None:
            logger.warning("Samples per frame not initialized, returning INCOMPLETE")
            self._last_state = EndOfTurnState.INCOMPLETE
            return EndOfTurnState.INCOMPLETE

        try:
            # Add incoming audio to our buffer
            self._audio_buffer.extend(buffer)

            # Clear frame probabilities from previous call
            self._frame_probabilities = []

            total_samples = len(self._audio_buffer) // 2  # 2 bytes per int16 sample
            num_complete_frames = total_samples // self._samples_per_frame

            if num_complete_frames == 0:
                # Not enough samples for a complete frame yet, return current state
                self._last_state = EndOfTurnState.INCOMPLETE
                return EndOfTurnState.INCOMPLETE

            complete_samples_count = num_complete_frames * self._samples_per_frame
            bytes_to_process = complete_samples_count * 2  # 2 bytes per sample

            audio_to_process = bytes(self._audio_buffer[:bytes_to_process])

            self._audio_buffer = self._audio_buffer[bytes_to_process:]

            audio_int16 = np.frombuffer(audio_to_process, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            frames = audio_float32.reshape(-1, self._samples_per_frame)

            state = EndOfTurnState.INCOMPLETE

            # Process each complete frame
            for frame in frames:
                if is_speech:
                    # Track speech start time
                    if not self._speech_triggered:
                        logger.trace("Speech detected, turn analysis started")
                    self._speech_triggered = True
                # Note: We don't immediately mark as complete on silence detection.
                # Instead, we wait for the model's probability check below to confirm
                # end-of-turn based on the threshold.

                prob = self._tt_session.process(frame.tolist())

                # Negative values indicate the model is not ready yet (working with 100ms data)
                # Skip processing until we get positive probabilities
                if prob < 0:
                    continue

                # Store the probability for external access
                self._last_probability = prob
                self._frame_probabilities.append(prob)

                # Check if turn is complete based on probability threshold
                # Only mark as complete if we've detected speech and the model
                # confirms with sufficient confidence
                if self._speech_triggered and prob >= self._params.threshold:
                    state = EndOfTurnState.COMPLETE
                    self.clear()
                    break

            # Store the last state for analyze_end_of_turn()
            self._last_state = state
            return state

        except Exception as e:
            logger.error(f"Error during Krisp turn detection: {e}", exc_info=True)
            error_state = EndOfTurnState.INCOMPLETE
            self._last_state = error_state
            return error_state

    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        """Analyze the current audio state to determine if turn has ended.

        Returns:
            Tuple containing the end-of-turn state and optional metrics data.
            Returns the last state determined by append_audio().
        """
        # For real-time processing, the state is determined in append_audio
        # Return the last state that was computed
        return self._last_state, None

    def clear(self):
        """Reset the turn analyzer to its initial state."""
        self._speech_triggered = False
        self._audio_buffer.clear()
        self._last_state = EndOfTurnState.INCOMPLETE
