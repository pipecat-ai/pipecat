#
# Copyright (c) 2024-2026, Daily
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
from pipecat.audio.krisp_instance import (
    KrispVivaSDKManager,
    int_to_krisp_frame_duration,
    int_to_krisp_sample_rate,
)
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import krisp_audio
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use KrispVivaFilter, you need to install krisp_audio.")
    raise ImportError(f"Missing module: {e}") from e

# Seconds of no TTS before considering the TTS stream cleared.
_TTS_CLEARED_COOLDOWN = 0.5


class KrispVivaFilter(BaseAudioFilter):
    """Audio filter using the Krisp VIVA SDK.

    Provides real-time noise reduction for audio streams using Krisp's
    proprietary noise suppression algorithms. This filter requires a
    valid Krisp model file to operate.

    Optionally supports TTS detection (iPhone screening feature is standalone model) to delay voice isolation until
    bot speech playback has stopped, preventing later real human speech suppression artifacts.
    Provide ``tts_model_path`` (or set the ``KRISP_VIVA_TTS_MODEL_PATH`` environment variable) to enable this feature.
    """

    def __init__(
        self,
        model_path: str | None = None,
        frame_duration: int = 10,
        noise_suppression_level: int = 100,
        api_key: str = "",
        tts_model_path: str | None = None,
        tts_threshold: float = 0.5,
        tts_detection_timeout: float = 3.0,
    ) -> None:
        """Initialize the Krisp noise reduction filter.

        Args:
            model_path: Path to the Krisp NC model file (.kef extension).
                If None, uses KRISP_VIVA_FILTER_MODEL_PATH environment variable.
            frame_duration: Frame duration in milliseconds.
            noise_suppression_level: Noise suppression level.
            api_key: Krisp SDK API key. If empty, falls back to
                the KRISP_VIVA_API_KEY environment variable.
            tts_model_path: Path to the Krisp TTS detection model file (.kef extension).
                If None, uses KRISP_VIVA_TTS_MODEL_PATH environment variable.
                When not set, TTS detection is disabled and NC starts immediately.
            tts_threshold: Probability threshold (0–1) above which a frame is
                classified as containing TTS. Only used when tts_model_path is set.
            tts_detection_timeout: Seconds to wait for TTS before starting NC.
                If no TTS is detected within this window the NC filter activates
                immediately. Only used when tts_model_path is set.

        Raises:
            ValueError: If model_path is not provided and KRISP_VIVA_FILTER_MODEL_PATH is not set.
            Exception: If a model file doesn't have .kef extension.
            FileNotFoundError: If a model file doesn't exist.
            RuntimeError: If Krisp SDK initialization fails.
        """
        super().__init__()

        self._api_key = api_key

        try:
            # Set model path, checking environment if not specified
            if model_path:
                self._model_path = model_path
            else:
                # Check new environment variable first
                self._model_path = os.getenv("KRISP_VIVA_FILTER_MODEL_PATH")
                # Fall back to old environment variable for backward compatibility
                if not self._model_path:
                    self._model_path = os.getenv("KRISP_VIVA_MODEL_PATH")
                    if self._model_path:
                        logger.warning(
                            "KRISP_VIVA_MODEL_PATH is deprecated. "
                            "Please use KRISP_VIVA_FILTER_MODEL_PATH instead."
                        )
            if not self._model_path:
                logger.error(
                    "Model path is not provided and KRISP_VIVA_FILTER_MODEL_PATH is not set."
                )
                raise ValueError("Model path for KrispAudioProcessor must be provided.")

            if not self._model_path.endswith(".kef"):
                raise Exception("Model is expected with .kef extension")

            if not os.path.isfile(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")

            # Resolve TTS detection model path (optional)
            if tts_model_path:
                self._tts_model_path = tts_model_path
            else:
                self._tts_model_path = os.getenv("KRISP_VIVA_TTS_MODEL_PATH")
            if self._tts_model_path:
                if not self._tts_model_path.endswith(".kef"):
                    raise Exception("TTS model is expected with .kef extension")
                if not os.path.isfile(self._tts_model_path):
                    raise FileNotFoundError(f"TTS model file not found: {self._tts_model_path}")

            self._session = None
            self._tts_detector = None
            self._sdk_acquired = False
            self._samples_per_frame = None
            self._noise_suppression_level = noise_suppression_level
            self._frame_duration_ms = frame_duration
            self._audio_buffer = bytearray()
            self._filtering = True

            # TTS detection state (active only when tts_model_path is set)
            self._tts_threshold = tts_threshold
            self._tts_detection_timeout = tts_detection_timeout
            self._tts_detection_active = False
            self._tts_elapsed_s: float = 0.0
            self._tts_ever_detected = False
            self._tts_last_detected_s: float | None = None

        except Exception:
            # If initialization fails, release the SDK reference
            KrispVivaSDKManager.release()
            raise

    def _create_session(self, sample_rate: int, frame_duration: int):
        """Create a Krisp session with a specific sample rate.

        Args:
            sample_rate: Sample rate for the session
            frame_duration: Frame duration in milliseconds

        Raises:
            Exception: If session creation fails
        """
        try:
            model_info = krisp_audio.ModelInfo()
            model_info.path = self._model_path

            nc_cfg = krisp_audio.NcSessionConfig()
            nc_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
            nc_cfg.inputFrameDuration = int_to_krisp_frame_duration(frame_duration)
            nc_cfg.outputSampleRate = nc_cfg.inputSampleRate
            nc_cfg.modelInfo = model_info

            self._samples_per_frame = int((sample_rate * frame_duration) / 1000)
            self._current_sample_rate = sample_rate
            session = krisp_audio.NcInt16.create(nc_cfg)
            return session
        except Exception as e:
            logger.error(f"Failed to create Krisp session: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create Krisp processing session: {e}") from e

    def _create_tts_detector(self, sample_rate: int, frame_duration: int):
        """Create a Krisp TTS detector session.

        Args:
            sample_rate: Sample rate for the session.
            frame_duration: Frame duration in milliseconds.

        Raises:
            RuntimeError: If detector creation fails.
        """
        try:
            model_info = krisp_audio.ModelInfo()
            model_info.path = self._tts_model_path

            config = krisp_audio.TtsDetectionSessionConfig()
            config.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
            config.inputFrameDuration = int_to_krisp_frame_duration(frame_duration)
            config.modelInfo = model_info

            return krisp_audio.TtsDetectorFloat.create(config)
        except Exception as e:
            logger.error(f"Failed to create TTS detector: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create TTS detector session: {e}") from e

    def _advance_tts_detection(self, frames: np.ndarray) -> bool:
        """Run TTS detection on audio frames.

        Args:
            frames: Audio frames as int16 samples shaped (num_frames, samples_per_frame).

        Returns:
            True when noise cancellation should activate, False while still in the
            TTS detection phase.
        """
        frame_duration_s = self._frame_duration_ms / 1000.0

        for tts_frame in frames:
            frame_float = tts_frame.astype(np.float32) / 32768.0
            probability = self._tts_detector.process(frame_float)
            self._tts_elapsed_s += frame_duration_s
            if probability > self._tts_threshold:
                self._tts_ever_detected = True
                self._tts_last_detected_s = self._tts_elapsed_s

        if self._tts_ever_detected:
            if self._tts_elapsed_s - self._tts_last_detected_s >= _TTS_CLEARED_COOLDOWN:
                logger.debug("TTS cleared, starting NC filter")
                return True
            return False

        if self._tts_elapsed_s >= self._tts_detection_timeout:
            logger.debug(
                f"TTS detection timeout ({self._tts_detection_timeout}s elapsed), "
                "starting NC filter"
            )
            return True

        return False

    async def start(self, sample_rate: int):
        """Initialize the Krisp processor with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        try:
            # Acquire SDK reference (will initialize on first call)
            KrispVivaSDKManager.acquire(api_key=self._api_key)
            self._sdk_acquired = True
            self._session = self._create_session(sample_rate, self._frame_duration_ms)

            if self._tts_model_path:
                self._tts_detector = self._create_tts_detector(sample_rate, self._frame_duration_ms)
                self._tts_detection_active = True
                self._tts_elapsed_s = 0.0
                self._tts_ever_detected = False
                self._tts_last_detected_s = None
                logger.debug(
                    f"TTS detection enabled (timeout={self._tts_detection_timeout}s), "
                    "delaying filter start"
                )
        except Exception as e:
            logger.error(f"Failed to start Krisp session: {e}", exc_info=True)
            self._session = None
            raise RuntimeError(f"Failed to create Krisp processing session: {e}") from e

    async def stop(self):
        """Release the Krisp processor and its SDK reference.

        The SDK release is guarded so repeated calls do not over-decrement the
        shared reference count.
        """
        try:
            self._session = None
            self._tts_detector = None
            self._tts_detection_active = False
            self._audio_buffer.clear()
            if self._sdk_acquired:
                self._sdk_acquired = False
                KrispVivaSDKManager.release()
        except Exception as e:
            logger.error(f"Error in stop: {e}", exc_info=True)
            raise RuntimeError(f"Failed to stop Krisp processor: {e}") from e

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply Krisp noise reduction to audio data.

        When TTS detection is enabled the audio passes through unmodified until
        TTS is no longer present (or the detection timeout expires), after which
        noise cancellation activates for the remainder of the session.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Noise-reduced audio data as bytes, or the original audio while in
            the TTS detection phase.
        """
        if not self._filtering:
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

            samples = np.frombuffer(audio_to_process, dtype=np.int16)
            frames = samples.reshape(-1, self._samples_per_frame)

            # TTS detection phase: pass audio through until bot speech clears
            if self._tts_detection_active and self._tts_detector is not None:
                if not self._advance_tts_detection(frames):
                    return audio_to_process
                self._tts_detection_active = False

            # Apply NC filter
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
