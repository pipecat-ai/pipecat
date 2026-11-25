#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Silero Voice Activity Detection (VAD) implementation for Pipecat.

This module provides a VAD analyzer based on the Silero VAD ONNX model,
which can detect voice activity in audio streams with high accuracy.
Supports 8kHz and 16kHz sample rates.
"""

from typing import cast

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState

# Model state resets.
#
# The model carries a recurrent state from one window to the next. Non-speech
# shifts that state so the model under-reports the speech that follows: music
# strongly at either sample rate, and long silence or noise strongly at 8kHz
# and slightly at 16kHz. A reset clears it.
#
# Seconds of audio between resets. After non-speech that suppresses detection,
# speech is detected again within about one interval. A shorter interval
# recovers sooner but puts more resets just ahead of speech, where the model
# occasionally confirms speech a few hundred milliseconds late.
_MODEL_RESET_INTERVAL_SECS = 2.0
# A due reset waits for clear non-speech: the VAD must be quiet and this many
# consecutive windows (about 160 ms) must score below this confidence. One low
# window can be a dip inside a word or the soft start of one; a run of them
# can't. The threshold sits far below the VAD's speech threshold, so borderline
# windows never count.
_MODEL_RESET_MAX_CONFIDENCE = 0.1
_MODEL_RESET_LOW_CONFIDENCE_WINDOWS = 5

try:
    import onnxruntime

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Silero VAD, you need to `uv add pipecat-ai`.")
    raise ImportError(f"Missing module(s): {e}") from e


class SileroOnnxModel:
    """ONNX runtime wrapper for the Silero VAD model.

    Provides voice activity detection using the pre-trained Silero VAD model
    with ONNX runtime for efficient inference. Handles model state management
    and input validation for audio processing.
    """

    def __init__(self, path, force_onnx_cpu=True):
        """Initialize the Silero ONNX model.

        Args:
            path: Path to the ONNX model file.
            force_onnx_cpu: Whether to force CPU execution provider.
        """
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, 16000]

    def _validate_input(self, x, sr: int):
        """Validate and preprocess input audio data."""
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)
        if np.ndim(x) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.ndim}")

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiple of 16000)"
            )
        if sr / np.shape(x)[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        """Reset the internal model states.

        Args:
            batch_size: Batch size for state initialization. Defaults to 1.
        """
        self._state = np.zeros((2, batch_size, 128), dtype="float32")
        self._context = np.zeros((batch_size, 0), dtype="float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int) -> np.ndarray:
        """Process audio input through the VAD model."""
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if np.shape(x)[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {np.shape(x)[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = np.shape(x)[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if not np.shape(self._context)[1]:
            self._context = np.zeros((batch_size, context_size), dtype="float32")

        x = np.concatenate((self._context, x), axis=1)

        if sr in [8000, 16000]:
            ort_inputs = {"input": x, "state": self._state, "sr": np.array(sr, dtype="int64")}
            ort_outs = self.session.run(None, ort_inputs)
            out = cast(np.ndarray, ort_outs[0])
            self._state = cast(np.ndarray, ort_outs[1])
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        return out


class SileroVADAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using the Silero VAD model.

    Implements VAD analysis using the pre-trained Silero ONNX model for
    accurate voice activity detection. Supports 8kHz and 16kHz sample rates.

    The model's recurrent state is reset about every two seconds of audio, but
    only during clear non-speech. Non-speech such as music, or long silence at
    8kHz, leaves the state unable to detect the speech that follows until it is
    reset. A reset during speech makes the model briefly lose confidence, which
    can end the user's speech early, so a due reset waits until the user is
    quiet.
    """

    def __init__(self, *, sample_rate: int | None = None, params: VADParams | None = None):
        """Initialize the Silero VAD analyzer.

        Args:
            sample_rate: Audio sample rate (8000 or 16000 Hz). If None, will be set later.
            params: VAD parameters for detection thresholds and timing.
        """
        super().__init__(sample_rate=sample_rate, params=params)

        logger.debug("Loading Silero VAD model...")

        model_name = "silero_vad.onnx"
        package_path = "pipecat.audio.vad.data"

        try:
            import importlib_resources as impresources

            model_file_path = str(impresources.files(package_path).joinpath(model_name))
        except BaseException:
            from importlib import resources as impresources

            try:
                with impresources.path(package_path, model_name) as f:
                    model_file_path = f
            except BaseException:
                model_file_path = str(impresources.files(package_path).joinpath(model_name))

        self._model = SileroOnnxModel(model_file_path, force_onnx_cpu=True)

        # Windows scored since the model was last reset, and how many of the
        # most recent windows in a row scored below _MODEL_RESET_MAX_CONFIDENCE.
        self._windows_since_reset = 0
        self._low_confidence_windows = 0

        logger.debug("Loaded Silero VAD")

    #
    # VADAnalyzer
    #

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate (must be 8000 or 16000 Hz).

        Raises:
            ValueError: If sample rate is not 8000 or 16000 Hz.
        """
        if sample_rate != 16000 and sample_rate != 8000:
            raise ValueError(
                f"Silero VAD sample rate needs to be 16000 or 8000 (sample rate: {sample_rate})"
            )

        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            Number of frames required (512 for 16kHz, 256 for 8kHz).
        """
        return 512 if self.sample_rate == 16000 else 256

    def voice_confidence(self, buffer) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            # Divide by 32768 because we have signed 16-bit data.
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            confidence = self._model(audio_float32, self.sample_rate)[0]
            self._maybe_reset_model(confidence.item())
            return confidence
        except Exception as e:
            # This comes from an empty audio array
            logger.error(f"Error analyzing audio with Silero VAD: {e}")
            return 0

    def _maybe_reset_model(self, confidence: float):
        """Reset the model's state once a reset is due and the user is quiet.

        Called after each window is scored, so a reset takes effect from the
        next window. A reset clears the model's recurrent state and the audio
        context it prepends to each window; the VAD's own state is untouched.

        A reset becomes due after ``_MODEL_RESET_INTERVAL_SECS`` of audio, and
        happens on the first window after that where the user is quiet and the
        recent windows are clearly non-speech. ``_vad_state`` is still the
        state from before this window, since the caller updates it from the
        confidence returned here. In practice a reset follows soon after the
        user stops speaking, and recurs every interval during non-speech.

        Time is counted in windows of audio rather than by the wall clock, so
        where resets land depends only on the audio.

        Args:
            confidence: The model's speech confidence for the window just scored.
        """
        self._windows_since_reset += 1
        if confidence < _MODEL_RESET_MAX_CONFIDENCE:
            self._low_confidence_windows += 1
        else:
            self._low_confidence_windows = 0

        if (
            self._windows_since_reset * self.num_frames_required()
            >= _MODEL_RESET_INTERVAL_SECS * self.sample_rate
            and self._vad_state == VADState.QUIET
            and self._low_confidence_windows >= _MODEL_RESET_LOW_CONFIDENCE_WINDOWS
        ):
            self._model.reset_states()
            self._windows_since_reset = 0
