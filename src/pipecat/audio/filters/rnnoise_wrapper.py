#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Python wrapper for RNNoise C library."""

import ctypes
import os
from ctypes import POINTER, c_char_p, c_float, c_int, c_void_p
from typing import Optional

from loguru import logger

# Define type aliases to avoid call expressions in annotations, which
# some type checkers (e.g., Pylance) flag as "Call expression not allowed in type expression".
FloatPtr = POINTER(c_float)


class RNNoiseWrapper:
    """Python wrapper for RNNoise C library using ctypes."""

    def __init__(self, library_path: Optional[str] = None):
        """Initialize RNNoise wrapper.

        Args:
            library_path: Path to librnnoise.so. If None, searches common paths.
        """
        self._lib = None
        self._library_path = library_path
        self._load_library()
        self._setup_functions()

    def _load_library(self):
        """Load the RNNoise shared library."""
        if not self._library_path:
            raise Exception(
                """Please provide the path for librnnoise.so by building it from source. You can find the instruction
                at https://github.com/xiph/rnnoise/blob/main/README". It essentially involves running make."""
            )

        try:
            if os.path.exists(self._library_path):
                self._lib = ctypes.CDLL(self._library_path)
                logger.debug(f"Successfully loaded RNNoise library from {self._library_path}")
                return
        except OSError as e:
            logger.debug(f"Failed to load RNNoise library from {self._library_path}: {e}")

        raise OSError(
            "Could not find or load RNNoise library. Please ensure librnnoise.so is installed."
        )

    def _setup_functions(self):
        """Set up C function signatures."""
        # int rnnoise_get_size(void)
        self._lib.rnnoise_get_size.argtypes = []
        self._lib.rnnoise_get_size.restype = c_int

        # int rnnoise_get_frame_size(void)
        self._lib.rnnoise_get_frame_size.argtypes = []
        self._lib.rnnoise_get_frame_size.restype = c_int

        # int rnnoise_init(DenoiseState *st, RNNModel *model)
        self._lib.rnnoise_init.argtypes = [c_void_p, c_void_p]
        self._lib.rnnoise_init.restype = c_int

        # DenoiseState *rnnoise_create(RNNModel *model)
        self._lib.rnnoise_create.argtypes = [c_void_p]
        self._lib.rnnoise_create.restype = c_void_p

        # void rnnoise_destroy(DenoiseState *st)
        self._lib.rnnoise_destroy.argtypes = [c_void_p]
        self._lib.rnnoise_destroy.restype = None

        # float rnnoise_process_frame(DenoiseState *st, float *out, const float *in)
        self._lib.rnnoise_process_frame.argtypes = [c_void_p, FloatPtr, FloatPtr]
        self._lib.rnnoise_process_frame.restype = c_float

        # RNNModel *rnnoise_model_from_filename(const char *filename)
        self._lib.rnnoise_model_from_filename.argtypes = [c_char_p]
        self._lib.rnnoise_model_from_filename.restype = c_void_p

        # void rnnoise_model_free(RNNModel *model)
        self._lib.rnnoise_model_free.argtypes = [c_void_p]
        self._lib.rnnoise_model_free.restype = None

    def get_size(self) -> int:
        """Get the size of DenoiseState."""
        return self._lib.rnnoise_get_size()

    def get_frame_size(self) -> int:
        """Get the frame size (number of samples processed at once)."""
        return self._lib.rnnoise_get_frame_size()

    def create_state(self, model_path: Optional[str] = None) -> c_void_p:
        """Create a new DenoiseState.

        Args:
            model_path: Path to custom model file. If None, uses default model.

        Returns:
            Pointer to DenoiseState.
        """
        model = None
        if model_path:
            model = self._lib.rnnoise_model_from_filename(model_path.encode("utf-8"))
            if not model:
                raise RuntimeError(f"Failed to load model from {model_path}")

        state = self._lib.rnnoise_create(model)
        if not state:
            if model:
                self._lib.rnnoise_model_free(model)
            raise RuntimeError("Failed to create RNNoise state")

        return state

    def destroy_state(self, state: c_void_p):
        """Destroy a DenoiseState."""
        try:
            if state:
                self._lib.rnnoise_destroy(state)
        except Exception as e:
            logger.error(f"Error destroying RNNoise state: {e}")

    def process_frame(self, state: c_void_p, audio_in: FloatPtr, audio_out: FloatPtr) -> float:  # type: ignore
        """Process a frame of audio.

        Args:
            state: DenoiseState pointer
            audio_in: Input audio frame (frame_size samples)
            audio_out: Output audio frame (frame_size samples)

        Returns:
            Voice activity probability (0.0 to 1.0)
        """
        try:
            if not state:
                raise ValueError("RNNoise state is None")
            return self._lib.rnnoise_process_frame(state, audio_out, audio_in)
        except Exception as e:
            logger.error(f"Error processing frame in RNNoise: {e}")
            # Return 0.0 for VAD probability on error
            return 0.0
