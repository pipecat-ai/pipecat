#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RNNoise filter for audio noise reduction."""

import ctypes
from ctypes import POINTER, c_float
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.filters.rnnoise_wrapper import RNNoiseWrapper
from pipecat.frames.frames import (
    FilterControlFrame,
    FilterEnableFrame,
)


class RNNoiseFilter(BaseAudioFilter):
    """Audio filter using RNNoise for noise reduction.

    RNNoise processes audio at 48kHz in frames of 480 samples (10ms).
    Input audio at different sample rates will be resampled.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        library_path: Optional[str] = None,
        target_sample_rate: int = 48000,
        **kwargs,
    ) -> None:
        """Initialize RNNoise filter.

        Args:
            model_path: Path to custom RNNoise model. If None, uses default model.
            library_path: Path to librnnoise.so. If None, searches common paths.
            target_sample_rate: Sample rate for RNNoise processing (should be 48000).
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__()

        self._model_path = model_path
        self._library_path = library_path
        self._target_sample_rate = target_sample_rate
        self._input_sample_rate = 0
        self._filtering = True

        # RNNoise components
        self._rnnoise = None
        self._state = None
        self._frame_size = 0

        # Audio buffering for frame-based processing
        self._input_buffer = np.array([], dtype=np.float32)
        self._need_resampling = False

        # Factor by which we upsample (and later down-sample) incoming audio so
        # that RNNoise always sees 48 kHz input. Calculated in start().
        self._upsample_factor: int = 1

        # Initialize RNNoise wrapper
        try:
            self._rnnoise = RNNoiseWrapper(library_path)
            self._frame_size = self._rnnoise.get_frame_size()
            logger.info(f"RNNoise initialized: frame_size={self._frame_size}")
        except Exception as e:
            logger.error(f"Failed to initialize RNNoise: {e}")
            raise

    async def start(self, sample_rate: int):
        """Start the filter with given sample rate."""
        self._input_sample_rate = sample_rate

        # Determine whether we can perform cheap integer-factor resampling.  We
        # rely on simple sample repetition/decimation which only works when the
        # input rate is an integer divisor of 48 kHz (e.g. 8 k, 16 k, 24 k).
        if self._target_sample_rate % sample_rate == 0:
            self._upsample_factor = self._target_sample_rate // sample_rate
            self._need_resampling = self._upsample_factor != 1
        else:
            # Fallback: disable resampling and warn – audio will be fed to
            # RNNoise at the original rate which may yield sub-optimal results.
            logger.warning(
                "Input sample rate {} Hz is not an integer divisor of {} Hz; "
                "simple up-sampling is impossible. RNNoise will process audio "
                "at the original rate which may reduce denoising quality.",
                sample_rate,
                self._target_sample_rate,
            )
            self._upsample_factor = 1
            self._need_resampling = False

        logger.debug(
            f"RNNOiseFilter - sample_rate: {self._input_sample_rate} upsample_factor: {self._upsample_factor}"
        )

        # Create RNNoise state
        try:
            self._state = self._rnnoise.create_state(self._model_path)
            logger.info("RNNoise state created successfully")
        except Exception as e:
            logger.error(f"Failed to create RNNoise state: {e}")
            raise

        # Reset buffer
        self._input_buffer = np.array([], dtype=np.float32)

    async def stop(self):
        """Stop the filter and cleanup resources."""
        try:
            if self._state and self._rnnoise:
                self._rnnoise.destroy_state(self._state)
                self._state = None

            self._input_buffer = np.array([], dtype=np.float32)
            logger.info("RNNoise filter stopped")
        except Exception as e:
            logger.error(f"Error stopping RNNoise filter: {e}")

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames."""
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable
            logger.debug(f"RNNoise filtering {'enabled' if frame.enable else 'disabled'}")

    async def filter(self, audio: bytes) -> bytes:
        """Filter audio through RNNoise.

        Args:
            audio: Input audio as bytes (int16 PCM)

        Returns:
            Filtered audio as bytes
        """
        # Early return if filtering disabled or no state
        if not self._filtering or not self._state:
            return audio

        # Handle empty audio
        if not audio or len(audio) == 0:
            return audio

        try:
            # Convert bytes to int16 numpy array for processing
            original_audio_int16 = np.frombuffer(audio, dtype=np.int16)

            # Convert to float32 for processing *without* normalising to ±1.
            # RNNoise expects sample values in int16 range but as floats.
            audio_data = original_audio_int16.astype(np.float32)

            # -----------------------------------------------------------------
            # Cheap integer-factor up-sampling (sample-holding) so that the
            # signal matches RNNoise's expected 48 kHz input rate.
            # -----------------------------------------------------------------
            if self._upsample_factor > 1:
                audio_data = np.repeat(audio_data, self._upsample_factor)

            # Add to buffer
            self._input_buffer = np.concatenate([self._input_buffer, audio_data])

            # Process complete frames
            output_frames = []
            while len(self._input_buffer) >= self._frame_size:
                frame_data = self._input_buffer[: self._frame_size]
                self._input_buffer = self._input_buffer[self._frame_size :]

                # Process frame through RNNoise
                processed_frame = self._process_single_frame(frame_data)
                if processed_frame is not None and len(processed_frame) > 0:
                    output_frames.append(processed_frame)

            if output_frames:
                # Concatenate all processed frames
                output_audio = np.concatenate(output_frames)

                # Round and clip back to int16 range at 48 kHz.
                output_audio_int16_48k = np.clip(np.rint(output_audio), -32768, 32767).astype(
                    np.int16
                )

                # -------------------------------------------------------------
                # Down-sample back to the original input rate by simple
                # decimation (take every n-th sample).
                # -------------------------------------------------------------
                if self._upsample_factor > 1:
                    output_audio_int16 = output_audio_int16_48k[:: self._upsample_factor]
                else:
                    output_audio_int16 = output_audio_int16_48k

                return output_audio_int16.tobytes()
            else:
                # Not enough data for a complete frame yet
                return b""

        except Exception as e:
            logger.error(f"Error in RNNoise filter: {e}")
            # On error, return original audio and clear buffer to prevent future issues
            self._input_buffer = np.array([], dtype=np.float32)
            return audio

    def _process_single_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """Process a single frame through RNNoise.

        Args:
            frame_data: Audio frame as float32 numpy array (frame_size samples)

        Returns:
            Processed frame as float32 numpy array
        """
        try:
            # Ensure frame is exactly the right size
            if len(frame_data) != self._frame_size:
                logger.warning(
                    f"Frame size mismatch: got {len(frame_data)}, expected {self._frame_size}"
                )
                # Pad or truncate to correct size
                if len(frame_data) < self._frame_size:
                    frame_data = np.pad(frame_data, (0, self._frame_size - len(frame_data)))
                else:
                    frame_data = frame_data[: self._frame_size]

            # Validate state before processing
            if not self._state:
                logger.error("RNNoise state is None, cannot process frame")
                return frame_data  # Return unprocessed data

            # Create ctypes arrays
            input_array = (c_float * self._frame_size)(*frame_data)
            output_array = (c_float * self._frame_size)()

            # Process through RNNoise
            vad_prob = self._rnnoise.process_frame(
                self._state,
                ctypes.cast(input_array, POINTER(c_float)),
                ctypes.cast(output_array, POINTER(c_float)),
            )

            # Convert back to numpy array
            processed_data = np.array(output_array[:], dtype=np.float32)

            return processed_data

        except Exception as e:
            logger.error(f"Error processing frame through RNNoise: {e}")
            # Return unprocessed frame on error
            return frame_data
