#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ai-coustics AIC SDK audio filter for Pipecat.

This module provides an audio filter implementation using ai-coustics' AIC SDK to
enhance audio streams in real time. It mirrors the structure of other filters like
the Koala filter and integrates with Pipecat's input transport pipeline.
"""

from typing import List, Optional

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    # AIC SDK (https://ai-coustics.github.io/aic-sdk-py/api/)
    from aic import AICModelType, AICParameter, Model
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the AIC filter, you need to `pip install pipecat-ai[aic]`.")
    raise Exception(f"Missing module: {e}")


class AICFilter(BaseAudioFilter):
    """Audio filter using ai-coustics' AIC SDK for real-time enhancement.

    Buffers incoming audio to the model's preferred block size and processes
    planar frames in-place using float32 samples in the linear -1..+1 range.
    """

    def __init__(
        self,
        *,
        license_key: str = "",
        model_type: AICModelType = AICModelType.QUAIL_L,
        enhancement_level: Optional[float] = 1.0,
        voice_gain: Optional[float] = 1.0,
        noise_gate_enable: Optional[bool] = True,
    ) -> None:
        """Initialize the AIC filter.

        Args:
            license_key: ai-coustics license key for authentication.
            model_type: Model variant to load.
            enhancement_level: Optional overall enhancement strength (0.0..1.0).
            voice_gain: Optional linear gain applied to detected speech (0.0..4.0).
            noise_gate_enable: Optional enable/disable noise gate (default: True).
        """
        self._license_key = license_key
        self._model_type = model_type

        self._enhancement_level = enhancement_level
        self._voice_gain = voice_gain
        self._noise_gate_enable = noise_gate_enable

        self._enabled = True
        self._sample_rate = 0
        self._aic_ready = False
        self._frames_per_block = 0
        self._audio_buffer = bytearray()
        # Model will be created in start() since the API now requires sample_rate
        self._aic = None

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.

        Returns:
            None
        """
        self._sample_rate = sample_rate

        try:
            # Create model with required runtime parameters
            self._aic = Model(
                model_type=self._model_type,
                license_key=self._license_key or None,
                sample_rate=self._sample_rate,
                channels=1,
            )
            self._frames_per_block = self._aic.optimal_num_frames()

            # Optional parameter configuration
            if self._enhancement_level is not None:
                self._aic.set_parameter(
                    AICParameter.ENHANCEMENT_LEVEL,
                    float(self._enhancement_level if self._enabled else 0.0),
                )
            if self._voice_gain is not None:
                self._aic.set_parameter(AICParameter.VOICE_GAIN, float(self._voice_gain))
            if self._noise_gate_enable is not None:
                self._aic.set_parameter(
                    AICParameter.NOISE_GATE_ENABLE, 1.0 if bool(self._noise_gate_enable) else 0.0
                )

            self._aic_ready = True

            # Log processor information
            logger.debug(f"ai-coustics filter started:")
            logger.debug(f"  Sample rate: {self._sample_rate} Hz")
            logger.debug(f"  Frames per chunk: {self._frames_per_block}")
            logger.debug(f"  Enhancement strength: {int(self._enhancement_level * 100)}%")
            logger.debug(f"  Optimal input buffer size: {self._aic.optimal_num_frames()} samples")
            logger.debug(f"  Optimal sample rate: {self._aic.optimal_sample_rate()} Hz")
            logger.debug(
                f"  Current algorithmic latency: {self._aic.processing_latency() / self._sample_rate * 1000:.2f}ms"
            )
        except Exception as e:  # noqa: BLE001 - surfacing SDK initialization errors
            logger.error(f"AIC model initialization failed: {e}")
            self._aic_ready = False

    async def stop(self):
        """Clean up the AIC model when stopping.

        Returns:
            None
        """
        try:
            if self._aic is not None:
                self._aic.close()
        finally:
            self._aic = None
            self._aic_ready = False
            self._audio_buffer.clear()

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.

        Returns:
            None
        """
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable
            if self._aic is not None:
                try:
                    level = float(self._enhancement_level if self._enabled else 0.0)
                    self._aic.set_parameter(AICParameter.ENHANCEMENT_LEVEL, level)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"AIC set_parameter failed: {e}")

    async def filter(self, audio: bytes) -> bytes:
        """Apply AIC enhancement to audio data.

        Buffers incoming audio and processes it in chunks that match the AIC
        model's required block length. Returns enhanced audio data.

        Args:
            audio: Raw audio data as bytes to be filtered (int16 PCM, planar).

        Returns:
            Enhanced audio data as bytes (int16 PCM, planar).
        """
        if not self._aic_ready or self._aic is None:
            return audio

        self._audio_buffer.extend(audio)

        filtered_chunks: List[bytes] = []

        # Number of int16 samples currently buffered
        available_frames = len(self._audio_buffer) // 2

        while available_frames >= self._frames_per_block:
            # Consume exactly one block worth of frames
            samples_to_consume = self._frames_per_block * 1
            bytes_to_consume = samples_to_consume * 2
            block_bytes = bytes(self._audio_buffer[:bytes_to_consume])

            # Convert to float32 in -1..+1 range and reshape to planar (channels, frames)
            block_i16 = np.frombuffer(block_bytes, dtype=np.int16)
            block_f32 = (block_i16.astype(np.float32) / 32768.0).reshape(
                (1, self._frames_per_block)
            )

            # Process planar in-place; returns ndarray (same shape)
            out_f32 = self._aic.process(block_f32)

            # Convert back to int16 bytes, planar layout
            out_i16 = np.clip(out_f32 * 32768.0, -32768, 32767).astype(np.int16)
            filtered_chunks.append(out_i16.reshape(-1).tobytes())

            # Slide buffer
            self._audio_buffer = self._audio_buffer[bytes_to_consume:]
            available_frames = len(self._audio_buffer) // 2

        # Do not flush incomplete frames; keep them buffered for the next call
        return b"".join(filtered_chunks)
