#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ai-coustics AIC SDK audio filter for Pipecat.

This module provides an audio filter implementation using ai-coustics' AIC SDK to
enhance audio streams in real time. It mirrors the structure of other filters like
the Koala filter and integrates with Pipecat's input transport pipeline.

Classes:
    AICFilter: For aic-sdk (uses 'aic_sdk' module)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from aic_sdk import (
    Model,
    ParameterFixedError,
    ProcessorAsync,
    ProcessorConfig,
    ProcessorParameter,
    set_sdk_id,
)
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.vad.aic_vad import AICVADAnalyzer
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame


class AICFilter(BaseAudioFilter):
    """Audio filter using ai-coustics' AIC SDK for real-time enhancement.

    Buffers incoming audio to the model's preferred block size and processes
    frames using float32 samples normalized to the range -1 to +1.
    """

    def __init__(
        self,
        *,
        license_key: str,
        model_id: Optional[str] = None,
        model_path: Optional[Path] = None,
        model_download_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the AIC filter.

        Args:
            license_key: ai-coustics license key for authentication.
            model_id: Model identifier to download from CDN. Required if model_path
                is not provided. See https://artifacts.ai-coustics.io/ for available models.
            model_path: Optional path to a local .aicmodel file. If provided,
                model_id is ignored and no download occurs.
            model_download_dir: Directory for downloading models as a Path object.
                Defaults to a cache directory in user's home folder.

        Raises:
            ValueError: If neither model_id nor model_path is provided.
        """
        # Set SDK ID for telemetry identification (6 = pipecat)
        set_sdk_id(6)

        if model_id is None and model_path is None:
            raise ValueError(
                "Either 'model_id' or 'model_path' must be provided. "
                "See https://artifacts.ai-coustics.io/ for available models."
            )

        self._license_key = license_key
        self._model_id = model_id
        self._model_path = model_path
        self._model_download_dir = model_download_dir or (
            Path.home() / ".cache" / "pipecat" / "aic-models"
        )

        self._bypass = False
        self._sample_rate = 0
        self._aic_ready = False
        self._frames_per_block = 0
        self._audio_buffer = bytearray()

        # Audio format constants
        self._bytes_per_sample = 2  # int16 = 2 bytes
        self._dtype = np.int16
        self._scale = (
            32768.0  # 2^15, for normalizing int16 (-32768 to 32767) to float32 (-1.0 to 1.0)
        )

        # AIC SDK objects
        self._model = None
        self._processor = None
        self._processor_ctx = None
        self._vad_ctx = None

        # Pre-allocated buffers (resized in start() once frames_per_block is known)
        self._in_f32 = None
        self._out_i16 = None

    def get_vad_context(self):
        """Return the VAD context once the processor exists.

        Returns:
            The VadContext instance bound to the underlying processor.
            Raises RuntimeError if the processor has not been initialized.
        """
        if self._vad_ctx is None:
            raise RuntimeError("AIC processor not initialized yet. Call start(sample_rate) first.")
        return self._vad_ctx

    def create_vad_analyzer(
        self,
        *,
        speech_hold_duration: Optional[float] = None,
        minimum_speech_duration: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ):
        """Return an analyzer that will lazily instantiate the AIC VAD when ready.

        AIC VAD parameters:
          - speech_hold_duration:
              How long VAD continues detecting after speech ends (in seconds).
              Range: 0.0 to 100x model window length, Default (SDK): 0.05s
          - minimum_speech_duration:
              Minimum duration of speech required before VAD reports speech detected
              (in seconds). Range: 0.0 to 1.0, Default (SDK): 0.0s
          - sensitivity:
              Energy threshold sensitivity. Energy threshold = 10 ** (-sensitivity).
              Range: 1.0 to 15.0, Default (SDK): 6.0

        Args:
            speech_hold_duration: Optional speech hold duration to configure on the VAD.
                If None, SDK default (0.05s) is used.
            minimum_speech_duration: Optional minimum speech duration before VAD reports
                speech detected. If None, SDK default (0.0s) is used.
            sensitivity: Optional sensitivity (energy threshold) to configure on the VAD.
                Range: 1.0 to 15.0. If None, SDK default (6.0) is used.

        Returns:
            A lazily-initialized AICVADAnalyzer that will bind to the VAD context
            once the filter's processor has been created (after start(sample_rate)).
        """
        return AICVADAnalyzer(
            vad_context_factory=lambda: self.get_vad_context(),
            speech_hold_duration=speech_hold_duration,
            minimum_speech_duration=minimum_speech_duration,
            sensitivity=sensitivity,
        )

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.

        Returns:
            None
        """
        self._sample_rate = sample_rate

        # Load or download model
        if self._model_path:
            logger.debug(f"Loading AIC model from: {self._model_path}")
            self._model = Model.from_file(str(self._model_path))
        else:
            logger.debug(f"Downloading AIC model: {self._model_id}")
            self._model_download_dir.mkdir(parents=True, exist_ok=True)
            model_path = await Model.download_async(self._model_id, str(self._model_download_dir))
            logger.debug(f"Model downloaded to: {model_path}")
            self._model = Model.from_file(model_path)

        # Get optimal frames for this sample rate
        self._frames_per_block = self._model.get_optimal_num_frames(self._sample_rate)

        # Allocate processing buffers now that we know the block size
        self._in_f32 = np.zeros((1, self._frames_per_block), dtype=np.float32)
        self._out_i16 = np.zeros(self._frames_per_block, dtype=np.int16)

        # Create configuration
        config = ProcessorConfig.optimal(
            self._model,
            sample_rate=self._sample_rate,
        )

        # Create async processor
        try:
            self._processor = ProcessorAsync(self._model, self._license_key, config)
        except Exception as e:  # noqa: BLE001 - surfacing SDK initialization errors
            logger.error(f"AIC model initialization failed: {e}")
            self._processor = None

        self._aic_ready = self._processor is not None

        if not self._aic_ready:
            logger.debug(f"ai-coustics filter is not ready.")
            return

        # Get contexts for parameter control and VAD
        self._processor_ctx = self._processor.get_processor_context()
        self._vad_ctx = self._processor.get_vad_context()

        # Apply initial parameters
        try:
            self._processor_ctx.set_parameter(
                ProcessorParameter.Bypass, 1.0 if self._bypass else 0.0
            )
        except ParameterFixedError as e:
            logger.error(f"AIC parameter update failed: {e}")

        # Log processor information
        logger.debug(f"ai-coustics filter started:")
        logger.debug(f"  Model ID: {self._model.get_id()}")
        logger.debug(f"  Sample rate: {self._sample_rate} Hz")
        logger.debug(f"  Frames per chunk: {self._frames_per_block}")
        logger.debug(f"  Optimal sample rate: {self._model.get_optimal_sample_rate()} Hz")
        logger.debug(
            f"  Optimal number of frames for {self._sample_rate} Hz: {self._model.get_optimal_num_frames(self._sample_rate)}"
        )
        logger.debug(
            f"  Output delay: {self._processor_ctx.get_output_delay()} samples "
            f"({self._processor_ctx.get_output_delay() / self._sample_rate * 1000:.2f}ms)"
        )

    async def stop(self):
        """Clean up the AIC processor when stopping.

        Returns:
            None
        """
        try:
            if self._processor_ctx is not None:
                self._processor_ctx.reset()
        finally:
            self._processor = None
            self._processor_ctx = None
            self._vad_ctx = None
            self._model = None
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
            self._bypass = not frame.enable
            if self._processor_ctx is not None:
                try:
                    self._processor_ctx.set_parameter(
                        ProcessorParameter.Bypass, 1.0 if self._bypass else 0.0
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"AIC set_parameter failed: {e}")

    async def filter(self, audio: bytes) -> bytes:
        """Apply AIC enhancement to audio data.

        Buffers incoming audio and processes it in chunks that match the AIC
        model's required block length. Returns enhanced audio data.

        Args:
            audio: Raw audio data as bytes (int16 PCM).

        Returns:
            Enhanced audio data as bytes (int16 PCM).
        """
        if not self._aic_ready or self._processor is None:
            return audio

        self._audio_buffer.extend(audio)
        available_frames = len(self._audio_buffer) // self._bytes_per_sample
        num_blocks = available_frames // self._frames_per_block

        if num_blocks == 0:
            return b""

        filtered_chunks: List[bytes] = []
        mv = memoryview(self._audio_buffer)
        block_size = self._frames_per_block * self._bytes_per_sample

        for i in range(num_blocks):
            start = i * block_size
            block_i16 = np.frombuffer(mv[start : start + block_size], dtype=self._dtype)

            # Reuse input buffer, in-place divide
            np.copyto(self._in_f32[0], block_i16)
            self._in_f32 /= self._scale

            out_f32 = await self._processor.process_async(self._in_f32)

            # Convert float32 output back to int16
            np.multiply(out_f32, self._scale, out=self._in_f32)  # reuse in_f32 as temp
            np.clip(self._in_f32, -self._scale, self._scale - 1, out=self._in_f32)
            np.copyto(self._out_i16, self._in_f32[0].astype(self._dtype))

            filtered_chunks.append(self._out_i16.tobytes())

        self._audio_buffer = self._audio_buffer[num_blocks * block_size :]
        return b"".join(filtered_chunks)
