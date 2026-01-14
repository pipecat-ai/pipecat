#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ai-coustics AIC SDK audio filter for Pipecat.

This module provides audio filter implementations using ai-coustics' AIC SDK to
enhance audio streams in real time. It mirrors the structure of other filters like
the Koala filter and integrates with Pipecat's input transport pipeline.

Classes:
    AICFilter: For aic-sdk < 2.0.0 (uses 'aic' module)
    AICFilterV2: For aic-sdk >= 2.0.0 (uses 'aic_sdk' module)
"""

import os
from typing import List, Optional

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame


class AICFilter(BaseAudioFilter):
    """Audio filter using ai-coustics' AIC SDK for real-time enhancement.

    Buffers incoming audio to the model's preferred block size and processes
    planar frames in-place using float32 samples in the linear -1..+1 range.

    .. note::
        This class requires aic-sdk < 2.0.0 (uses 'aic' module).
        For aic-sdk >= 2.0.0, use :class:`AICFilterV2` instead.
    """

    def __init__(
        self,
        *,
        license_key: str = "",
        model_type: Optional["AICModelType"] = None,
        enhancement_level: Optional[float] = 1.0,
        voice_gain: Optional[float] = 1.0,
        noise_gate_enable: Optional[bool] = True,
    ) -> None:
        """Initialize the AIC filter.

        Args:
            license_key: ai-coustics license key for authentication.
            model_type: Model variant to load. If None, defaults to AICModelType.QUAIL_STT.
            enhancement_level: Optional overall enhancement strength (0.0..1.0).
            voice_gain: Optional linear gain applied to detected speech (0.0..4.0).
            noise_gate_enable: Optional enable/disable noise gate (default: True).

                .. deprecated:: 1.3.0
                    The `noise_gate_enable` parameter is deprecated and no longer has any effect.
                    It will be removed in a future version.
        """
        from pipecat.audio.utils import check_aic_sdk_version

        check_aic_sdk_version("v1")

        # Import AIC SDK v1 types
        from aic import AICModelType

        self._license_key = license_key
        self._model_type = model_type if model_type is not None else AICModelType.QUAIL_STT

        self._enhancement_level = enhancement_level
        self._voice_gain = voice_gain
        if noise_gate_enable is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter `noise_gate_enable` is deprecated and no longer has any effect. "
                    "It will be removed in a future version. Use AIC VAD instead (create_vad_analyzer()).",
                    DeprecationWarning,
                )

        self._noise_gate_enable = noise_gate_enable

        self._enabled = True
        self._sample_rate = 0
        self._aic_ready = False
        self._frames_per_block = 0
        self._audio_buffer = bytearray()
        # Model will be created in start() since the API now requires sample_rate
        self._aic = None

    def get_vad_factory(self):
        """Return a zero-arg factory that will create the VAD once the model exists.

        Returns:
            A zero-argument callable that, when invoked, returns an initialized
            VoiceActivityDetector bound to the underlying AIC model. Raises a
            RuntimeError if the model has not been initialized (i.e. start()
            has not been called successfully).
        """

        def _factory():
            if self._aic is None:
                raise RuntimeError("AIC model not initialized yet. Call start(sample_rate) first.")
            return self._aic.create_vad()

        return _factory

    def create_vad_analyzer(
        self,
        *,
        lookback_buffer_size: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ):
        """Return an analyzer that will lazily instantiate the AIC VAD when ready.

        AIC VAD parameters:
          - lookback_buffer_size:
              Number of window-length audio buffers used as a lookback buffer.
              Higher values increase prediction stability but add latency.
              Range: 1.0 .. 20.0, Default (SDK): 6.0
          - sensitivity:
              Energy threshold sensitivity. Energy threshold = 10 ** (-sensitivity).
              Range: 1.0 .. 15.0, Default (SDK): 6.0

        Args:
            lookback_buffer_size: Optional lookback buffer size to configure on the VAD.
                Range: 1.0 .. 20.0. If None, SDK default is used.
            sensitivity: Optional sensitivity (energy threshold) to configure on the VAD.
                Range: 1.0 .. 15.0. If None, SDK default is used.

        Returns:
            A lazily-initialized AICVADAnalyzer that will bind to the VAD backend
            once the filter's model has been created (after start(sample_rate)).
        """
        from pipecat.audio.vad.aic_vad import AICVADAnalyzer

        return AICVADAnalyzer(
            vad_factory=self.get_vad_factory(),
            lookback_buffer_size=lookback_buffer_size,
            sensitivity=sensitivity,
        )

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.

        Returns:
            None
        """
        from aic import AICParameter, Model

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
            from aic import AICParameter

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
            out_f32 = await self._aic.process_async(block_f32)

            # Convert back to int16 bytes, planar layout
            out_i16 = np.clip(out_f32 * 32768.0, -32768, 32767).astype(np.int16)
            filtered_chunks.append(out_i16.reshape(-1).tobytes())

            # Slide buffer
            self._audio_buffer = self._audio_buffer[bytes_to_consume:]
            available_frames = len(self._audio_buffer) // 2

        # Do not flush incomplete frames; keep them buffered for the next call
        return b"".join(filtered_chunks)


class AICFilterV2(BaseAudioFilter):
    """Audio filter using ai-coustics' AIC SDK v2 for real-time enhancement.

    Buffers incoming audio to the model's preferred block size and processes
    planar frames in-place using float32 samples in the linear -1..+1 range.

    .. note::
        This class requires aic-sdk >= 2.0.0 (uses 'aic_sdk' module).
        For aic-sdk < 2.0.0, use :class:`AICFilter` instead.
    """

    def __init__(
        self,
        *,
        license_key: str = "",
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        model_download_dir: Optional[str] = None,
        enhancement_level: Optional[float] = 1.0,
        voice_gain: Optional[float] = 1.0,
    ) -> None:
        """Initialize the AIC filter.

        Args:
            license_key: ai-coustics license key for authentication.
            model_id: Model identifier to download from CDN. Required if model_path
                is not provided. See https://artifacts.ai-coustics.io/ for available models.
            model_path: Optional path to a local .aicmodel file. If provided,
                model_id is ignored and no download occurs.
            model_download_dir: Directory for downloading models. Defaults to
                a cache directory in user's home folder.
            enhancement_level: Optional overall enhancement strength (0.0..1.0).
            voice_gain: Optional linear gain applied to detected speech (0.1..4.0).

        Raises:
            ValueError: If neither model_id nor model_path is provided.
        """
        from pipecat.audio.utils import check_aic_sdk_version

        check_aic_sdk_version("v2")

        if model_id is None and model_path is None:
            raise ValueError(
                "Either 'model_id' or 'model_path' must be provided. "
                "See https://artifacts.ai-coustics.io/ for available models."
            )

        self._license_key = license_key
        self._model_id = model_id
        self._model_path = model_path
        self._model_download_dir = model_download_dir or os.path.expanduser(
            "~/.cache/pipecat/aic-models"
        )

        self._enhancement_level = enhancement_level
        self._voice_gain = voice_gain

        self._enabled = True
        self._sample_rate = 0
        self._aic_ready = False
        self._frames_per_block = 0
        self._audio_buffer = bytearray()

        # v2 API objects
        self._model = None
        self._processor = None
        self._processor_ctx = None
        self._vad_ctx = None

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
        sensitivity: Optional[float] = None,
    ):
        """Return an analyzer that will lazily instantiate the AIC VAD when ready.

        AIC VAD parameters (v2):
          - speech_hold_duration:
              How long VAD continues detecting after speech ends (in seconds).
              Range: 0.0 .. 20x model window length, Default (SDK): 0.05s
          - sensitivity:
              Energy threshold sensitivity. Energy threshold = 10 ** (-sensitivity).
              Range: 1.0 .. 15.0, Default (SDK): 6.0

        Args:
            speech_hold_duration: Optional speech hold duration to configure on the VAD.
                If None, SDK default (0.05s) is used.
            sensitivity: Optional sensitivity (energy threshold) to configure on the VAD.
                Range: 1.0 .. 15.0. If None, SDK default (6.0) is used.

        Returns:
            A lazily-initialized AICVADAnalyzerV2 that will bind to the VAD context
            once the filter's processor has been created (after start(sample_rate)).
        """
        from pipecat.audio.vad.aic_vad import AICVADAnalyzerV2

        return AICVADAnalyzerV2(
            vad_context_factory=lambda: self.get_vad_context(),
            speech_hold_duration=speech_hold_duration,
            sensitivity=sensitivity,
        )

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.

        Returns:
            None
        """
        from aic_sdk import Model, ProcessorAsync, ProcessorConfig, ProcessorParameter

        self._sample_rate = sample_rate

        try:
            # Load or download model
            if self._model_path:
                logger.debug(f"Loading AIC model from: {self._model_path}")
                self._model = Model.from_file(self._model_path)
            else:
                logger.debug(f"Downloading AIC model: {self._model_id}")
                os.makedirs(self._model_download_dir, exist_ok=True)
                model_path = await Model.download_async(self._model_id, self._model_download_dir)
                logger.debug(f"Model downloaded to: {model_path}")
                self._model = Model.from_file(model_path)

            # Create async processor
            self._processor = ProcessorAsync(self._model, self._license_key or "")

            # Get optimal frames for this sample rate
            self._frames_per_block = self._model.get_optimal_num_frames(self._sample_rate)

            # Create configuration
            config = ProcessorConfig(
                sample_rate=self._sample_rate,
                num_channels=1,
                num_frames=self._frames_per_block,
                allow_variable_frames=False,
            )

            # Initialize processor
            await self._processor.initialize_async(config)

            # Get contexts for parameter control and VAD
            self._processor_ctx = self._processor.get_processor_context()
            self._vad_ctx = self._processor.get_vad_context()

            # Apply initial parameters
            if self._enhancement_level is not None:
                level = float(self._enhancement_level if self._enabled else 0.0)
                self._processor_ctx.set_parameter(ProcessorParameter.EnhancementLevel, level)
            if self._voice_gain is not None:
                self._processor_ctx.set_parameter(
                    ProcessorParameter.VoiceGain, float(self._voice_gain)
                )

            self._aic_ready = True

            # Log processor information
            logger.debug(f"ai-coustics filter (v2) started:")
            logger.debug(f"  Model ID: {self._model.get_id()}")
            logger.debug(f"  Sample rate: {self._sample_rate} Hz")
            logger.debug(f"  Frames per chunk: {self._frames_per_block}")
            logger.debug(f"  Enhancement strength: {int((self._enhancement_level or 1.0) * 100)}%")
            logger.debug(f"  Optimal sample rate: {self._model.get_optimal_sample_rate()} Hz")
            logger.debug(
                f"  Output delay: {self._processor_ctx.get_output_delay()} samples "
                f"({self._processor_ctx.get_output_delay() / self._sample_rate * 1000:.2f}ms)"
            )
        except Exception as e:  # noqa: BLE001 - surfacing SDK initialization errors
            logger.error(f"AIC model initialization failed: {e}")
            self._aic_ready = False

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
            from aic_sdk import ProcessorParameter

            self._enabled = frame.enable
            if self._processor_ctx is not None:
                try:
                    level = float(self._enhancement_level if self._enabled else 0.0)
                    self._processor_ctx.set_parameter(ProcessorParameter.EnhancementLevel, level)
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
        if not self._aic_ready or self._processor is None:
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

            # Convert to float32 in -1..+1 range and reshape to (channels, frames)
            block_i16 = np.frombuffer(block_bytes, dtype=np.int16)
            block_f32 = (block_i16.astype(np.float32) / 32768.0).reshape(
                (1, self._frames_per_block)
            )

            # Process via async processor; returns ndarray (same shape)
            out_f32 = await self._processor.process_async(block_f32)

            # Convert back to int16 bytes
            out_i16 = np.clip(out_f32 * 32768.0, -32768, 32767).astype(np.int16)
            filtered_chunks.append(out_i16.reshape(-1).tobytes())

            # Slide buffer
            self._audio_buffer = self._audio_buffer[bytes_to_consume:]
            available_frames = len(self._audio_buffer) // 2

        # Do not flush incomplete frames; keep them buffered for the next call
        return b"".join(filtered_chunks)
