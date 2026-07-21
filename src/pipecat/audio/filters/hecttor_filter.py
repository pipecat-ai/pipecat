#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hecttor audio denoising filter for Pipecat.

This module provides an audio filter implementation using the Hecttor SDK
to enhance speech audio in real time. It uses the ASR-optimized enhancer,
which is tuned to improve intelligibility for speech recognition pipelines.
"""

import os

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    from hecttor_sdk import ASRSpeechEnhancer, ASRSpeechEnhancerConfig, ModelConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use HecttorFilter, you need to install the hecttor_sdk package. "
        "Contact Hecttor (https://hecttor.ai) for SDK access and an API key."
    )
    raise ImportError(f"Missing module: {e}") from e


#: ASR-optimized models exposed by the Hecttor SDK.
SUPPORTED_MODELS = ("crest-1.0", "crest-2.0", "mist-1.0", "coda-1.0", "coda-vi-1.0")

#: Models that only support a 20 ms chunk size.
_MODELS_REQUIRING_20MS = ("crest-2.0", "coda-1.0", "coda-vi-1.0")


class HecttorFilter(BaseAudioFilter):
    """Audio filter using the Hecttor SDK for speech enhancement.

    Provides real-time audio denoising for audio streams using Hecttor's
    neural network-based processing. Uses the ASR-optimized enhancer, which
    removes background noise to improve transcription accuracy before audio
    reaches an STT service.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "coda-vi-1.0",
        chunk_size_ms: int = 20,
        enhancer_weight: float | None = None,
    ) -> None:
        """Initialize the Hecttor audio filter.

        Args:
            api_key: Hecttor API key. If None, uses the HECTTOR_API_KEY
                environment variable.
            model_name: ASR enhancement model to use. One of
                ``"crest-1.0"``, ``"crest-2.0"``, ``"mist-1.0"``,
                ``"coda-1.0"``, or ``"coda-vi-1.0"``.
            chunk_size_ms: Chunk size in milliseconds. Either 16 or 20.
                ``"crest-2.0"``, ``"coda-1.0"``, and ``"coda-vi-1.0"``
                require 20.
            enhancer_weight: Blend factor between original and enhanced
                audio in the range [0.0, 1.0]. 1.0 = fully enhanced,
                0.0 = original audio. If None, the model's default weight
                is used.

        Raises:
            ValueError: If api_key is not provided and HECTTOR_API_KEY is not
                set, if model_name is unsupported, chunk_size_ms is not 16 or
                20 (or is unsupported by the chosen model), or enhancer_weight
                is out of range.
        """
        super().__init__()

        self._api_key = api_key or os.getenv("HECTTOR_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Hecttor API key must be provided via api_key parameter "
                "or HECTTOR_API_KEY environment variable."
            )

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model_name '{model_name}'. Must be one of {SUPPORTED_MODELS}."
            )

        if chunk_size_ms not in (16, 20):
            raise ValueError(f"chunk_size_ms must be 16 or 20, got {chunk_size_ms}.")

        if model_name in _MODELS_REQUIRING_20MS and chunk_size_ms != 20:
            raise ValueError(
                f"Model '{model_name}' requires chunk_size_ms=20, got {chunk_size_ms}."
            )

        if enhancer_weight is not None and not 0.0 <= enhancer_weight <= 1.0:
            raise ValueError(f"enhancer_weight must be between 0.0 and 1.0, got {enhancer_weight}.")

        self._model_name = model_name
        self._chunk_size_ms = chunk_size_ms
        self._enhancer_weight = enhancer_weight

        self._enhancer = None
        self._samples_per_chunk = 0
        self._audio_buffer = bytearray()
        self._filtering = True
        self._first_chunk = True

    async def start(self, sample_rate: int):
        """Initialize the Hecttor enhancer with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.

        Raises:
            RuntimeError: If Hecttor initialization fails.
        """
        try:
            self._enhancer = ASRSpeechEnhancer()
            config = ASRSpeechEnhancerConfig(
                api_key=self._api_key,
                model_config=ModelConfig(
                    model_name=self._model_name,
                    enhancer_weight=self._enhancer_weight,
                ),
                chunk_size_ms=self._chunk_size_ms,
                sample_rate=sample_rate,
            )
            success, message = self._enhancer.initialize(config)

            if not success:
                self._enhancer = None
                raise RuntimeError(f"Hecttor initialization failed: {message}")

            self._samples_per_chunk = self._enhancer.get_chunk_size_samples()
            self._audio_buffer.clear()
            self._first_chunk = True

            logger.debug(
                f"Hecttor filter started: model={self._model_name}, "
                f"sample_rate={sample_rate}, "
                f"chunk_size={self._samples_per_chunk} samples "
                f"({self._chunk_size_ms}ms)"
            )
        except Exception as e:
            self._enhancer = None
            logger.error(f"Failed to start Hecttor filter: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start Hecttor filter: {e}") from e

    async def stop(self):
        """Clean up the Hecttor enhancer when stopping."""
        try:
            if self._enhancer is not None:
                self._enhancer.reset_caches()
            self._enhancer = None
            self._audio_buffer.clear()
            self._first_chunk = True
        except Exception as e:
            logger.error(f"Error stopping Hecttor filter: {e}", exc_info=True)
            raise RuntimeError(f"Failed to stop Hecttor filter: {e}") from e

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Apply Hecttor speech enhancement to audio data.

        Args:
            audio: Raw PCM int16 audio data as bytes.

        Returns:
            Enhanced audio data as bytes, or empty bytes while buffering.
        """
        if not self._filtering:
            return audio

        if self._enhancer is None:
            return audio

        try:
            self._audio_buffer.extend(audio)

            bytes_per_sample = 2  # int16
            total_samples = len(self._audio_buffer) // bytes_per_sample
            num_complete_chunks = total_samples // self._samples_per_chunk

            if num_complete_chunks == 0:
                return b""

            bytes_per_chunk = self._samples_per_chunk * bytes_per_sample
            total_bytes = num_complete_chunks * bytes_per_chunk
            audio_to_process = bytes(self._audio_buffer[:total_bytes])
            self._audio_buffer = self._audio_buffer[total_bytes:]

            output_chunks = []
            for i in range(num_complete_chunks):
                start = i * bytes_per_chunk
                chunk_bytes = audio_to_process[start : start + bytes_per_chunk]

                samples_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32) / 32768.0

                enhanced, message = self._enhancer.process_chunk(samples_float32)

                if enhanced is None:
                    if self._first_chunk:
                        logger.debug("Hecttor filter: initial buffering (first chunk)")
                        self._first_chunk = False
                    continue

                enhanced_int16 = (np.clip(enhanced, -1.0, 1.0) * 32767).astype(np.int16)
                output_chunks.append(enhanced_int16.tobytes())

            if not output_chunks:
                return b""

            return b"".join(output_chunks)

        except Exception as e:
            logger.error(f"Error during Hecttor filtering: {e}", exc_info=True)
            return audio
