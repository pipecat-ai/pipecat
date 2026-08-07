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
    AICModelManager: Singleton manager for read-only AIC Model instances.
"""

import asyncio
from pathlib import Path
from threading import Lock

import numpy as np
from aic_sdk import (
    Model,
    ParameterOutOfRangeError,
    ProcessorAsync,
    ProcessorConfig,
    ProcessorParameter,
    VadAsync,
    VadContext,
    set_sdk_id,
)
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

# Telemetry identifier registered with the AIC SDK; identifies pipecat to the
# vendor's usage pipeline. Kept private (leading underscore) to avoid making it
# accidental public API.
_AIC_SDK_PIPECAT_ID = 6


class AICModelManager:
    """Singleton manager for read-only AIC Model instances with reference counting.

    Caches Model instances by path or (model_id + download_dir). Multiple
    AICFilter instances using the same model share one Model; the manager
    acquires on first use and releases when the last reference is dropped.
    """

    _cache: dict[str, tuple[Model, int]] = {}  # key -> (model, ref_count)
    _lock = Lock()
    _loading: dict[
        str, asyncio.Task[Model]
    ] = {}  # key -> load task (deduplicates concurrent loads)

    @classmethod
    def _increment_reference(cls, cache_key: str, entry: tuple[Model, int]) -> tuple[Model, str]:
        """Increment reference count for cached entry. Caller must hold _lock."""
        cached_model, ref_count = entry
        cls._cache[cache_key] = (cached_model, ref_count + 1)
        logger.debug(f"AIC model cache key={cache_key!r} ref_count={ref_count + 1}")
        return cached_model, cache_key

    @classmethod
    def _store_new_reference(cls, cache_key: str, model: Model) -> tuple[Model, str]:
        """Store new model in cache with ref count 1. Caller must hold _lock."""
        cls._cache[cache_key] = (model, 1)
        logger.debug(f"AIC model cached key={cache_key!r} ref_count=1")
        return model, cache_key

    @classmethod
    async def _load_model_from_file(
        cls,
        cache_key: str,
        *,
        model_path: Path | None = None,
        model_id: str | None = None,
        model_download_dir: Path | None = None,
    ) -> Model:
        """Run the actual load (file or download). Separate to allow create_task and deduplication."""
        if model_path is not None:
            logger.debug(f"Loading AIC model from file: {model_path}")
            model_path_str = str(model_path)

        elif model_id is not None and model_download_dir is not None:
            logger.debug(f"Downloading AIC model: {model_id}")
            model_download_dir.mkdir(parents=True, exist_ok=True)
            model_path_str = await Model.download_async(model_id, str(model_download_dir))
            logger.debug(f"Model downloaded to: {model_path_str}")

        else:
            raise ValueError("Unexpected model_path or (model_id and model_download_dir) state.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: Model.from_file(model_path_str))

    @staticmethod
    def _get_cache_key(
        *,
        model_path: Path | None = None,
        model_id: str | None = None,
        model_download_dir: Path | None = None,
    ) -> str:
        """Build a stable cache key for the model.

        Args:
            model_path: Path to a local .aicmodel file.
            model_id: Model identifier (See https://artifacts.ai-coustics.io/ for available models).
            model_download_dir: Directory used for downloading models.

        Returns:
            A string key unique per (path) or (model_id + download_dir).
        """
        if model_path is not None:
            return f"path:{model_path.resolve()}"

        if model_id is not None and model_download_dir is not None:
            return f"id:{model_id}:{model_download_dir.resolve()}"

        raise ValueError("Either model_path or (model_id and model_download_dir) must be set.")

    @classmethod
    async def acquire(
        cls,
        *,
        model_path: Path | None = None,
        model_id: str | None = None,
        model_download_dir: Path | None = None,
    ) -> tuple[Model, str]:
        """Get or load a Model and increment its reference count.

        Call this when starting a filter. Store the returned key and pass it
        to release() when stopping the filter.

        Args:
            model_path: Path to a local .aicmodel file. If set, model_id is ignored.
            model_id: Model identifier to download from CDN.
            model_download_dir: Directory for downloading models. Required if
                model_id is used.

        Returns:
            Tuple of (shared Model instance, cache key for release).

        Raises:
            ValueError: If neither model_path nor (model_id + model_download_dir)
                is provided, or if model_id is set without model_download_dir.
        """
        cache_key = cls._get_cache_key(
            model_path=model_path,
            model_id=model_id,
            model_download_dir=model_download_dir,
        )

        with cls._lock:
            entry = cls._cache.get(cache_key)
            if entry is not None:
                return cls._increment_reference(cache_key, entry)

            # Deduplicate concurrent loads for the same key
            load_task = cls._loading.get(cache_key)
            if load_task is None:
                load_task = asyncio.create_task(
                    cls._load_model_from_file(
                        cache_key,
                        model_path=model_path,
                        model_id=model_id,
                        model_download_dir=model_download_dir,
                    )
                )
                cls._loading[cache_key] = load_task

        try:
            model = await load_task
        finally:
            with cls._lock:
                cls._loading.pop(cache_key, None)

        with cls._lock:
            entry = cls._cache.get(cache_key)
            if entry is not None:
                return cls._increment_reference(cache_key, entry)
            return cls._store_new_reference(cache_key, model)

    @classmethod
    def release(cls, key: str) -> None:
        """Release a reference to a cached model.

        Call this when stopping a filter, with the key returned from
        get_model(). When the last reference is released, the model
        is removed from the cache.

        Args:
            key: Cache key returned by get_model().
        """
        with cls._lock:
            entry = cls._cache.get(key)

            if entry is None:
                logger.warning(f"AIC model release unknown key={key!r}")
                return

            model, ref_count = entry
            ref_count -= 1

            if ref_count <= 0:
                del cls._cache[key]
                logger.debug(f"AIC model evicted key={key!r}")
            else:
                cls._cache[key] = (model, ref_count)
                logger.debug(f"AIC model key={key!r} ref_count={ref_count}")


class AICFilter(BaseAudioFilter):
    """Audio filter using ai-coustics' AIC SDK for real-time enhancement.

    Buffers incoming audio to the model's preferred block size and processes
    frames using float32 samples normalized to the range -1 to +1.

    The filter can additionally run a dedicated VAD model. Because the filter
    is the only component that sees audio before enhancement, the VAD runs on
    the original block rather than the enhanced output — the ordering the AIC
    SDK requires when enhancement and detection are used together. Pair it with
    :class:`pipecat.audio.vad.aic_filter_vad.AICFilterVADAnalyzer` to surface
    those predictions to the pipeline.
    """

    def __init__(
        self,
        *,
        license_key: str,
        model_id: str | None = None,
        model_path: Path | None = None,
        model_download_dir: Path | None = None,
        enhancement_level: float | None = None,
        vad_model_id: str | None = None,
        vad_model_path: Path | None = None,
    ) -> None:
        """Initialize the AIC filter.

        Args:
            license_key: ai-coustics license key for authentication.
            model_id: Model identifier to download from CDN. Required if model_path
                is not provided. See https://artifacts.ai-coustics.io/ for available models.
            model_path: Optional path to a local .aicmodel file. If provided,
                model_id is ignored and no download occurs.
            model_download_dir: Directory for downloading models as a Path object.
                Defaults to a cache directory in user's home folder. Shared by the
                enhancement model and the VAD model.
            enhancement_level: Optional overall enhancement strength (0.0..1.0).
                If None, the model default is used.
            vad_model_id: Optional dedicated VAD model identifier to download from
                CDN, e.g. ``"vad-2.1-xxs-16khz"``. When set, the filter runs that
                VAD on pre-enhancement audio and exposes it via
                :meth:`get_vad_context`.
            vad_model_path: Optional path to a local VAD ``.aicmodel`` file. If
                provided, vad_model_id is ignored.

        Raises:
            ValueError: If neither model_id nor model_path is provided, or if
                enhancement_level is out of range.
        """
        # Set SDK ID for telemetry identification.
        set_sdk_id(_AIC_SDK_PIPECAT_ID)

        if model_id is None and model_path is None:
            raise ValueError(
                "Either 'model_id' or 'model_path' must be provided. "
                "See https://artifacts.ai-coustics.io/ for available models."
            )

        if enhancement_level is not None and not 0.0 <= enhancement_level <= 1.0:
            raise ValueError("'enhancement_level' must be between 0.0 and 1.0.")

        self._license_key = license_key
        self._model_id = model_id
        self._model_path = model_path
        self._model_download_dir = model_download_dir or (
            Path.home() / ".cache" / "pipecat" / "aic-models"
        )
        self._enhancement_level = enhancement_level
        self._vad_model_id = vad_model_id
        self._vad_model_path = vad_model_path
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

        # AIC SDK objects; models are shared via AICModelManager
        self._model_cache_key: str | None = None
        self._model = None
        self._processor = None
        self._processor_ctx = None
        self._vad_model_cache_key: str | None = None
        self._vad_model = None
        self._vad = None
        self._vad_ctx = None

        # Pre-allocated buffers (resized in start() once frames_per_block is known)
        self._in_f32 = None
        self._out_i16 = None

    @property
    def has_vad_model(self) -> bool:
        """Whether the filter was configured to run a dedicated VAD."""
        return self._vad_model_id is not None or self._vad_model_path is not None

    @property
    def frames_per_block(self) -> int:
        """Number of samples the filter feeds to the SDK per processing call.

        Returns:
            The block size in frames, or 0 before :meth:`start` has run.
        """
        return self._frames_per_block

    def get_vad_context(self) -> VadContext:
        """Return the context of the filter's dedicated VAD.

        The VAD advances on pre-enhancement audio, so its predictions describe
        the original signal rather than the enhanced output.

        Returns:
            The VadContext bound to the filter's VAD.

        Raises:
            ValueError: If the filter was constructed without a VAD model.
            RuntimeError: If the filter has not been started yet.
        """
        if self._vad_model_id is None and self._vad_model_path is None:
            raise ValueError(
                "AICFilter has no VAD model. Pass 'vad_model_id' or 'vad_model_path' "
                "to enable voice activity detection."
            )
        if self._vad_ctx is None:
            raise RuntimeError("AIC VAD not initialized yet. Call start(sample_rate) first.")
        return self._vad_ctx

    def _apply_enhancement_level(self):
        """Apply enhancement_level if configured and supported by the active model."""
        if self._processor_ctx is None or self._enhancement_level is None:
            return

        try:
            self._processor_ctx.set_parameter(
                ProcessorParameter.EnhancementLevel, self._enhancement_level
            )
        except ParameterOutOfRangeError as e:
            logger.warning(f"AIC EnhancementLevel set_parameter out-of-range: {e}")
            self._enhancement_level = None

    def _apply_bypass(self):
        """Apply bypass parameter to the active processor."""
        if self._processor_ctx is None:
            return

        self._processor_ctx.set_parameter(ProcessorParameter.Bypass, 1.0 if self._bypass else 0.0)

    async def _start_vad(self):
        """Create the dedicated VAD, sized to the enhancement block.

        The VAD shares the filter's block size so both objects can be fed the
        same original buffer, as the SDK requires.
        """
        if self._vad_model_id is None and self._vad_model_path is None:
            return

        self._vad_model, self._vad_model_cache_key = await AICModelManager.acquire(
            model_path=self._vad_model_path,
            model_id=self._vad_model_id,
            model_download_dir=self._model_download_dir,
        )

        vad_config = ProcessorConfig(
            sample_rate=self._sample_rate,
            block_size=self._frames_per_block,
        )
        self._vad = VadAsync(self._vad_model, self._license_key, vad_config)
        self._vad_ctx = self._vad.get_context()

        logger.debug(
            f"  VAD model: {self._vad_model.get_id()}, "
            f"prediction delay: {self._vad_ctx.get_prediction_delay()} samples"
        )

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.

        Returns:
            None
        """
        self._sample_rate = sample_rate

        # Acquire shared read-only model from singleton manager
        self._model, self._model_cache_key = await AICModelManager.acquire(
            model_path=self._model_path,
            model_id=self._model_id,
            model_download_dir=self._model_download_dir,
        )

        # Get optimal block size for this sample rate
        self._frames_per_block = self._model.get_optimal_block_size(self._sample_rate)

        # Allocate processing buffers now that we know the block size
        self._in_f32 = np.zeros(self._frames_per_block, dtype=np.float32)
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

        # Get context for parameter control
        self._processor_ctx = self._processor.get_context()

        # Apply initial control parameters
        self._apply_bypass()
        self._apply_enhancement_level()

        # Log processor information
        logger.debug(f"ai-coustics filter started:")
        logger.debug(f"  Model ID: {self._model.get_id()}")
        logger.debug(f"  Sample rate: {self._sample_rate} Hz")
        logger.debug(f"  Frames per chunk: {self._frames_per_block}")
        if self._enhancement_level is not None:
            logger.debug(f"  Enhancement level: {self._enhancement_level}")
        else:
            logger.debug("  Enhancement level not configured; using the model's default behavior.")
        logger.debug(f"  Optimal sample rate: {self._model.get_optimal_sample_rate()} Hz")
        logger.debug(
            f"  Optimal block size for {self._sample_rate} Hz: "
            f"{self._model.get_optimal_block_size(self._sample_rate)}"
        )
        logger.debug(
            f"  Audio delay: {self._processor_ctx.get_audio_delay()} samples "
            f"({self._processor_ctx.get_audio_delay() / self._sample_rate * 1000:.2f}ms)"
        )

        await self._start_vad()

    async def stop(self):
        """Terminate the AIC sessions and release the models when stopping.

        Returns:
            None
        """
        # Terminate independently so one failure doesn't strand the other session.
        for label, obj in (("VAD", self._vad), ("processor", self._processor)):
            if obj is None:
                continue
            try:
                await obj.terminate_session_async()
            except Exception as e:  # noqa: BLE001 - teardown is best-effort
                logger.debug(f"AIC {label} session termination failed: {e}")

        self._processor = None
        self._processor_ctx = None
        self._vad = None
        self._vad_ctx = None
        self._model = None
        self._vad_model = None
        self._aic_ready = False
        self._audio_buffer.clear()

        if self._model_cache_key is not None:
            AICModelManager.release(self._model_cache_key)
            self._model_cache_key = None

        if self._vad_model_cache_key is not None:
            AICModelManager.release(self._vad_model_cache_key)
            self._vad_model_cache_key = None

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
                    self._apply_bypass()
                    self._apply_enhancement_level()
                except Exception as e:  # noqa: BLE001
                    logger.error(f"AIC set_parameter failed: {e}")

    async def filter(self, audio: bytes) -> bytes:
        """Apply AIC enhancement to audio data.

        Buffers incoming audio and processes it in chunks that match the AIC
        model's required block length. Returns enhanced audio data.

        When a VAD model is configured, each block advances the VAD before it is
        enhanced, so predictions describe the original signal.

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

        block_size = self._frames_per_block * self._bytes_per_sample
        total_size = num_blocks * block_size
        blocks_data = bytes(self._audio_buffer[:total_size])
        self._audio_buffer = self._audio_buffer[total_size:]

        filtered_chunks: list[bytes] = []

        for i in range(num_blocks):
            start = i * block_size
            block_i16 = np.frombuffer(blocks_data[start : start + block_size], dtype=self._dtype)

            # Reuse input buffer, in-place divide
            np.copyto(self._in_f32, block_i16)
            self._in_f32 /= self._scale

            # VAD first: it reads the original block without modifying it, so
            # the prediction describes the unenhanced signal.
            if self._vad is not None:
                await self._vad.process_async(self._in_f32)

            out_f32 = await self._processor.process_async(self._in_f32)

            # Convert float32 output back to int16
            np.multiply(out_f32, self._scale, out=self._in_f32)  # reuse in_f32 as temp
            np.clip(self._in_f32, -self._scale, self._scale - 1, out=self._in_f32)
            np.copyto(self._out_i16, self._in_f32.astype(self._dtype))

            filtered_chunks.append(self._out_i16.tobytes())

        return b"".join(filtered_chunks)
