#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DPDFNet speech enhancement audio filter for Pipecat.

This module provides an audio filter implementation using DPDFNet, a family of
causal dual-path deep filtering models for real-time speech enhancement, via the
dpdfnet library.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.resamplers.base_audio_resampler import SoxrQuality
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    from dpdfnet.models import MODEL_REGISTRY
    from dpdfnet.stream import StreamEnhancer
except ModuleNotFoundError as e:
    MODEL_REGISTRY = {}
    StreamEnhancer = None
    logger.error(f"Exception: {e}")
    logger.error('In order to use the DPDFNet filter, you need to `uv add "pipecat-ai[dpdfnet]"`.')


class DPDFNetFilter(BaseAudioFilter):
    """Audio filter using DPDFNet for real-time speech enhancement.

    DPDFNet is a family of causal dual-path deep filtering models that run on CPU
    via ONNX Runtime. Inference is executed on a dedicated worker thread so the
    event loop is never blocked, and audio is resampled to and from the model's
    native sample rate when the transport rate differs.

    The model is causal, so it adds a fixed algorithmic delay of roughly 40 ms to
    the input path. That delay sits upstream of VAD and turn detection, so it adds
    directly to end-of-turn latency.

    Available models, and the quality/CPU trade-off between them:

    - ``baseline`` (16 kHz): cheapest. Use on constrained hosts or at high concurrency.
    - ``dpdfnet2`` (16 kHz): best quality/CPU balance. The default.
    - ``dpdfnet4``, ``dpdfnet8`` (16 kHz): higher quality, higher cost.
    - ``dpdfnet2_8khz``, ``dpdfnet8_8khz`` (8 kHz): telephony-native.
    - ``dpdfnet2_48khz_hr``, ``dpdfnet8_48khz_hr`` (48 kHz): full-band.

    Match the model's native rate to your transport where you can. When the rates
    already agree no resampler is created at all, which avoids both the extra CPU
    and the resampler's own delay.

    On first use the ONNX weights are downloaded from Hugging Face
    (``Ceva-IP/DPDFNet``) into the dpdfnet cache directory. This happens inside
    ``start()`` and will delay pipeline startup by several seconds. In containers,
    pre-warm the cache at image build time or pass ``onnx_path``.

    If the model cannot be loaded for any reason, the filter logs an error and
    passes audio through unmodified. It never raises into the transport's audio
    task.

    Example::

        from pipecat.audio.filters.dpdfnet_filter import DPDFNetFilter

        transport = SmallWebRTCTransport(
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_filter=DPDFNetFilter(model="dpdfnet2"),
            ),
        )
    """

    def __init__(
        self,
        *,
        model: str = "dpdfnet2",
        onnx_path: str | Path | None = None,
        resampler_quality: SoxrQuality = "QQ",
    ) -> None:
        """Initialize the DPDFNet speech enhancement filter.

        Args:
            model: Name of the DPDFNet model to run. Defaults to "dpdfnet2".
                   Use "baseline" for the lowest CPU cost, or one of the 8 kHz and
                   48 kHz variants to match a transport's native rate.
            onnx_path: Optional path to a local ONNX file, which bypasses the
                       download entirely. The model argument must still name the
                       matching registry entry, because the native sample rate is
                       resolved from it.
            resampler_quality: Quality of the resampler if resampling is needed.
                               One of "VHQ", "HQ", "MQ", "LQ", "QQ". Defaults to "QQ"
                               (Quick) for lowest latency.

        Raises:
            ValueError: If model is not a known DPDFNet model.
        """
        self._model = model
        self._onnx_path = onnx_path
        self._resampler_quality: SoxrQuality = resampler_quality

        self._filtering = True
        self._sample_rate = 0
        # Native rate of the selected model, resolved in start(). 0 means unknown.
        self._model_sample_rate = 0
        self._enhancer = None
        self._dpdfnet_ready = False
        self._resampler_in = None
        self._resampler_out = None
        self._executor: ThreadPoolExecutor | None = None
        # Carries a trailing byte when a chunk is not int16-aligned.
        self._partial_byte = b""
        # Set when filtering is re-enabled, so the model is reset on the worker
        # thread rather than racing an in-flight process() call.
        self._reset_pending = False

        # Safe to raise here: __init__ runs in user code at pipeline construction
        # time, not inside the transport. Fail fast on a typo rather than silently
        # passing audio through for the whole session. Skipped when dpdfnet isn't
        # installed, since that case must degrade to pass-through.
        if MODEL_REGISTRY and model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown DPDFNet model {model!r}. "
                f"Available models: {', '.join(sorted(MODEL_REGISTRY))}."
            )

    async def start(self, sample_rate: int):
        """Initialize the filter with the transport's sample rate.

        Loads the DPDFNet model off the event loop. On first use this may download
        the ONNX weights, which can take several seconds. If loading fails for any
        reason the filter degrades to pass-through and never raises.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        self._partial_byte = b""
        self._reset_pending = False

        # Thread executor that will run the model. We only need one thread per
        # filter because one filter just handles one audio stream, and two threads
        # would interleave the model's recurrent state.
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

        try:
            # The module-level import sets `StreamEnhancer` to `None` if dpdfnet
            # isn't installed; raise instead of calling `None(...)` so the except
            # clause handles it cleanly.
            if StreamEnhancer is None:
                raise ImportError("dpdfnet is not installed")

            loop = asyncio.get_running_loop()
            self._enhancer = await loop.run_in_executor(self._executor, self._load_enhancer)

            # StreamEnhancer keeps its native rate private, but it derives that rate
            # from MODEL_REGISTRY[model] before it ever looks at onnx_path, so the
            # registry is authoritative in both the downloaded and local-file cases.
            self._model_sample_rate = MODEL_REGISTRY[self._model].sample_rate
            self._dpdfnet_ready = True
        except Exception as e:
            logger.error(
                f"Failed to initialize DPDFNet model {self._model!r}: {e}. "
                "Pass onnx_path= or pre-warm the dpdfnet cache to avoid a download."
            )
            self._enhancer = None
            self._dpdfnet_ready = False
            return

        if self._sample_rate != self._model_sample_rate:
            logger.info(
                f"DPDFNet filter enabling resampling: "
                f"{self._sample_rate} <-> {self._model_sample_rate}"
            )
            try:
                from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler

                # Two instances: a stream resampler cannot be reused with a
                # different rate pair.
                self._resampler_in = SOXRStreamAudioResampler(quality=self._resampler_quality)
                self._resampler_out = SOXRStreamAudioResampler(quality=self._resampler_quality)
            except ImportError as e:
                logger.error(f"Could not import SOXRStreamAudioResampler for resampling: {e}")
                self._dpdfnet_ready = False

    async def stop(self):
        """Clean up the DPDFNet engine when stopping.

        Called from the input transport's stop, cancel and cleanup paths, so this
        must be safe to call more than once.
        """
        # Cleared first, so any concurrent filter() call short-circuits to
        # pass-through before the executor goes away.
        self._dpdfnet_ready = False
        self._enhancer = None
        self._resampler_in = None
        self._resampler_out = None
        self._partial_byte = b""
        self._reset_pending = False

        # The swap makes the shutdown exactly-once under a re-entrant call.
        executor, self._executor = self._executor, None
        if executor is not None:
            # A queued-but-unstarted future is cancelled, which surfaces
            # CancelledError in the awaiting filter() call. That is intentional:
            # stop() is only reached once the audio task is already being torn
            # down, and CancelledError derives from BaseException, so filter()'s
            # `except Exception` correctly lets it through instead of hanging
            # shutdown.
            executor.shutdown(wait=False, cancel_futures=True)

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            # Re-enabling after a gap leaves stale recurrent state and half a
            # window of pre-gap audio behind. Flag a reset instead of calling
            # reset() here, so it is applied on the worker thread and cannot race
            # an in-flight process(). Only on a false -> true transition, so a
            # redundant enable doesn't punch a hole in the audio.
            if frame.enable and not self._filtering:
                self._reset_pending = True
            self._filtering = frame.enable

    def _load_enhancer(self):
        """Construct the StreamEnhancer. Runs on the executor thread."""
        return StreamEnhancer(model=self._model, onnx_path=self._onnx_path, verbose=False)

    def _run_enhancer(self, audio: bytes) -> bytes:
        """Run one chunk through DPDFNet. Runs on the executor thread."""
        # `_reset_pending` has one writer per thread and is a plain bool, so a torn
        # read is impossible; the worst interleaving applies the reset one chunk
        # late. No lock needed.
        if self._reset_pending:
            self._reset_pending = False
            self._enhancer.reset()

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # sample_rate=None means "already at the model's native rate", which makes
        # dpdfnet's own resampling a no-op. That resampling is per-chunk librosa
        # with no history across chunks, so we do the conversion ourselves with a
        # streaming resampler instead. Do not pass the transport rate here.
        enhanced = self._enhancer.process(samples, sample_rate=None)

        # Empty is normal, not a failure: the causal STFT needs a full window first.
        if enhanced.size == 0:
            return b""

        # Clip in the float domain BEFORE scaling. astype(int16) wraps on overflow,
        # so an overlap-add overshoot above 1.0 would flip sign and produce a
        # full-scale click.
        return (np.clip(enhanced, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    async def filter(self, audio: bytes) -> bytes:
        """Apply DPDFNet speech enhancement to audio data.

        Resamples to the model's native rate if needed, runs the model on a worker
        thread, and resamples the result back to the transport's rate.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Enhanced audio data as bytes, or empty bytes while the model is still
            filling its first analysis window. Returns the input unmodified if the
            filter is disabled or unavailable.
        """
        if not self._dpdfnet_ready or not self._filtering or self._enhancer is None:
            return audio

        # Also covers a stop() that raced us.
        executor = self._executor
        if executor is None:
            return audio

        if not audio:
            return b""

        try:
            # Realign to int16 boundaries, carrying any odd trailing byte. This has
            # to happen before resampling, because the resampler also reads the
            # buffer as int16 and would raise first. Truncating instead of carrying
            # would desync every later sample by one byte.
            if self._partial_byte:
                audio = self._partial_byte + audio
                self._partial_byte = b""
            if len(audio) % 2:
                audio, self._partial_byte = audio[:-1], audio[-1:]
                if not audio:
                    return b""

            # Resample input to the model's native rate if needed.
            in_audio = audio
            if self._resampler_in:
                in_audio = await self._resampler_in.resample(
                    audio, self._sample_rate, self._model_sample_rate
                )

            # A short chunk can legitimately resample to nothing.
            if not in_audio:
                return b""

            loop = asyncio.get_running_loop()
            enhanced = await loop.run_in_executor(executor, self._run_enhancer, in_audio)

            # Model is still filling its first analysis window.
            if not enhanced:
                return b""

            # Resample output back to the transport's rate if needed.
            if self._resampler_out:
                return await self._resampler_out.resample(
                    enhanced, self._model_sample_rate, self._sample_rate
                )

            return enhanced
        except Exception as e:
            # filter() is awaited on the transport's audio task, which does not
            # catch exceptions. Never let one escape; degrade to pass-through.
            logger.error(f"DPDFNet filter error, passing audio through: {e}")
            return audio
