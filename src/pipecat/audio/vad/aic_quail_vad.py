#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Standalone VAD analyzer for Pipecat backed by the ai-coustics SDK.

Runs a dedicated ai-coustics VAD model as its own detector. The analyzer owns
its :class:`aic_sdk.Vad` instance and can be placed anywhere in the pipeline.

Classes:
    AICQuailVADAnalyzer: Standalone ai-coustics VAD analyzer.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from aic_sdk import (
    Model,
    ProcessorConfig,
    Vad,
    VadContext,
    # Exported at runtime but absent from aic-sdk 3.0.0's type stub.
    set_sdk_id,  # type: ignore[attr-defined]
)
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

DEFAULT_QUAIL_VAD_MODEL_ID = "vad-2.1-xxs-16khz"

# Telemetry identifier registered with the AIC SDK; identifies pipecat to the
# vendor's usage pipeline. Mirrors the value used by AICFilter; kept private
# (leading underscore) to avoid making it accidental public API.
_AIC_SDK_PIPECAT_ID = 6

# 2^15: normalizes int16 samples (-32768..32767) to float32 (-1.0..0.99997).
_INT16_DTYPE = np.int16
_INT16_SCALE = 32768.0


class AICQuailVADAnalyzer(VADAnalyzer):
    """Standalone VAD analyzer powered by the ai-coustics SDK.

    The analyzer owns a dedicated :class:`aic_sdk.Vad` initialized with a
    VAD-only model. Each :meth:`voice_confidence` call processes one audio
    window and returns the model's raw speech probability in ``[0.0, 1.0]``
    (:meth:`aic_sdk.VadContext.raw_vad_probability`). The base
    :class:`VADAnalyzer` state machine then gates speech start/stop using its own
    :class:`VADParams` (``confidence`` threshold, ``start_secs``, ``stop_secs``),
    so the SDK's own VAD post-processing (sensitivity thresholding, speech-hold)
    is bypassed — Pipecat owns the thresholding.

    The analyzer reads whatever audio the pipeline feeds it. When
    :class:`pipecat.audio.filters.aic_filter.AICFilter` is installed as the
    transport's ``audio_in_filter``, that audio has already been enhanced,
    while the AIC SDK expects a VAD to run on the original signal. Detection
    still works, but the model is judging audio it was not trained on and the
    filter's own delay is added to the VAD's prediction delay. Pipecat offers no
    hook for tapping pre-enhancement audio further down the pipeline, so this is
    a limitation of pairing the two rather than something the analyzer can
    correct.

    Example::

        analyzer = AICQuailVADAnalyzer(license_key=os.environ["AIC_SDK_LICENSE"])
        # ``set_sample_rate`` is invoked by the pipeline once the transport
        # sample rate is known.
    """

    def __init__(
        self,
        *,
        license_key: str,
        model_id: str | None = DEFAULT_QUAIL_VAD_MODEL_ID,
        model_path: Path | None = None,
        model_download_dir: Path | None = None,
        speech_hold_duration: float | None = None,
        minimum_speech_duration: float | None = None,
        sensitivity: float | None = None,
        sample_rate: int | None = None,
        params: VADParams | None = None,
    ) -> None:
        """Initialize the VAD analyzer.

        Loads the model eagerly so the cold-start CDN download happens at
        construction time (typically before the event loop starts), rather than
        on the first :meth:`set_sample_rate` call from a running pipeline.

        Args:
            license_key: ai-coustics SDK license key.
            model_id: Dedicated VAD model identifier. Defaults to
                ``"vad-2.1-xxs-16khz"``. See https://artifacts.ai-coustics.io/
                for the catalogue. Ignored if ``model_path`` is provided.
                Enhancement models are rejected by the SDK.
            model_path: Optional path to a local ``.aicmodel`` file. Overrides
                ``model_id`` when set.
            model_download_dir: Directory for downloaded models. Defaults to
                ``~/.cache/pipecat/aic-models``.
            speech_hold_duration: Deprecated; no longer used. Speech timing is
                governed by Pipecat's ``VADParams``.

                .. deprecated:: 1.5.0
                    Use :class:`VADParams` (``start_secs``/``stop_secs``) instead.
                    ``speech_hold_duration`` is ignored and will be removed in 2.0.0.

            minimum_speech_duration: Deprecated; no longer used. Speech timing is
                governed by Pipecat's ``VADParams``.

                .. deprecated:: 1.5.0
                    Use :class:`VADParams` (``start_secs``/``stop_secs``) instead.
                    ``minimum_speech_duration`` is ignored and will be removed in 2.0.0.

            sensitivity: Deprecated; no longer used. The speech-probability
                threshold is now governed by Pipecat's ``VADParams.confidence``.

                .. deprecated:: 1.5.0
                    Use :class:`VADParams` (``confidence``) instead. ``sensitivity``
                    is ignored and will be removed in 2.0.0.

            sample_rate: Initial sample rate; the pipeline will set this via
                :meth:`set_sample_rate` once the transport rate is known.
            params: Optional :class:`VADParams` for the base state machine.

        Raises:
            ValueError: If neither ``model_id`` nor ``model_path`` is provided.
        """
        if model_id is None and model_path is None:
            raise ValueError(
                "Either 'model_id' or 'model_path' must be provided. "
                "See https://artifacts.ai-coustics.io/ for available models."
            )

        super().__init__(sample_rate=sample_rate, params=params)

        # These SDK-side knobs only affected the post-processed ``is_speech_detected``
        # path, which the raw-probability ``voice_confidence`` no longer uses. They are
        # accepted-but-ignored for one release cycle; gating now lives in ``VADParams``.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            for _name, _value in (
                ("speech_hold_duration", speech_hold_duration),
                ("minimum_speech_duration", minimum_speech_duration),
                ("sensitivity", sensitivity),
            ):
                if _value is not None:
                    warnings.warn(
                        f"`AICQuailVADAnalyzer.{_name}` is deprecated since 1.5.0 and will "
                        "be removed in 2.0.0. Use `VADParams` instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

        self._license_key = license_key
        self._model_id = model_id
        self._model_path = model_path
        self._model_download_dir = model_download_dir or (
            Path.home() / ".cache" / "pipecat" / "aic-models"
        )

        self._model: Model | None = None
        self._vad: Vad | None = None
        self._vad_ctx: VadContext | None = None
        self._frames_per_block: int = 0
        # Pre-allocated float32 buffer used by voice_confidence; sized at
        # VAD init. Avoids per-call heap allocations on the audio path.
        self._in_f32: np.ndarray | None = None
        # Latches so we log inference / buffer-size errors at ERROR once and
        # drop subsequent occurrences to DEBUG. The inference latch resets on a
        # successful inference so a recovery followed by a new failure surfaces
        # at ERROR again. The buffer-size latch resets on VAD re-init.
        self._inference_error_logged = False
        self._buffer_size_warning_logged = False

        # Eager model load shifts CDN download out of the hot path. If anything
        # in this block raises (telemetry registration, network, license, etc.)
        # we shut down the base-class executor so the half-constructed instance
        # doesn't leak its worker thread, then propagate the original error.
        try:
            set_sdk_id(_AIC_SDK_PIPECAT_ID)
            self._ensure_model_loaded()
            if sample_rate is not None:
                self._initialize_vad(sample_rate)
        except Exception:
            try:
                self._executor.shutdown(wait=False)
            except Exception as e:  # noqa: BLE001 - executor cleanup is best-effort
                logger.debug(f"AICQuailVADAnalyzer executor shutdown failed: {e}")
            raise

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if self._model_path is not None:
            logger.debug(f"Loading AIC VAD model from file: {self._model_path}")
            self._model = Model.from_file(str(self._model_path))
            return
        # model_id path (validated in __init__).
        assert self._model_id is not None
        self._model_download_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Downloading AIC VAD model {self._model_id!r} to {self._model_download_dir}")
        model_path = Model.download(self._model_id, str(self._model_download_dir))
        self._model = Model.from_file(model_path)

    def _initialize_vad(self, sample_rate: int) -> None:
        self._ensure_model_loaded()
        assert self._model is not None

        block_size = self._model.get_optimal_block_size(sample_rate)
        config = ProcessorConfig(
            sample_rate=sample_rate,
            block_size=block_size,
        )

        try:
            vad = Vad(self._model, self._license_key, config)
        except Exception:
            logger.error(
                f"AICQuailVADAnalyzer failed to construct Vad at {sample_rate} Hz; "
                "check license key, model type and SDK version."
            )
            raise

        # New VAD constructed successfully; only now is it safe to tear the
        # previous one down. Terminating before construction would wipe
        # in-flight state on the rollback path if Vad() raised.
        previous_vad = self._vad
        if previous_vad is not None:
            try:
                previous_vad.terminate_session()
            except Exception as e:  # noqa: BLE001 - teardown is best-effort
                logger.debug(f"Old Vad termination failed during re-init: {e}")

        vad_ctx = vad.get_context()
        self._vad = vad
        self._vad_ctx = vad_ctx
        self._frames_per_block = block_size
        self._in_f32 = np.zeros(block_size, dtype=np.float32)
        self._inference_error_logged = False
        self._buffer_size_warning_logged = False
        logger.debug(
            f"AICQuailVADAnalyzer initialized at {sample_rate} Hz, block_size={block_size}, "
            f"prediction delay={vad_ctx.get_prediction_delay()} samples"
        )

    def set_sample_rate(self, sample_rate: int) -> None:
        """Set the sample rate. Recreates the SDK VAD if the rate changed.

        Initializes the VAD before delegating to the base class so the
        base's internal sizing uses the correct ``num_frames_required()``
        (driven by the model's optimal block size) instead of the pre-init
        fallback. If VAD initialization fails, the previous VAD/state is
        restored so the analyzer stays usable at its old sample rate rather
        than half-initialized.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        # Snapshot current state for rollback if _initialize_vad raises.
        snapshot = (
            self._vad,
            self._vad_ctx,
            self._in_f32,
            self._frames_per_block,
        )
        try:
            self._initialize_vad(sample_rate)
        except Exception:
            (
                self._vad,
                self._vad_ctx,
                self._in_f32,
                self._frames_per_block,
            ) = snapshot
            raise
        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        """Return the number of int16 frames per analysis window."""
        if self._frames_per_block > 0:
            return self._frames_per_block
        # Pre-initialization fallback so the base class can compute internal
        # sizes before the pipeline calls set_sample_rate.
        return int(self.sample_rate * 0.01) if self.sample_rate else 160

    def voice_confidence(self, buffer: bytes) -> float:
        """Run the VAD model on one audio window.

        Args:
            buffer: int16 little-endian audio samples for one window of
                :meth:`num_frames_required` samples.

        Returns:
            The model's raw speech probability in ``[0.0, 1.0]``. The base
            :class:`VADAnalyzer` compares this against ``VADParams.confidence``
            to decide speech. Returns ``0.0`` if the VAD is not yet initialized
            (i.e. :meth:`set_sample_rate` has not run), if the buffer size does
            not match the expected window, or if an SDK inference error occurs.
        """
        if self._vad is None or self._vad_ctx is None or self._in_f32 is None:
            return 0.0
        expected_bytes = self._frames_per_block * 2  # int16 = 2 bytes per sample
        if len(buffer) != expected_bytes:
            if not self._buffer_size_warning_logged:
                logger.warning(
                    f"AIC VAD buffer size {len(buffer)} != expected {expected_bytes}; "
                    "skipping window. Subsequent size-mismatch warnings will be silenced."
                )
                self._buffer_size_warning_logged = True
            return 0.0
        try:
            # Reuse the pre-allocated buffer; np.copyto casts int16 -> float32,
            # then the in-place divide normalizes to [-1.0, 0.99997].
            np.copyto(self._in_f32, np.frombuffer(buffer, dtype=_INT16_DTYPE))
            self._in_f32 /= _INT16_SCALE
            self._vad.process(self._in_f32)
            # Successful inference re-arms the error latch so a fresh error
            # after a recovery is reported at ERROR rather than buried at DEBUG.
            self._inference_error_logged = False
            # Raw model probability (no SDK post-processing); clamp defensively
            # to the [0.0, 1.0] the VADAnalyzer state machine expects.
            probability = float(self._vad_ctx.raw_vad_probability())
            return max(0.0, min(1.0, probability))
        except Exception as e:  # noqa: BLE001 - keep the pipeline alive on SDK errors
            if not self._inference_error_logged:
                logger.error(f"AIC VAD inference error: {e}")
                self._inference_error_logged = True
            else:
                logger.debug(f"AIC VAD inference error: {e}")
            return 0.0

    async def cleanup(self) -> None:
        """Terminate the VAD session and release the model handle.

        Concurrency contract: callers must ensure no :meth:`voice_confidence`
        call is in flight when ``cleanup`` runs. The pipeline always orders
        ``cleanup`` after the source-stream stop, so in normal pipeline use
        this is guaranteed. If you invoke ``cleanup`` from outside that
        ordering, drain or cancel any in-flight :meth:`analyze_audio` work
        first.
        """
        await super().cleanup()
        if self._vad is not None:
            try:
                self._vad.terminate_session()
            except Exception as e:  # noqa: BLE001 - cleanup is best-effort
                logger.debug(f"AIC VAD session termination failed during cleanup: {e}")
        self._vad = None
        self._vad_ctx = None
        self._model = None
        self._in_f32 = None
        self._frames_per_block = 0
