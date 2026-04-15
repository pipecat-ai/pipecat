"""AIC-integrated VAD analyzer that lazily binds to the AIC SDK backend.

This module provides VAD analyzer implementations that query the AIC SDK's
is_speech_detected() and map it to a float confidence (1.0/0.0).

Classes:
    AICVADAnalyzer: For aic-sdk (uses 'aic_sdk' module)
"""

from typing import Any, Callable, Optional

from aic_sdk import VadParameter
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams


class AICVADAnalyzer(VADAnalyzer):
    """VAD analyzer that lazily binds to the AIC VadContext via a factory.

    The analyzer can be constructed before the AIC Processor exists. Once the filter has
    started and the Processor is available, the provided factory will succeed and the
    VadContext will be obtained. The context's is_speech_detected() boolean state is
    then mapped to 1.0 (speech) or 0.0 (no speech) to satisfy the VADAnalyzer interface.

    AIC VAD runtime parameters:
      - speech_hold_duration:
          Controls for how long the VAD continues to detect speech after the audio signal
          no longer contains speech (in seconds).
          Range: 0.0 to 100x model window length
          Default (SDK): 0.05s (50ms)
      - minimum_speech_duration:
          Controls for how long speech needs to be present in the audio signal before the
          VAD considers it speech (in seconds).
          Range: 0.0 to 1.0
          Default (SDK): 0.0s
      - sensitivity:
          Controls the sensitivity (energy threshold) of the VAD. This value is used by
          the VAD as the threshold a speech audio signal's energy has to exceed in order
          to be considered speech.
          Range: 1.0 to 15.0
          Formula: Energy threshold = 10 ** (-sensitivity)
          Default (SDK): 6.0
    """

    def __init__(
        self,
        *,
        vad_context_factory: Optional[Callable[[], Any]] = None,
        speech_hold_duration: Optional[float] = None,
        minimum_speech_duration: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ):
        """Create an AIC VAD analyzer.

        Args:
            vad_context_factory:
                Zero-arg callable that returns the AIC VadContext.
                This may raise until the filter's Processor has been created; the analyzer
                will retry on set_sample_rate/first use.
            speech_hold_duration:
                Optional override for AIC VAD speech hold duration (in seconds).
                Range: 0.0 to 100x model window length.
                If None, the SDK default (0.05s) is used.
            minimum_speech_duration:
                Optional override for minimum speech duration before VAD reports
                speech detected (in seconds).
                Range: 0.0 to 1.0.
                If None, the SDK default (0.0s) is used.
            sensitivity:
                Optional override for AIC VAD sensitivity (energy threshold).
                Range: 1.0 to 15.0. Energy threshold = 10 ** (-sensitivity).
                If None, the SDK default (6.0) is used.
        """
        # Use fixed VAD parameters for AIC: no user override
        fixed_params = VADParams(confidence=0.5, start_secs=0.0, stop_secs=0.0, min_volume=0.0)
        super().__init__(sample_rate=None, params=fixed_params)

        self._vad_context_factory = vad_context_factory
        self._vad_ctx: Optional[Any] = None
        self._pending_speech_hold_duration: Optional[float] = speech_hold_duration
        self._pending_minimum_speech_duration: Optional[float] = minimum_speech_duration
        self._pending_sensitivity: Optional[float] = sensitivity

    def bind_vad_context_factory(self, vad_context_factory: Callable[[], Any]):
        """Attach or replace the factory post-construction."""
        self._vad_context_factory = vad_context_factory
        self._ensure_vad_context_initialized()

    def _apply_vad_params(self):
        """Apply optional AIC VAD parameters if available."""
        if self._vad_ctx is None or VadParameter is None:
            return

        try:
            if self._pending_speech_hold_duration is not None:
                self._vad_ctx.set_parameter(
                    VadParameter.SpeechHoldDuration, self._pending_speech_hold_duration
                )
            if self._pending_minimum_speech_duration is not None:
                self._vad_ctx.set_parameter(
                    VadParameter.MinimumSpeechDuration, self._pending_minimum_speech_duration
                )
            if self._pending_sensitivity is not None:
                self._vad_ctx.set_parameter(VadParameter.Sensitivity, self._pending_sensitivity)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"AIC VAD parameter application deferred/failed: {e}")

    def _ensure_vad_context_initialized(self):
        if self._vad_ctx is not None:
            return
        if not self._vad_context_factory:
            return
        try:
            self._vad_ctx = self._vad_context_factory()
            self._apply_vad_params()
            # With VAD context ready, recompute internal frame sizing
            super().set_params(self._params)
            logger.debug("AIC VAD context initialized in analyzer.")
        except Exception as e:  # noqa: BLE001
            # Filter may not be started yet; try again later
            logger.debug(f"Deferring AIC VAD context initialization: {e}")

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        # Set rate and attempt VAD context initialization once we know SR
        self._sample_rate = self._init_sample_rate or sample_rate
        self._ensure_vad_context_initialized()
        # Ensure params are initialized even if VAD context not ready yet
        try:
            super().set_params(self._params)
        except Exception:
            pass

    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            Number of frames needed for VAD processing.
        """
        # Use 10 ms windows based on sample rate
        return int(self.sample_rate * 0.01) if self.sample_rate > 0 else 160

    def voice_confidence(self, buffer: bytes) -> float:
        """Return voice activity detection result for the given audio buffer.

        Note:
            The AIC SDK provides binary speech detection (not a probability score).
            This method returns 1.0 when speech is detected and 0.0 otherwise,
            rather than a true confidence value.

        Args:
            buffer: Audio buffer (unused - AIC VAD state is updated internally
                by the enhancement pipeline).

        Returns:
            1.0 if speech is detected, 0.0 otherwise.
        """
        # Ensure VAD context exists (filter might have started since last call)
        self._ensure_vad_context_initialized()
        if self._vad_ctx is None:
            return 0.0

        # We do not need to analyze 'buffer' here since the processor's VAD is updated
        # as part of the enhancement pipeline. Simply query the boolean and map it.
        try:
            is_speech = self._vad_ctx.is_speech_detected()
            return 1.0 if is_speech else 0.0
        except Exception as e:  # noqa: BLE001
            logger.error(f"AIC VAD inference error: {e}")
            return 0.0
