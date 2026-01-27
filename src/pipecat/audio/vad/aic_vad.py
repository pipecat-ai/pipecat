"""AIC-integrated VAD analyzer that lazily binds to the AIC SDK backend.

This analyzer queries the backend's is_speech_detected() and maps it to a float
confidence (1.0/0.0). It uses 10 ms windows based on the sample rate and applies
optional AIC VAD parameters (lookback_buffer_size, sensitivity) when available.
"""

from typing import Any, Callable, Optional

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    from aic import AICVadParameter
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the AIC filter, you need to `pip install pipecat-ai[aic]`.")
    raise Exception(f"Missing module: {e}")


class AICVADAnalyzer(VADAnalyzer):
    """VAD analyzer that lazily instantiates the AIC VoiceActivityDetector via a factory.

    The analyzer can be constructed before the AIC Model exists. Once the filter has
    started and the Model is available, the provided factory will succeed and the
    backend VAD will be created. We then switch to single-sample updates where
    num_frames_required() returns 1 and confidence is derived from the backend's
    boolean is_speech_detected() state.

    AIC VAD runtime parameters:
      - lookback_buffer_size:
          Controls the lookback buffer size used by the VAD, i.e. the number of
          window-length audio buffers used as a lookback buffer. Larger values improve
          stability but increase latency.
          Range: 1.0 .. 20.0
          Default (SDK): 6.0
      - sensitivity:
          Controls the energy threshold sensitivity. Higher values make the detector
          less sensitive (require more energy to count as speech).
          Range: 1.0 .. 15.0
          Formula: Energy threshold = 10 ** (-sensitivity)
          Default (SDK): 6.0
    """

    def __init__(
        self,
        *,
        vad_factory: Optional[Callable[[], Any]] = None,
        lookback_buffer_size: Optional[float] = None,
        sensitivity: Optional[float] = None,
    ):
        """Create an AIC VAD analyzer.

        Args:
            vad_factory:
                Zero-arg callable that returns an initialized AIC VoiceActivityDetector.
                This may raise until the filter's Model has been created; the analyzer
                will retry on set_sample_rate/first use.
            lookback_buffer_size:
                Optional override for AIC VAD lookback buffer size.
                Range: 1.0 .. 20.0. Larger values increase stability at the cost of latency.
                If None, the SDK default (6.0) is used.
            sensitivity:
                Optional override for AIC VAD sensitivity (energy threshold).
                Range: 1.0 .. 15.0. Energy threshold = 10 ** (-sensitivity).
                If None, the SDK default (6.0) is used.
        """
        # Use fixed VAD parameters for AIC: no user override
        fixed_params = VADParams(confidence=0.5, start_secs=0.0, stop_secs=0.0, min_volume=0.0)
        super().__init__(sample_rate=None, params=fixed_params)
        self._vad_factory = vad_factory
        self._backend_vad: Optional[Any] = None
        self._pending_lookback: Optional[float] = lookback_buffer_size
        self._pending_sensitivity: Optional[float] = sensitivity

    def bind_vad_factory(self, vad_factory: Callable[[], Any]):
        """Attach or replace the factory post-construction."""
        self._vad_factory = vad_factory
        self._ensure_backend_initialized()

    def _apply_backend_params(self):
        """Apply optional AIC VAD parameters if available."""
        if self._backend_vad is None or AICVadParameter is None:
            return
        try:
            if self._pending_lookback is not None:
                self._backend_vad.set_parameter(
                    AICVadParameter.LOOKBACK_BUFFER_SIZE, float(self._pending_lookback)
                )
            if self._pending_sensitivity is not None:
                self._backend_vad.set_parameter(
                    AICVadParameter.SENSITIVITY, float(self._pending_sensitivity)
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"AIC VAD parameter application deferred/failed: {e}")

    def _ensure_backend_initialized(self):
        if self._backend_vad is not None:
            return
        if not self._vad_factory:
            return
        try:
            self._backend_vad = self._vad_factory()
            self._apply_backend_params()
            # With backend ready, recompute internal frame sizing
            super().set_params(self._params)
            logger.debug("AIC VAD backend initialized in analyzer.")
        except Exception as e:  # noqa: BLE001
            # Filter may not be started yet; try again later
            logger.debug(f"Deferring AIC VAD backend initialization: {e}")

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        # Set rate and attempt backend initialization once we know SR
        self._sample_rate = self._init_sample_rate or sample_rate
        self._ensure_backend_initialized()
        # Ensure params are initialized even if backend not ready yet
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
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score is 0.0 or 1.0.
        """
        # Ensure backend exists (filter might have started since last call)
        self._ensure_backend_initialized()
        if self._backend_vad is None:
            return 0.0

        # We do not need to analyze 'buffer' here since the model's VAD is updated
        # as part of the enhancement pipeline. Simply query the boolean and map it.
        try:
            is_speech = self._backend_vad.is_speech_detected()
            return 1.0 if is_speech else 0.0
        except Exception as e:  # noqa: BLE001
            logger.error(f"AIC VAD inference error: {e}")
            return 0.0
