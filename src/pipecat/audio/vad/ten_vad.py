"""TEN VAD analyzer integration for Pipecat.

This module provides a VAD analyzer implementation using the TEN VAD
backend. Audio buffers are processed in fixed-size chunks and converted
into voice activity confidence scores compatible with Pipecat's
VADAnalyzer interface.
"""

from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    from ten_vad import TenVad
except ModuleNotFoundError as e:
    logger.error("TEN VAD not installed.")
    raise Exception(
        "TEN VAD dependency missing. Install required package to use TenVadAnalyzer."
    ) from e


class TenVadAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using TEN VAD."""

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = None,
        threshold: float = 0.6,
        params: Optional[VADParams] = None,
    ):
        """Initialize TEN VAD analyzer.

        Args:
            sample_rate: Audio sample rate in Hz. Must be 16000.
            threshold: Detection threshold used by TEN VAD.
            params: Optional VAD parameters controlling smoothing behavior.
        """
        super().__init__(sample_rate=sample_rate, params=params)

        self._vad = TenVad(threshold=threshold)
        self._chunk_size = 256
        self._buffer = np.array([], dtype=np.int16)

    def set_sample_rate(self, sample_rate: int):
        """Set analyzer sample rate.

        Args:
            sample_rate: Audio sample rate in Hz.

        Raises:
            ValueError: If sample rate is unsupported.
        """
        if sample_rate != 16000:
            raise ValueError("TEN VAD requires 16000 Hz sample rate")

        super().set_sample_rate(sample_rate)

    def num_frames_required(self) -> int:
        """Return number of frames required per inference step."""
        return self._chunk_size

    def voice_confidence(self, buffer: bytes) -> float:
        """Compute voice confidence for the given audio buffer.

        Args:
            buffer: Raw PCM16 audio bytes.

        Returns:
            Float confidence between 0.0 and 1.0.
        """
        try:
            pcm = np.frombuffer(buffer, dtype=np.int16)
            self._buffer = np.concatenate([self._buffer, pcm])

            confidence = 0.0

            while len(self._buffer) >= self._chunk_size:
                chunk = self._buffer[: self._chunk_size]
                self._buffer = self._buffer[self._chunk_size :]

                result = self._vad.process(chunk)

                if isinstance(result, tuple):
                    confidence = float(result[0])
                else:
                    confidence = 1.0 if result else 0.0

            return confidence

        except Exception as e:
            logger.error(f"TEN VAD error: {e}")
            return 0.0
