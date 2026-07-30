#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import unittest

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import (
    _VAD_MAX_CONSECUTIVE_ERRORS,
    SileroVADAnalyzer,
)


class TestSileroVAD(unittest.TestCase):
    """Tests for the Silero VAD analyzer and its ONNX model wrapper."""

    @classmethod
    def setUpClass(cls):
        # Loading the ONNX model once is enough; it's reused across tests.
        cls.analyzer = SileroVADAnalyzer(sample_rate=16000)
        cls.analyzer.set_sample_rate(16000)
        cls.model = cls.analyzer._model

    def test_too_many_dimensions_raises_value_error(self):
        """A >2-D input must raise a descriptive ValueError, not AttributeError.

        The error message previously called ``x.dim()`` (a PyTorch method) on a
        NumPy array, which raised ``AttributeError`` and masked the real error.
        """
        bad_input = np.zeros((1, 1, 512), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            self.model(bad_input, 16000)
        # The dimension count should be interpolated into the message.
        self.assertIn("3", str(ctx.exception))

    def test_voice_confidence_silence(self):
        """voice_confidence converts a raw int16 buffer and returns a valid score."""
        silence = np.zeros(512, dtype=np.int16).tobytes()
        confidence = float(np.ravel(self.analyzer.voice_confidence(silence))[0])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        # Silence should not be confidently classified as speech.
        self.assertLess(confidence, 0.5)

    def test_voice_confidence_conversion_matches_expected(self):
        """The int16->float32 conversion feeds the model the expected values.

        Confirms the simplified single-conversion path (no redundant
        ``np.frombuffer`` on an already-converted array) produces the same
        normalized float32 samples the model is expected to receive.
        """
        samples = np.array([0, 16384, -32768, 32767], dtype=np.int16)
        buffer = samples.tobytes()

        captured = {}
        original_model = self.analyzer._model

        def capturing_model(audio_float32, sample_rate):
            captured["audio"] = audio_float32
            # Pad to a valid frame so the underlying model doesn't complain.
            padded = np.zeros(512, dtype=np.float32)
            return original_model(padded, sample_rate)

        self.analyzer._model = capturing_model
        try:
            self.analyzer.voice_confidence(buffer)
        finally:
            self.analyzer._model = original_model

        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(captured["audio"], expected)


class TestSileroVADErrorReporting(unittest.TestCase):
    """A persistently failing analyzer must be legible, not a log flood.

    `voice_confidence` returns 0 on any exception. That is the right fail-safe,
    but 0 is indistinguishable downstream from real silence, so a persistent
    fault presents as a caller who never speaks — while emitting one ERROR per
    audio frame (~50/second/leg). Reproduced with a mismatched sample rate.
    """

    def _make_failing_analyzer(self) -> SileroVADAnalyzer:
        analyzer = SileroVADAnalyzer(sample_rate=16000)

        def always_fails(audio_float32, sample_rate):
            raise ValueError(f"Supported sampling rates: [8000, 16000] (or multiply of 16000)")

        analyzer._model = always_fails
        return analyzer

    def test_repeated_identical_errors_are_rate_limited(self):
        analyzer = self._make_failing_analyzer()
        buffer = np.zeros(512, dtype=np.int16).tobytes()

        log_output = io.StringIO()
        handler_id = logger.add(log_output, level="ERROR", format="{message}")
        try:
            for _ in range(200):
                self.assertEqual(analyzer.voice_confidence(buffer), 0)
        finally:
            logger.remove(handler_id)

        lines = [line for line in log_output.getvalue().splitlines() if line]
        # The first occurrence at full detail, plus one degraded-analyzer line.
        self.assertLessEqual(len(lines), 3, lines)
        self.assertTrue(any("Supported sampling rates" in line for line in lines))
        degraded = [line for line in lines if "permanently degraded" in line]
        self.assertEqual(len(degraded), 1, lines)
        self.assertGreaterEqual(analyzer._consecutive_errors, _VAD_MAX_CONSECUTIVE_ERRORS)

    def test_counter_resets_on_recovery(self):
        analyzer = self._make_failing_analyzer()
        buffer = np.zeros(512, dtype=np.int16).tobytes()
        for _ in range(5):
            analyzer.voice_confidence(buffer)
        self.assertEqual(analyzer._consecutive_errors, 5)

        class _HealthyModel:
            def __call__(self, audio_float32, sample_rate):
                return [0.42]

            def reset_states(self):
                pass

        analyzer._model = _HealthyModel()
        self.assertAlmostEqual(analyzer.voice_confidence(buffer), 0.42)
        self.assertEqual(analyzer._consecutive_errors, 0)


if __name__ == "__main__":
    unittest.main()
