#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer


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


if __name__ == "__main__":
    unittest.main()
