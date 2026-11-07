#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that Silero VAD analyzers share one ONNX session without sharing state."""

import unittest

import numpy as np

from pipecat.audio.vad.silero import SileroVADAnalyzer


def _analyzer() -> SileroVADAnalyzer:
    analyzer = SileroVADAnalyzer()
    analyzer.set_sample_rate(16000)
    return analyzer


def _frame() -> bytes:
    rng = np.random.default_rng(0)
    return (rng.normal(0, 0.3, 512) * 32767).astype("int16").tobytes()


class TestSileroVADSessionSharing(unittest.TestCase):
    def test_analyzers_share_one_session(self):
        self.assertIs(_analyzer()._model.session, _analyzer()._model.session)

    def test_state_stays_per_analyzer(self):
        frame = _frame()
        first, second, reference = _analyzer(), _analyzer(), _analyzer()

        # Advancing one analyzer must not move another that shares its session
        for _ in range(5):
            second.voice_confidence(frame)
        confidence = np.ravel(first.voice_confidence(frame))[0]
        self.assertEqual(confidence, np.ravel(reference.voice_confidence(frame))[0])
        # the frames do move the state, so the check above is not vacuous
        self.assertNotEqual(confidence, np.ravel(second.voice_confidence(frame))[0])


if __name__ == "__main__":
    unittest.main()
