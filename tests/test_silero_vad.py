#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState

SAMPLE_RATE = 16000
# 20 ms, the frame size transports deliver.
CHUNK_SAMPLES = 320
# Synthesized (Kokoro) speech with every pause removed, so a VAD listening to
# it should never report the speaker stopping.
SPEECH_PATH = Path(__file__).parent / "assets" / "continuous_speech_16k.wav"


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


class TestSileroVADLongStreams(unittest.IsolatedAsyncioTestCase):
    """Silero VAD over long streams, where the model's recurrent state persists."""

    @classmethod
    def setUpClass(cls):
        speech, sample_rate = sf.read(SPEECH_PATH, dtype="int16")
        assert sample_rate == SAMPLE_RATE
        cls.speech = speech

    def setUp(self):
        # A fresh analyzer per test, like a new audio stream.
        self.analyzer = SileroVADAnalyzer(sample_rate=SAMPLE_RATE)
        self.analyzer.set_sample_rate(SAMPLE_RATE)
        self.rng = np.random.default_rng(0)

    def noise(self, seconds: float, std: float) -> np.ndarray:
        return self.rng.normal(0, std, int(seconds * SAMPLE_RATE))

    async def analyze(self, audio: np.ndarray) -> list[VADState]:
        """Feed audio to the analyzer in transport-sized frames.

        Tests run much faster than real time, so the wall clock is driven from
        the audio position. Anything the analyzer times by the wall clock then
        behaves as it would in a live stream.

        Returns:
            The VAD state after each frame.
        """
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        clock = [1_000_000.0]
        states = []
        with patch("time.time", lambda: clock[0]):
            for start in range(0, len(audio) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
                clock[0] += CHUNK_SAMPLES / SAMPLE_RATE
                frame = audio[start : start + CHUNK_SAMPLES].tobytes()
                states.append(await self.analyzer.analyze_audio(frame))
        return states

    async def test_detects_speech_after_minutes_of_background_sound(self):
        """Quiet speech is still detected after minutes of room tone or noise.

        A model whose state drifts during non-speech loses confidence in the
        speech that follows, which shows first on quiet speech.
        """
        quiet_speech = self.speech * 0.15
        for name, std in (("room tone", 60), ("noise", 300)):
            with self.subTest(name):
                self.setUp()
                lead_in = self.noise(120, std)
                speech = quiet_speech + self.noise(len(quiet_speech) / SAMPLE_RATE, std)

                states = await self.analyze(np.concatenate([lead_in, speech]))

                during_speech = states[len(lead_in) // CHUNK_SAMPLES :]
                speaking = during_speech.count(VADState.SPEAKING) / len(during_speech)
                self.assertGreater(
                    speaking, 0.8, f"VAD missed speech after 2 minutes of {name}: {speaking:.0%}"
                )

    async def test_sustained_speech_has_no_false_stops(self):
        """Minutes of uninterrupted speech over background noise never end early."""
        speech = np.tile(self.speech, 30).astype(np.float64)  # ~3 minutes
        signal_rms = np.sqrt(np.mean(speech**2))
        noisy = speech + self.rng.normal(0, signal_rms / np.sqrt(10), len(speech))  # 10 dB SNR

        states = await self.analyze(noisy)

        after_start = states[states.index(VADState.SPEAKING) :]
        self.assertNotIn(
            VADState.QUIET,
            after_start,
            "VAD reported the speaker stopping during continuous speech",
        )

    async def test_model_state_is_not_reset_mid_stream(self):
        """The model keeps its state across a stream, however long it runs."""
        # The model initializes its state on the first frame it sees.
        await self.analyze(self.noise(1, 300))
        audio = np.concatenate([self.noise(60, 300), self.speech, self.noise(60, 300)])
        with patch.object(self.analyzer._model, "reset_states") as reset_states:
            await self.analyze(audio)
        reset_states.assert_not_called()


if __name__ == "__main__":
    unittest.main()
