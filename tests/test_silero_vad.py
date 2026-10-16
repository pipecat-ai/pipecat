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
import soxr

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState

SAMPLE_RATE = 16000
# 8 kHz is phone bandwidth.
PHONE_SAMPLE_RATE = 8000
# 20 ms, the frame size transports deliver.
FRAME_SECS = 0.02
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
    """Silero VAD over long streams, where the model's recurrent state persists.

    Test audio is built at 16 kHz. Tests that run at 8 kHz downsample it first,
    which leaves it with phone bandwidth.
    """

    @classmethod
    def setUpClass(cls):
        speech, sample_rate = sf.read(SPEECH_PATH, dtype="int16")
        assert sample_rate == SAMPLE_RATE
        cls.speech = speech.astype(np.float64)

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.start_stream(SAMPLE_RATE)

    def start_stream(self, sample_rate: int):
        """Start a fresh analyzer, like a new audio stream."""
        self.sample_rate = sample_rate
        self.analyzer = SileroVADAnalyzer(sample_rate=sample_rate)
        self.analyzer.set_sample_rate(sample_rate)
        self.position = 0.0
        self.reset_positions: list[float] = []

        # Record where in the audio the analyzer resets the model. The model
        # also calls reset_states(batch_size) itself to initialize its state;
        # those calls carry an argument and aren't the analyzer's resets.
        model = self.analyzer._model
        reset_states = model.reset_states

        def record_reset(*args, **kwargs):
            if not args and not kwargs:
                self.reset_positions.append(self.position)
            reset_states(*args, **kwargs)

        model.reset_states = record_reset

    def noise(self, seconds: float, std: float) -> np.ndarray:
        return self.rng.normal(0, std, int(seconds * SAMPLE_RATE))

    def pink_noise(self, seconds: float, rms: float) -> np.ndarray:
        """Noise weighted toward low frequencies, like room and line noise."""
        samples = int(seconds * SAMPLE_RATE)
        spectrum = np.fft.rfft(self.rng.normal(0, 1, samples))
        frequencies = np.fft.rfftfreq(samples, 1 / SAMPLE_RATE)
        frequencies[0] = frequencies[1]
        pink = np.fft.irfft(spectrum / np.sqrt(frequencies), samples)
        return pink / np.sqrt(np.mean(pink**2)) * rms

    def music(self, seconds: float) -> np.ndarray:
        """Synthesized music: plucked notes over a sustained chord."""
        samples = int(seconds * SAMPLE_RATE)
        notes = np.zeros(samples)
        note_samples = int(0.25 * SAMPLE_RATE)
        for start in range(0, samples, note_samples):
            t = np.arange(min(note_samples, samples - start)) / SAMPLE_RATE
            frequency = self.rng.choice([261.6, 293.7, 329.6, 392.0, 440.0, 523.3])
            harmonics = sum(np.sin(2 * np.pi * frequency * h * t) / h for h in range(1, 6))
            notes[start : start + len(t)] = harmonics * np.exp(-6 * t)
        t = np.arange(samples) / SAMPLE_RATE
        chord = 0.3 * sum(np.sin(2 * np.pi * f * t) for f in (130.8, 164.8, 196.0))
        music = notes + chord
        return music / np.abs(music).max() * 6000

    def speaking_share(self, states: list[VADState], start: float) -> float:
        """Share of frames from ``start`` seconds on that the VAD reports as speech."""
        after = states[int(start / FRAME_SECS) :]
        return after.count(VADState.SPEAKING) / len(after)

    async def analyze(self, audio: np.ndarray) -> list[VADState]:
        """Feed audio to the analyzer in transport-sized frames.

        Tests run much faster than real time, so the wall clock is driven from
        the audio position. Anything the analyzer times by the wall clock then
        behaves as it would in a live stream.

        Args:
            audio: Audio at 16 kHz, resampled to the stream's sample rate.

        Returns:
            The VAD state after each frame.
        """
        if self.sample_rate != SAMPLE_RATE:
            audio = soxr.resample(audio.astype(np.float32), SAMPLE_RATE, self.sample_rate)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        frame_samples = int(FRAME_SECS * self.sample_rate)
        clock = [1_000_000.0]
        states = []
        with patch("time.time", lambda: clock[0]):
            for start in range(0, len(audio) - frame_samples + 1, frame_samples):
                self.position = start / self.sample_rate
                clock[0] += FRAME_SECS
                frame = audio[start : start + frame_samples].tobytes()
                states.append(await self.analyzer.analyze_audio(frame))
        return states

    async def test_detects_speech_after_minutes_of_background_sound(self):
        """Quiet speech is still detected after minutes of room tone or noise.

        A model whose state drifts during non-speech loses confidence in the
        speech that follows, which shows first on quiet speech.
        """
        quiet_speech = self.speech * 0.15
        for sample_rate in (SAMPLE_RATE, PHONE_SAMPLE_RATE):
            backgrounds = {
                "room tone": lambda seconds: self.noise(seconds, 60),
                "pink noise": lambda seconds: self.pink_noise(seconds, 200),
            }
            for name, background in backgrounds.items():
                with self.subTest(sample_rate=sample_rate, background=name):
                    self.start_stream(sample_rate)
                    lead_in = background(120)
                    speech = quiet_speech + background(len(quiet_speech) / SAMPLE_RATE)

                    states = await self.analyze(np.concatenate([lead_in, speech]))

                    speaking = self.speaking_share(states, len(lead_in) / SAMPLE_RATE)
                    self.assertGreater(
                        speaking,
                        0.8,
                        f"VAD missed speech after 2 minutes of {name}: {speaking:.0%}",
                    )

    async def test_detects_speech_after_music(self):
        """Quiet speech is still detected shortly after a minute of music.

        Music leaves the model unable to detect the speech that follows until
        its state is reset. Runs on phone audio, where music most often reaches
        the inbound track (hold music on outbound calls) and suppresses speech
        most.
        """
        self.start_stream(PHONE_SAMPLE_RATE)
        lead_in = np.concatenate([self.music(60), self.noise(2.5, 60)])

        states = await self.analyze(np.concatenate([lead_in, self.speech * 0.15]))

        speaking = self.speaking_share(states, len(lead_in) / SAMPLE_RATE)
        self.assertGreater(speaking, 0.8, f"VAD missed speech after music: {speaking:.0%}")

    async def test_sustained_speech_has_no_false_stops(self):
        """Minutes of uninterrupted speech over background noise never end early.

        Runs at 16 kHz only: on this looped synthesized voice in white noise,
        the 8 kHz model occasionally reports a stop.
        """
        speech = np.tile(self.speech, 30)  # ~3 minutes
        signal_rms = np.sqrt(np.mean(speech**2))
        noisy = speech + self.rng.normal(0, signal_rms / np.sqrt(10), len(speech))  # 10 dB SNR

        states = await self.analyze(noisy)

        after_start = states[states.index(VADState.SPEAKING) :]
        self.assertNotIn(
            VADState.QUIET,
            after_start,
            "VAD reported the speaker stopping during continuous speech",
        )

    async def test_model_is_reset_during_non_speech(self):
        """The model is reset about every two seconds while there's no speech."""
        await self.analyze(self.noise(30, 300))

        self.assertTrue(
            12 <= len(self.reset_positions) <= 15,
            f"expected a reset about every 2s over 30s, got {len(self.reset_positions)}",
        )
        spacing = np.diff(self.reset_positions)
        self.assertGreaterEqual(spacing.min(), 2.0)

    async def test_model_is_never_reset_during_speech(self):
        """A minute of uninterrupted speech gets no resets, and resets resume after it."""
        speech = np.tile(self.speech, 10)  # ~1 minute
        speech_start = 10.0
        speech_end = speech_start + len(speech) / SAMPLE_RATE

        await self.analyze(
            np.concatenate([self.noise(speech_start, 60), speech, self.noise(10, 60)])
        )

        during_speech = [p for p in self.reset_positions if speech_start <= p <= speech_end]
        self.assertEqual(during_speech, [], "model was reset during speech")
        self.assertTrue(any(p > speech_end for p in self.reset_positions))


if __name__ == "__main__":
    unittest.main()
