#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the VADAnalyzer state machine.

The analysis frame is the unit the state machine advances on, so the state the
analyzer settles on must not depend on how many frames a caller's buffer
happens to carry.
"""

import unittest

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # what SileroVADAnalyzer requires at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2
# start_secs and stop_secs of 0.2 come to six analysis frames at this rate.
THRESHOLD_FRAMES = 6


class ScriptedVADAnalyzer(VADAnalyzer):
    """A VAD analyzer that reports a scripted confidence per analysis frame."""

    def __init__(self, confidences: list[float]):
        super().__init__(
            sample_rate=SAMPLE_RATE,
            params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.0),
        )
        self._confidences = confidences
        self._index = 0
        self.set_sample_rate(SAMPLE_RATE)

    def num_frames_required(self) -> int:
        return FRAME_SAMPLES

    def voice_confidence(self, buffer: bytes) -> float:
        confidence = self._confidences[self._index]
        self._index += 1
        return confidence


class TestVADAnalyzer(unittest.IsolatedAsyncioTestCase):
    async def _settled_state(self, confidences: list[float], frames_per_chunk: int) -> VADState:
        """Feed the script in equal chunks and return the state it settles on."""
        analyzer = ScriptedVADAnalyzer(confidences)
        audio = b"\x00" * FRAME_BYTES * len(confidences)
        step = FRAME_BYTES * frames_per_chunk
        state = VADState.QUIET
        for offset in range(0, len(audio), step):
            state = await analyzer.analyze_audio(audio[offset : offset + step])
        await analyzer.cleanup()
        return state

    async def test_speech_start_survives_a_dip_in_the_same_chunk(self):
        """A dip after the start threshold leaves the analyzer SPEAKING."""
        confidences = [1.0] * THRESHOLD_FRAMES + [0.0] + [1.0] * 5

        for frames_per_chunk in (1, 2, 3, 4, 6, 12):
            with self.subTest(frames_per_chunk=frames_per_chunk):
                state = await self._settled_state(confidences, frames_per_chunk)
                self.assertEqual(state, VADState.SPEAKING)

    async def test_speech_resuming_in_the_same_chunk_still_ends_the_turn(self):
        """Speech returning after the stop threshold leaves the turn ended.

        The trailing speech frame reopens STARTING, which it can only do from
        QUIET.
        """
        confidences = [1.0] * 8 + [0.0] * THRESHOLD_FRAMES + [1.0]

        for frames_per_chunk in (1, 2, 3, 4, 6):
            with self.subTest(frames_per_chunk=frames_per_chunk):
                state = await self._settled_state(confidences, frames_per_chunk)
                self.assertEqual(state, VADState.STARTING)


if __name__ == "__main__":
    unittest.main()
