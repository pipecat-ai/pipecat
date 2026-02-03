#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from typing import List

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.tests.utils import run_test


class MockVADAnalyzer(VADAnalyzer):
    """A mock VAD analyzer that returns states from a predefined sequence."""

    def __init__(self, states: List[VADState]):
        super().__init__(sample_rate=16000)
        self._states = list(states)
        self._call_index = 0

    def num_frames_required(self) -> int:
        return 512

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.9

    async def analyze_audio(self, buffer: bytes) -> VADState:
        if self._call_index < len(self._states):
            state = self._states[self._call_index]
            self._call_index += 1
            return state
        return VADState.QUIET


class TestVADProcessor(unittest.IsolatedAsyncioTestCase):
    def _make_audio_frame(self):
        return InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)

    async def test_forwards_audio_frames(self):
        """Test that audio frames are forwarded downstream."""
        analyzer = MockVADAnalyzer([VADState.QUIET])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame()],
            expected_down_frames=[InputAudioRawFrame],
        )

    async def test_pushes_started_speaking_frame(self):
        """Test that VADUserStartedSpeakingFrame is pushed when speech starts."""
        analyzer = MockVADAnalyzer([VADState.QUIET, VADState.SPEAKING])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[
                InputAudioRawFrame,
                VADUserStartedSpeakingFrame,
                UserSpeakingFrame,
                InputAudioRawFrame,
            ],
        )

    async def test_pushes_stopped_speaking_frame(self):
        """Test that VADUserStoppedSpeakingFrame is pushed when speech stops."""
        analyzer = MockVADAnalyzer([VADState.SPEAKING, VADState.QUIET])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[
                VADUserStartedSpeakingFrame,
                UserSpeakingFrame,
                InputAudioRawFrame,
                VADUserStoppedSpeakingFrame,
                InputAudioRawFrame,
            ],
        )

    async def test_pushes_user_speaking_frame(self):
        """Test that UserSpeakingFrame is pushed while speaking."""
        analyzer = MockVADAnalyzer([VADState.SPEAKING, VADState.SPEAKING])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[
                VADUserStartedSpeakingFrame,
                UserSpeakingFrame,
                InputAudioRawFrame,
                UserSpeakingFrame,
                InputAudioRawFrame,
            ],
        )

    async def test_no_vad_frames_on_starting_state(self):
        """Test that STARTING state doesn't push VAD frames."""
        analyzer = MockVADAnalyzer([VADState.STARTING])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame()],
            expected_down_frames=[InputAudioRawFrame],
        )

    async def test_no_vad_frames_on_stopping_state(self):
        """Test that STOPPING state doesn't push VAD frames."""
        analyzer = MockVADAnalyzer([VADState.STOPPING])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame()],
            expected_down_frames=[InputAudioRawFrame],
        )

    async def test_no_vad_frames_when_quiet(self):
        """Test that no VAD frames are pushed when staying quiet."""
        analyzer = MockVADAnalyzer([VADState.QUIET, VADState.QUIET])
        processor = VADProcessor(vad_analyzer=analyzer)

        await run_test(
            processor,
            frames_to_send=[self._make_audio_frame(), self._make_audio_frame()],
            expected_down_frames=[InputAudioRawFrame, InputAudioRawFrame],
        )


if __name__ == "__main__":
    unittest.main()
