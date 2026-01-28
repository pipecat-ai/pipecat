#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from typing import List

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import Frame, InputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection


class MockVADAnalyzer(VADAnalyzer):
    """A mock VAD analyzer that returns a configurable state."""

    def __init__(self):
        """Initialize with default QUIET state."""
        super().__init__(sample_rate=16000)
        self._next_state = VADState.QUIET

    def set_next_state(self, state: VADState):
        """Set the state to return on the next analyze_audio call.

        Args:
            state: The VADState to return.
        """
        self._next_state = state

    def num_frames_required(self) -> int:
        return 512

    def voice_confidence(self, buffer: bytes) -> float:
        return 0.9

    async def analyze_audio(self, buffer: bytes) -> VADState:
        """Return the configured state."""
        return self._next_state


class TestVADController(unittest.IsolatedAsyncioTestCase):
    async def test_speech_started_event(self):
        """Test that on_speech_started event is triggered when speech begins."""
        analyzer = MockVADAnalyzer()
        controller = VADController(analyzer)

        speech_started = False

        @controller.event_handler("on_speech_started")
        async def on_speech_started(_controller):
            nonlocal speech_started
            speech_started = True

        start_frame = StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000)
        await controller.process_frame(start_frame)

        audio_frame = InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)

        # Process with QUIET state - no event should fire
        analyzer.set_next_state(VADState.QUIET)
        await controller.process_frame(audio_frame)
        self.assertFalse(speech_started)

        # Process with SPEAKING state - event should fire
        analyzer.set_next_state(VADState.SPEAKING)
        await controller.process_frame(audio_frame)
        self.assertTrue(speech_started)

    async def test_speech_stopped_event(self):
        """Test that on_speech_stopped event is triggered when speech ends."""
        analyzer = MockVADAnalyzer()
        controller = VADController(analyzer)

        speech_stopped = False

        @controller.event_handler("on_speech_stopped")
        async def on_speech_stopped(_controller):
            nonlocal speech_stopped
            speech_stopped = True

        start_frame = StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000)
        await controller.process_frame(start_frame)

        audio_frame = InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)

        # Start speaking
        analyzer.set_next_state(VADState.SPEAKING)
        await controller.process_frame(audio_frame)
        self.assertFalse(speech_stopped)

        # Stop speaking - event should fire
        analyzer.set_next_state(VADState.QUIET)
        await controller.process_frame(audio_frame)
        self.assertTrue(speech_stopped)

    async def test_speech_activity_event(self):
        """Test that on_speech_activity event is triggered while speaking."""
        analyzer = MockVADAnalyzer()
        controller = VADController(analyzer)

        activity_count = 0

        @controller.event_handler("on_speech_activity")
        async def on_speech_activity(_controller):
            nonlocal activity_count
            activity_count += 1

        start_frame = StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000)
        await controller.process_frame(start_frame)

        audio_frame = InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)

        # Activity events fire while in SPEAKING state
        analyzer.set_next_state(VADState.SPEAKING)
        await controller.process_frame(audio_frame)
        await controller.process_frame(audio_frame)
        self.assertEqual(activity_count, 2)

    async def test_push_frame_event(self):
        """Test that push_frame emits on_push_frame event."""
        analyzer = MockVADAnalyzer()
        controller = VADController(analyzer)

        pushed_frames: List[tuple] = []

        @controller.event_handler("on_push_frame")
        async def on_push_frame(_controller, frame: Frame, direction: FrameDirection):
            pushed_frames.append((frame, direction))

        test_frame = InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)
        await controller.push_frame(test_frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(len(pushed_frames), 1)
        self.assertEqual(pushed_frames[0][0], test_frame)
        self.assertEqual(pushed_frames[0][1], FrameDirection.DOWNSTREAM)

    async def test_broadcast_frame_event(self):
        """Test that broadcast_frame emits on_broadcast_frame event."""
        analyzer = MockVADAnalyzer()
        controller = VADController(analyzer)

        broadcast_calls: List[tuple] = []

        @controller.event_handler("on_broadcast_frame")
        async def on_broadcast_frame(_controller, frame_cls, **kwargs):
            broadcast_calls.append((frame_cls, kwargs))

        await controller.broadcast_frame(
            InputAudioRawFrame, audio=b"\x00", sample_rate=16000, num_channels=1
        )

        self.assertEqual(len(broadcast_calls), 1)
        self.assertEqual(broadcast_calls[0][0], InputAudioRawFrame)
        self.assertEqual(broadcast_calls[0][1]["sample_rate"], 16000)

    async def test_no_event_on_transitional_states(self):
        """Test that STARTING and STOPPING states don't trigger events."""
        analyzer = MockVADAnalyzer()
        controller = VADController(analyzer)

        events_triggered = []

        @controller.event_handler("on_speech_started")
        async def on_speech_started(_controller):
            events_triggered.append("started")

        @controller.event_handler("on_speech_stopped")
        async def on_speech_stopped(_controller):
            events_triggered.append("stopped")

        start_frame = StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000)
        await controller.process_frame(start_frame)

        audio_frame = InputAudioRawFrame(audio=b"\x00" * 1024, sample_rate=16000, num_channels=1)

        # STARTING is a transitional state - no event
        analyzer.set_next_state(VADState.STARTING)
        await controller.process_frame(audio_frame)
        self.assertEqual(events_triggered, [])

        # STOPPING is a transitional state - no event
        analyzer.set_next_state(VADState.STOPPING)
        await controller.process_frame(audio_frame)
        self.assertEqual(events_triggered, [])


if __name__ == "__main__":
    unittest.main()
