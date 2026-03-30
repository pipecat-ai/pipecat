#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for VADTurnAnalyzerUserTurnStopStrategy."""

import unittest
from typing import Optional, Tuple
from unittest.mock import AsyncMock

import pytest

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)
from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    SpeechControlParamsFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.user_stop.vad_turn_analyzer_user_turn_stop_strategy import (
    VADTurnAnalyzerUserTurnStopStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager


class MockTurnAnalyzer(BaseTurnAnalyzer):
    """Mock turn analyzer for testing."""

    def __init__(self):
        super().__init__()
        self._append_audio_state = EndOfTurnState.INCOMPLETE
        self._analyze_state = EndOfTurnState.INCOMPLETE
        self._prediction: Optional[MetricsData] = None
        self._speech_triggered = False

    @property
    def speech_triggered(self) -> bool:
        return self._speech_triggered

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return self._append_audio_state

    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        return self._analyze_state, self._prediction

    def update_vad_start_secs(self, vad_start_secs: float):
        pass

    def clear(self):
        pass


class TestVADTurnAnalyzerUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.analyzer = MockTurnAnalyzer()
        self.strategy = VADTurnAnalyzerUserTurnStopStrategy(turn_analyzer=self.analyzer)
        self.task_manager = TaskManager()
        await self.strategy.setup(self.task_manager)

        self.turn_stopped_called = False

        @self.strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.turn_stopped_called = True

        self.pushed_frames = []

        @self.strategy.event_handler("on_push_frame")
        async def on_push_frame(strategy, frame, direction):
            self.pushed_frames.append(frame)

        self.broadcast_frames = []

        @self.strategy.event_handler("on_broadcast_frame")
        async def on_broadcast_frame(strategy, frame_cls, **kwargs):
            self.broadcast_frames.append((frame_cls, kwargs))

    async def asyncTearDown(self):
        await self.strategy.cleanup()

    async def test_vad_stop_complete_triggers_immediately(self):
        """VAD stop with COMPLETE should trigger user turn stopped immediately."""
        self.analyzer._analyze_state = EndOfTurnState.COMPLETE

        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))

        assert self.turn_stopped_called

    async def test_vad_stop_incomplete_does_not_trigger(self):
        """VAD stop with INCOMPLETE should not trigger user turn stopped."""
        self.analyzer._analyze_state = EndOfTurnState.INCOMPLETE

        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))

        assert not self.turn_stopped_called

    async def test_streaming_complete_via_append_audio_triggers(self):
        """Streaming COMPLETE from append_audio should trigger immediately."""
        self.analyzer._append_audio_state = EndOfTurnState.COMPLETE
        # analyze_end_of_turn is called after append_audio returns COMPLETE
        self.analyzer._analyze_state = EndOfTurnState.COMPLETE

        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await self.strategy.process_frame(
            InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
        )

        assert self.turn_stopped_called

    async def test_streaming_incomplete_does_not_trigger(self):
        """Streaming INCOMPLETE from append_audio should not trigger."""
        self.analyzer._append_audio_state = EndOfTurnState.INCOMPLETE

        await self.strategy.process_frame(
            InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
        )

        assert not self.turn_stopped_called

    async def test_vad_start_resets_speaking_state(self):
        """VAD start should set speaking state to True."""
        assert not self.strategy._vad_user_speaking

        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))

        assert self.strategy._vad_user_speaking

    async def test_vad_stop_resets_speaking_state(self):
        """VAD stop should set speaking state to False."""
        self.analyzer._analyze_state = EndOfTurnState.INCOMPLETE

        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        assert self.strategy._vad_user_speaking

        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))
        assert not self.strategy._vad_user_speaking

    async def test_reset_clears_state(self):
        """Reset should clear speaking state."""
        self.strategy._vad_user_speaking = True

        await self.strategy.reset()

        assert not self.strategy._vad_user_speaking

    async def test_metrics_frame_pushed_on_prediction(self):
        """MetricsFrame should be pushed when prediction result is available."""
        prediction = MetricsData(processor="test_analyzer")
        self.analyzer._analyze_state = EndOfTurnState.COMPLETE
        self.analyzer._prediction = prediction

        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))

        assert len(self.pushed_frames) == 1
        assert isinstance(self.pushed_frames[0], MetricsFrame)
        assert self.pushed_frames[0].data == [prediction]

    async def test_no_metrics_frame_when_no_prediction(self):
        """No MetricsFrame should be pushed when prediction is None."""
        self.analyzer._analyze_state = EndOfTurnState.COMPLETE
        self.analyzer._prediction = None

        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))

        assert len(self.pushed_frames) == 0

    async def test_speech_control_params_broadcast_on_start(self):
        """SpeechControlParamsFrame should be broadcast on StartFrame."""
        await self.strategy.process_frame(
            StartFrame(audio_in_sample_rate=16000, audio_out_sample_rate=16000)
        )

        assert len(self.broadcast_frames) == 1
        frame_cls, kwargs = self.broadcast_frames[0]
        assert frame_cls is SpeechControlParamsFrame

    async def test_transcription_frames_ignored(self):
        """TranscriptionFrame should not affect state or trigger turn stop."""
        self.analyzer._analyze_state = EndOfTurnState.INCOMPLETE

        await self.strategy.process_frame(
            TranscriptionFrame(text="hello", user_id="user1", timestamp="now")
        )

        assert not self.turn_stopped_called

    async def test_process_frame_returns_continue(self):
        """process_frame should always return CONTINUE."""
        from pipecat.turns.types import ProcessFrameResult

        result = await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        assert result == ProcessFrameResult.CONTINUE


if __name__ == "__main__":
    unittest.main()
