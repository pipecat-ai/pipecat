#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for VADUserTurnStopStrategy."""

import unittest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.vad_user_turn_stop_strategy import VADUserTurnStopStrategy
from pipecat.utils.asyncio.task_manager import TaskManager


class TestVADUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.strategy = VADUserTurnStopStrategy()
        self.task_manager = TaskManager()
        await self.strategy.setup(self.task_manager)

        self.turn_stopped_called = False

        @self.strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            self.turn_stopped_called = True

    async def asyncTearDown(self):
        await self.strategy.cleanup()

    async def test_vad_stop_triggers_user_turn_stopped(self):
        """VADUserStoppedSpeakingFrame should trigger user turn stopped."""
        await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))

        assert self.turn_stopped_called

    async def test_vad_start_does_not_trigger(self):
        """VADUserStartedSpeakingFrame should not trigger user turn stopped."""
        await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))

        assert not self.turn_stopped_called

    async def test_transcription_frames_ignored(self):
        """TranscriptionFrame should not trigger user turn stopped."""
        await self.strategy.process_frame(
            TranscriptionFrame(text="hello", user_id="user1", timestamp="now")
        )

        assert not self.turn_stopped_called

    async def test_input_audio_frames_ignored(self):
        """InputAudioRawFrame should not trigger user turn stopped."""
        await self.strategy.process_frame(
            InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)
        )

        assert not self.turn_stopped_called

    async def test_process_frame_returns_continue(self):
        """process_frame should always return CONTINUE."""
        result = await self.strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.3))
        assert result == ProcessFrameResult.CONTINUE

        result = await self.strategy.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        assert result == ProcessFrameResult.CONTINUE

    async def test_reset_does_not_crash(self):
        """reset() should complete without error."""
        await self.strategy.reset()


if __name__ == "__main__":
    unittest.main()
