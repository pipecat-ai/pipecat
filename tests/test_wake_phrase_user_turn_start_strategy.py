#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.turns.process_frame_result import ProcessFrameResult
from pipecat.turns.user_start.wake_phrase_user_turn_start_strategy import (
    WakePhraseUserTurnStartStrategy,
    _WakeState,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


class TestWakePhraseUserTurnStartStrategy(unittest.IsolatedAsyncioTestCase):
    def _create_strategy(self, **kwargs) -> WakePhraseUserTurnStartStrategy:
        kwargs.setdefault("phrases", ["hey pipecat"])
        kwargs.setdefault("timeout", 10.0)
        return WakePhraseUserTurnStartStrategy(**kwargs)

    async def _setup_strategy(self, strategy: WakePhraseUserTurnStartStrategy):
        task_manager = TaskManager()
        loop = asyncio.get_running_loop()
        task_manager.setup(TaskManagerParams(loop=loop))
        await strategy.setup(task_manager)
        return task_manager

    async def test_wake_phrase_in_final_transcription(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_interim_transcription_ignored(self):
        """Interim transcriptions are never used for wake phrase matching."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            InterimTranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        await strategy.cleanup()

    async def test_no_wake_phrase_returns_stop(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hello world", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        await strategy.cleanup()

    async def test_non_matching_text_resets_aggregation(self):
        """Non-matching transcription triggers aggregation reset to prevent LLM context pollution."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        reset_called = False

        @strategy.event_handler("on_reset_aggregation")
        async def on_reset_aggregation(strategy):
            nonlocal reset_called
            reset_called = True

        await strategy.process_frame(
            TranscriptionFrame(text="hello world", user_id="user1", timestamp="")
        )
        self.assertTrue(reset_called)

        await strategy.cleanup()

    async def test_vad_frame_returns_stop_in_listening(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        await strategy.cleanup()

    async def test_inactive_returns_continue(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        # Trigger wake phrase first.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Subsequent frames should return CONTINUE.
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.CONTINUE)

        result = await strategy.process_frame(
            TranscriptionFrame(text="what is the weather", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.CONTINUE)

        await strategy.cleanup()

    async def test_accumulation_across_frames(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hey", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        result = await strategy.process_frame(
            TranscriptionFrame(text="pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_multiple_phrases(self):
        strategy = self._create_strategy(phrases=["hey pipecat", "ok computer"])
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="ok computer", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_punctuation_stripped(self):
        """STT punctuation like 'Hey, Pipecat!' should still match."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="Hey, Pipecat!", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_reset_preserves_inactive_state(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.reset()
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_timeout_returns_to_listening(self):
        strategy = self._create_strategy(timeout=0.1)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Wait for timeout to expire.
        await asyncio.sleep(0.3)

        self.assertEqual(strategy.state, _WakeState.LISTENING)

        await strategy.cleanup()

    async def test_activity_refreshes_timeout(self):
        strategy = self._create_strategy(timeout=0.2)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Send activity before timeout.
        await asyncio.sleep(0.1)
        await strategy.process_frame(UserSpeakingFrame())
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Send more activity.
        await asyncio.sleep(0.1)
        await strategy.process_frame(BotSpeakingFrame())
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Wait for timeout to expire after last activity.
        await asyncio.sleep(0.3)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        await strategy.cleanup()

    async def test_wake_phrase_detected_event(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        detected_phrase = None

        @strategy.event_handler("on_wake_phrase_detected")
        async def on_wake_phrase_detected(strategy, phrase):
            nonlocal detected_phrase
            detected_phrase = phrase

        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )

        # Event fires in a background task, give it a moment.
        await asyncio.sleep(0.05)
        self.assertEqual(detected_phrase, "hey pipecat")

        await strategy.cleanup()

    async def test_wake_phrase_timeout_event(self):
        strategy = self._create_strategy(timeout=0.1)
        await self._setup_strategy(strategy)

        timeout_fired = False

        @strategy.event_handler("on_wake_phrase_timeout")
        async def on_wake_phrase_timeout(strategy):
            nonlocal timeout_fired
            timeout_fired = True

        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )

        # Wait for timeout.
        await asyncio.sleep(0.3)
        self.assertTrue(timeout_fired)

        await strategy.cleanup()

    async def test_single_activation_resets_to_listening(self):
        """In single activation mode, reset() returns to LISTENING."""
        strategy = self._create_strategy(single_activation=True)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Simulate turn start (controller calls reset on all start strategies).
        await strategy.reset()
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        # Frames should now be blocked again.
        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.STOP)

        await strategy.cleanup()

    async def test_single_activation_requires_wake_phrase_each_turn(self):
        """Single activation mode requires wake phrase for each new turn."""
        strategy = self._create_strategy(single_activation=True)
        await self._setup_strategy(strategy)

        # First turn: wake phrase -> INACTIVE -> reset -> LISTENING.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.INACTIVE)
        await strategy.reset()
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        # Without wake phrase, frames are blocked.
        result = await strategy.process_frame(
            TranscriptionFrame(text="what is the weather", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)

        # Second turn: wake phrase again.
        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_single_activation_no_timeout_task(self):
        """Single activation mode does not start a timeout task."""
        strategy = self._create_strategy(single_activation=True, timeout=0.1)
        await self._setup_strategy(strategy)

        # Trigger wake phrase.
        await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Wait longer than timeout -- should NOT return to LISTENING.
        await asyncio.sleep(0.3)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()


if __name__ == "__main__":
    unittest.main()
