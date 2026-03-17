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
from pipecat.turns.user_start.base_user_turn_start_strategy import ProcessFrameResult
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

    async def test_initial_state_is_listening(self):
        strategy = self._create_strategy()
        self.assertEqual(strategy.state, _WakeState.LISTENING)

    async def test_wake_phrase_in_final_transcription(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_wake_phrase_in_interim_transcription(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            InterimTranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_no_wake_phrase_returns_break(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="hello world", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.BREAK)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        await strategy.cleanup()

    async def test_vad_frame_returns_break_in_listening(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(result, ProcessFrameResult.BREAK)
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
        self.assertEqual(result, ProcessFrameResult.BREAK)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        result = await strategy.process_frame(
            TranscriptionFrame(text="pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_use_interim_false_ignores_interim(self):
        strategy = self._create_strategy(use_interim=False)
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            InterimTranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.BREAK)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        # Final transcription should still work.
        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_multiple_phrases(self):
        strategy = self._create_strategy(phrases=["hey pipecat", "ok computer"])
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="ok computer", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_case_insensitive_matching(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="HEY PIPECAT", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_reset_preserves_listening_state(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        self.assertEqual(strategy.state, _WakeState.LISTENING)
        await strategy.reset()
        self.assertEqual(strategy.state, _WakeState.LISTENING)

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

    async def test_accumulated_text_capped(self):
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        # Send lots of text to exceed the 500 char cap.
        for _ in range(20):
            await strategy.process_frame(
                TranscriptionFrame(text="some random words " * 5, user_id="user1", timestamp="")
            )

        # Accumulated text should be capped.
        self.assertLessEqual(len(strategy._accumulated_text), 510)

        await strategy.cleanup()

    async def test_wake_phrase_with_extra_text(self):
        """Wake phrase embedded in a longer sentence."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(
                text="I said hey pipecat what time is it", user_id="user1", timestamp=""
            )
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_reawaken_after_timeout(self):
        """After timeout, wake phrase can be spoken again."""
        strategy = self._create_strategy(timeout=0.1)
        await self._setup_strategy(strategy)

        # First activation.
        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        # Wait for timeout.
        await asyncio.sleep(0.3)
        self.assertEqual(strategy.state, _WakeState.LISTENING)

        # Second activation.
        result = await strategy.process_frame(
            TranscriptionFrame(text="hey pipecat", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_punctuation_stripped(self):
        """STT punctuation like 'Hey, Pipecat!' should still match."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="Hey, Pipecat!", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()

    async def test_punctuation_stripped_across_frames(self):
        """Punctuation stripped when accumulating across frames."""
        strategy = self._create_strategy()
        await self._setup_strategy(strategy)

        result = await strategy.process_frame(
            TranscriptionFrame(text="Hey,", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.BREAK)

        result = await strategy.process_frame(
            TranscriptionFrame(text="Pipecat!", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.TRIGGERED)
        self.assertEqual(strategy.state, _WakeState.INACTIVE)

        await strategy.cleanup()


if __name__ == "__main__":
    unittest.main()
