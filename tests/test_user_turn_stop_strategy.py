#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    PreemptiveUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

AGGREGATION_TIMEOUT = 0.1
# Use 0 STT timeout for deterministic test timing
STT_TIMEOUT = 0.0


class TestSpeechTimeoutUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_strategy(self, user_speech_timeout=AGGREGATION_TIMEOUT):
        """Create strategy and configure STT timeout via metadata frame."""
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=user_speech_timeout)
        await strategy.setup(self.task_manager)
        # Set STT timeout via metadata frame (as would happen in real pipeline)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=STT_TIMEOUT)
        )
        return strategy

    async def test_ste(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Transcription came in between user started/stopped. Now we wait for
        # timeout before triggering.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_site(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Transcription came in between user started/stopped. Now we wait for
        # timeout before triggering.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_st1iest2e(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Now we wait for timeout before triggering.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_siet(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_sieit(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_set(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_seit(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_st1et2(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Transcription came between user start/stopped speaking, wait for timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)
        should_start = None

        # Reset for next turn (in real usage, UserTurnController would do this)
        await strategy.reset()

        # S - new turn starts
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_set1t2(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_siet1it2(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T1
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T2
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # Transcription comes after user stopped speaking, we need to wait for
        # at least the aggregation timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_t(self):
        """Transcription without VAD - uses fallback timeout."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # Transcription without VAD triggers fallback timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_it(self):
        """Interim + Transcription without VAD - uses fallback timeout."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # Transcription without VAD triggers fallback timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_sie_delay_it(self):
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Delay - timeout expires but no transcript yet
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        # Still no trigger because no transcript received
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )

        # T (finalized) - triggers immediately since timeout already elapsed
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="", finalized=True)
        )

        # Finalized transcript received after timeout, triggers immediately
        self.assertTrue(should_start)


class TestPreemptiveUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_strategy(self):
        """Create strategy and configure STT timeout via metadata frame."""
        strategy = PreemptiveUserTurnStopStrategy()
        await strategy.setup(self.task_manager)
        # Set STT timeout via metadata frame (as would happen in real pipeline)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=STT_TIMEOUT)
        )
        return strategy

    async def test_ste(self):
        """VAD start → Transcription → VAD stop: fires immediately on stop."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E - triggers immediately since we have text
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)

    async def test_set(self):
        """VAD start → VAD stop → Transcription: fires immediately on transcription."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # T - triggers immediately since VAD already stopped
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_seit(self):
        """VAD start → VAD stop → Interim → Transcription: fires immediately on transcription."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # I - interim transcription, not a TranscriptionFrame
        await strategy.process_frame(
            InterimTranscriptionFrame(text="How", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T - triggers immediately since VAD already stopped
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertTrue(should_start)

    async def test_site(self):
        """VAD start → Interim → Transcription → VAD stop: fires immediately on stop."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Hello!", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        # T
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # E - triggers immediately since we have text
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)

    async def test_se_no_text(self):
        """VAD start → VAD stop with no transcription: does NOT fire."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(should_start)

        # Wait for any fallback timeout to expire
        await asyncio.sleep(0.2)
        self.assertIsNone(should_start)

    async def test_st1et2(self):
        """Two clean turns: fires on each VAD stop, resets between turns."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # Turn 1: S → T1 → E
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)

        # Reset for next turn
        should_start = None
        await strategy.reset()

        # Turn 2: S → T2 → E
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertIsNone(should_start)

        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)

    async def test_t(self):
        """Transcription without VAD: fires after fallback timeout."""
        strategy = await self._create_strategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # T - no VAD, starts fallback timeout
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertIsNone(should_start)

        # Fallback timeout fires (STT_TIMEOUT is 0, so it fires quickly)
        await asyncio.sleep(0.1)
        self.assertTrue(should_start)

    async def test_set1t2(self):
        """VAD start → VAD stop → T1 → T2: fires on T1 immediately, ignores T2."""
        strategy = await self._create_strategy()

        trigger_count = 0

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal trigger_count
            trigger_count += 1

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(trigger_count, 0)

        # E
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertEqual(trigger_count, 0)

        # T1 - triggers immediately
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        self.assertEqual(trigger_count, 1)

        # T2 - should NOT trigger again (already triggered this turn)
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertEqual(trigger_count, 1)


class TestExternalUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_external_strategy(self):
        strategy = ExternalUserTurnStopStrategy()

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(UserStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(UserStoppedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(UserStartedSpeakingFrame())
        self.assertFalse(should_start)

        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertFalse(should_start)

        await strategy.process_frame(UserStoppedSpeakingFrame())
        self.assertTrue(should_start)


if __name__ == "__main__":
    unittest.main()
