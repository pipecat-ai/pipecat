#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import patch

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy, SpeechTimeoutUserTurnStopStrategy
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

    async def test_sie_delay_t(self):
        """Non-finalized transcript arriving after timeout triggers immediately."""
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
        # Still no trigger because no finalized transcript received
        self.assertIsNone(should_start)

        # T (non-finalized) - triggers immediately since timeout already elapsed
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))

        # Non-finalized transcript received after timeout, triggers immediately
        self.assertTrue(should_start)

    async def test_reset_clears_stale_text_no_premature_stop(self):
        """Test that reset() clears stale text and cancels timeout, preventing premature stop.

        Reproduces the bug from issue #4053: after turn 1 completes and
        reset() is called, a late transcription sets _text. If reset() is
        called again at turn 2 start, the stale _text should be cleared
        so no premature stop occurs on VAD stop.
        """
        strategy = await self._create_strategy()

        stop_count = 0

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal stop_count
            stop_count += 1

        # === Turn 1: S-T-E ===
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 1)

        # Reset after turn 1 (as controller would do at turn stop)
        await strategy.reset()

        # === Late transcription arrives between turns ===
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))

        # Reset at turn 2 start (the fix: controller now resets stop strategies at turn start)
        await strategy.reset()

        # === Turn 2: S-T-E (transcription arrives during turn) ===
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Wait for timeout — should get turn 2 stop with the real transcription
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 2)


class TestSpeechTimeoutStopSecsWarnings(unittest.IsolatedAsyncioTestCase):
    """Tests for stop_secs misconfiguration warnings."""

    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_strategy(self, stt_timeout=0.35):
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(self.task_manager)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=stt_timeout)
        )
        return strategy

    @patch("pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy.logger")
    async def test_warns_on_non_default_stop_secs(self, mock_logger):
        # Use high stt_timeout so only Warning A fires (stop_secs < stt_timeout)
        strategy = await self._create_strategy(stt_timeout=1.0)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))

        mock_logger.warning.assert_called_once()
        self.assertIn("differs from the recommended default", mock_logger.warning.call_args[0][0])

    @patch("pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy.logger")
    async def test_warns_on_stop_secs_gte_stt_timeout(self, mock_logger):
        strategy = await self._create_strategy(stt_timeout=0.35)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))

        # Both warnings fire: non-default stop_secs AND stop_secs >= stt_timeout
        self.assertEqual(mock_logger.warning.call_count, 2)
        self.assertIn("collapsed to 0s", mock_logger.warning.call_args_list[1][0][0])

    @patch("pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy.logger")
    async def test_warns_only_once(self, mock_logger):
        # Use high stt_timeout so only Warning A fires
        strategy = await self._create_strategy(stt_timeout=1.0)

        # First VAD stop — triggers warning
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))
        self.assertEqual(mock_logger.warning.call_count, 1)

        # Second VAD stop — no duplicate warning
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))
        self.assertEqual(mock_logger.warning.call_count, 1)

    @patch("pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy.logger")
    async def test_warning_resets_on_new_stt_metadata(self, mock_logger):
        # Use high stt_timeout so only Warning A fires
        strategy = await self._create_strategy(stt_timeout=1.0)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))
        self.assertEqual(mock_logger.warning.call_count, 1)

        # New STTMetadataFrame resets the warned flag
        await strategy.process_frame(STTMetadataFrame(service_name="test", ttfs_p99_latency=1.0))

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.5))
        self.assertEqual(mock_logger.warning.call_count, 2)

    @patch("pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy.logger")
    async def test_no_warning_on_default_stop_secs(self, mock_logger):
        strategy = await self._create_strategy()

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))

        mock_logger.warning.assert_not_called()


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
