#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import warnings
from unittest.mock import patch

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop import (
    BaseUserTurnStopStrategy,
    ExternalUserTurnCompletionStopStrategy,
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

AGGREGATION_TIMEOUT = 0.1
# Use 0 STT timeout for deterministic test timing
STT_TIMEOUT = 0.0


class TestSpeechTimeoutUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()

    async def _create_strategy(self, user_speech_timeout=AGGREGATION_TIMEOUT):
        """Create strategy and configure STT timeout via metadata frame."""
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=user_speech_timeout)
        await strategy.setup(frame_processor_setup(self.task_manager))
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

        # Arm for the next turn (the controller notifies stop strategies at turn start)
        await strategy.handle_user_turn_started()

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

    async def test_finalized_short_circuits_stt_wait(self):
        """Finalized transcript cancels the stt_timeout safety net.

        user_speech_timeout still runs to completion as a policy floor,
        but stt_timeout is skipped once STT says it's done. Net effect:
        the turn stops at user_speech_timeout, not stt_timeout.
        """
        stt_timeout = AGGREGATION_TIMEOUT * 4
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=stt_timeout)
        )

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S → E: starts user_speech_timeout (short) and stt_timeout (long).
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Finalized transcript arrives before user_speech_timeout elapses.
        await strategy.process_frame(
            TranscriptionFrame(text="Hello!", user_id="cat", timestamp="", finalized=True)
        )
        # user_speech_timeout is still running, so no trigger yet.
        self.assertIsNone(should_start)

        # user_speech_timeout elapses — stt_timeout was short-circuited,
        # so the turn stops now rather than waiting for stt_timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_non_finalized_waits_full_stt_timeout(self):
        """Non-finalized transcript does not short-circuit stt_timeout.

        When STT never signals finalization, the stt_timeout safety net
        must run its full course — the turn should not stop until the
        longer of the two timers has elapsed.
        """
        stt_timeout = AGGREGATION_TIMEOUT * 4
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=stt_timeout)
        )

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # S → E: both timers start.
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Non-finalized transcript during the wait.
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))

        # user_speech_timeout elapses but stt_timeout has not — no trigger.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertIsNone(should_start)

        # Wait for the remainder of stt_timeout.
        await asyncio.sleep(stt_timeout - AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_fallback_uses_only_user_speech_timeout(self):
        """Fallback path (no VAD) ignores stt_timeout and uses only user_speech_timeout.

        stt_timeout is defined as "p99 after VAD stop" — without a VAD
        reference point it has no meaning. The fallback measures
        inactivity since the last transcript, which is user_speech_timeout.
        """
        stt_timeout = AGGREGATION_TIMEOUT * 4
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(frame_processor_setup(self.task_manager))
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=stt_timeout)
        )

        should_start = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal should_start
            should_start = True

        # Transcript arrives without any VAD frame — fallback path.
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))

        # The fallback timer is user_speech_timeout, not stt_timeout.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertTrue(should_start)

    async def test_turn_start_mid_utterance_falsely_stops_turn(self):
        """Test that a mid-utterance turn start does not falsely stop the turn.

        ``UserTurnController`` calls ``handle_user_turn_started`` on all stop
        strategies when a turn starts (see ``_trigger_user_turn_start``),
        which can happen right after a ``VADUserStartedSpeakingFrame`` with
        no matching stop yet — the user is still speaking. The callback must
        preserve that live VAD state so a finalized transcript for a
        mid-utterance segment (as streaming STT services emit) is not
        treated as the no-VAD fallback and stopped by ``user_speech_timeout``
        while the user never stopped talking.
        """
        strategy = await self._create_strategy()

        stop_count = 0

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal stop_count
            stop_count += 1

        # User starts speaking; VAD reports start.
        await strategy.process_frame(VADUserStartedSpeakingFrame())

        # Turn start, as UserTurnController performs when the turn begins.
        # No VADUserStoppedSpeakingFrame has been received — the user is
        # still speaking.
        await strategy.handle_user_turn_started()

        # Streaming STT finalizes a mid-utterance segment while the user is
        # still talking (no VAD stop event was ever emitted).
        await strategy.process_frame(
            TranscriptionFrame(
                text="So I was thinking", user_id="cat", timestamp="", finalized=True
            )
        )

        # The user never stopped speaking, so the turn must not stop just
        # because user_speech_timeout elapses.
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 0)

    async def test_turn_callbacks_clear_stale_text_no_premature_stop(self):
        """Turn callbacks clear stale text and cancel timeouts, preventing premature stop.

        Reproduces the bug from issue #4053: after turn 1 completes and the
        stop callback runs, a late transcription sets _text. Arming at turn 2
        start (handle_user_turn_started) should clear the stale _text so no
        premature stop occurs on VAD stop.
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

        # Turn 1 ends (as the controller notifies stop strategies at turn stop)
        await strategy.handle_user_turn_stopped()

        # === Late transcription arrives between turns ===
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="cat", timestamp=""))

        # Turn 2 arms (the controller notifies stop strategies at turn start)
        await strategy.handle_user_turn_started()

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

    async def _create_strategy(self, stt_timeout=0.35):
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=AGGREGATION_TIMEOUT)
        await strategy.setup(frame_processor_setup(self.task_manager))
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


class TestExternalUserTurnCompletionStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()

    async def _create_strategy(self):
        strategy = ExternalUserTurnCompletionStopStrategy()
        await strategy.setup(frame_processor_setup(self.task_manager))
        return strategy

    async def test_finalizes_on_completion(self):
        """The strategy fires on_user_turn_stopped on UserTurnInferenceCompletedFrame.

        The stale-completion-while-speaking gate lives in the controller (which
        holds the authoritative user-speaking state), not in this strategy; see
        test_user_turn_controller.py.
        """
        strategy = await self._create_strategy()

        finalized = False

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal finalized
            finalized = True

        await strategy.process_frame(UserTurnInferenceCompletedFrame())
        self.assertTrue(finalized)


class TestExternalUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()

    async def _run_turn(self, strategy, *, started, stopped):
        """Drive one turn through the strategy and return the stop params."""
        params = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, stop_params):
            nonlocal params
            params = stop_params

        await strategy.process_frame(started)
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp="now"))
        await strategy.process_frame(stopped)
        return params

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

    async def test_proposal_finalizes_with_emission_enabled(self):
        strategy = ExternalUserTurnStopStrategy()
        await strategy.setup(frame_processor_setup(self.task_manager))
        params = await self._run_turn(
            strategy,
            started=ProposedUserStartedSpeakingFrame(),
            stopped=ProposedUserStoppedSpeakingFrame(),
        )
        self.assertIsNotNone(params)
        self.assertTrue(params.enable_user_speaking_frames)
        await strategy.cleanup()

    async def test_real_turn_frame_finalizes_with_emission_suppressed(self):
        strategy = ExternalUserTurnStopStrategy()
        await strategy.setup(frame_processor_setup(self.task_manager))
        params = await self._run_turn(
            strategy,
            started=UserStartedSpeakingFrame(),
            stopped=UserStoppedSpeakingFrame(),
        )
        self.assertIsNotNone(params)
        self.assertFalse(params.enable_user_speaking_frames)
        await strategy.cleanup()

    async def test_no_timer_driven_finalization_while_no_turn_is_open(self):
        """The idle timer must stay quiet between turns.

        With wait_for_transcript off (how realtime mode configures this
        strategy) the timer path would otherwise fire on every tick forever.
        """
        strategy = ExternalUserTurnStopStrategy(timeout=0.05, wait_for_transcript=False)
        await strategy.setup(frame_processor_setup(self.task_manager))

        fired = 0

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal fired
            fired += 1

        # Never started a turn: several timer ticks should pass in silence.
        await asyncio.sleep(0.3)
        self.assertEqual(fired, 0)

        # Once a turn opens, the timer finalizes it as before.
        await strategy.handle_user_turn_started()
        await strategy.process_frame(ProposedUserStoppedSpeakingFrame())
        self.assertEqual(fired, 1)

        # And falls silent again after the turn ends.
        await strategy.handle_user_turn_stopped()
        await asyncio.sleep(0.3)
        self.assertEqual(fired, 1)

        await strategy.cleanup()

    async def test_deferred_finalization_keeps_the_signals_emission_flags(self):
        """Finalization from the transcript timeout carries the right flags.

        With wait_for_transcript the stop can land from the internal timeout
        rather than the stop signal, long after the signal that determined
        whether emission is suppressed.
        """
        strategy = ExternalUserTurnStopStrategy(timeout=0.05)
        await strategy.setup(frame_processor_setup(self.task_manager))

        params = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, stop_params):
            nonlocal params
            params = stop_params

        # The controller opens the turn on every stop strategy before any
        # signals arrive; the timer path only runs while a turn is open.
        await strategy.handle_user_turn_started()

        # Stop signal first, transcript after: the stop resolves from the
        # timeout in the strategy's task handler.
        await strategy.process_frame(UserStartedSpeakingFrame())
        await strategy.process_frame(UserStoppedSpeakingFrame())
        self.assertIsNone(params)

        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp="now"))
        await asyncio.sleep(0.15)
        self.assertIsNotNone(params)
        self.assertFalse(params.enable_user_speaking_frames)
        await strategy.cleanup()

    async def test_subclass_can_delay_finalization(self):
        """A subclass can shift turn-stop timing — the motivator for proposals.

        Mirrors ``GracePeriodUserTurnStopStrategy`` from
        ``examples/turn-management/turn-management-custom-external-turn-strategy.py``.
        """

        class DelayedStopStrategy(ExternalUserTurnStopStrategy):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.pending = None

            async def process_frame(self, frame):
                if isinstance(frame, ProposedUserStartedSpeakingFrame) and self.pending:
                    task, self.pending = self.pending, None
                    await self.cancel_task(task)
                return await super().process_frame(frame)

            async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames=None):
                if self.pending:
                    return
                self.pending = self.create_task(self._finalize_later(enable_user_speaking_frames))

            async def _finalize_later(self, enable_user_speaking_frames):
                await asyncio.sleep(0.2)
                self.pending = None
                await super().trigger_user_turn_stopped(
                    enable_user_speaking_frames=enable_user_speaking_frames
                )

        strategy = DelayedStopStrategy()
        await strategy.setup(frame_processor_setup(self.task_manager))

        stop_params = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal stop_params
            stop_params = params

        await strategy.process_frame(ProposedUserStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp="now"))
        await strategy.process_frame(ProposedUserStoppedSpeakingFrame())
        # The base strategy would have finalized by now; this one holds the turn.
        self.assertIsNone(stop_params)

        # Resuming within the grace period keeps the turn open.
        await strategy.process_frame(ProposedUserStartedSpeakingFrame())
        await asyncio.sleep(0.3)
        self.assertIsNone(stop_params)

        # Falling silent again lets the delayed finalization through, still
        # carrying the emission setting the decide path handed the override.
        await strategy.process_frame(ProposedUserStoppedSpeakingFrame())
        await asyncio.sleep(0.3)
        self.assertIsNotNone(stop_params)
        self.assertTrue(stop_params.enable_user_speaking_frames)

        await strategy.cleanup()


class TestBaseUserTurnStopStrategyDeprecations(unittest.IsolatedAsyncioTestCase):
    async def _capture_params(self, strategy):
        captured = []

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            captured.append(params)

        return captured

    async def test_enable_user_speaking_frames_warns(self):
        with self.assertWarns(DeprecationWarning) as caught:
            BaseUserTurnStopStrategy(enable_user_speaking_frames=False)
        self.assertIn("enable_user_speaking_frames", str(caught.warning))

    async def test_enable_user_speaking_frames_applies(self):
        with self.assertWarns(DeprecationWarning):
            strategy = BaseUserTurnStopStrategy(enable_user_speaking_frames=False)
        captured = await self._capture_params(strategy)

        await strategy.trigger_user_turn_stopped()
        self.assertFalse(captured[0].enable_user_speaking_frames)

    async def test_omitting_enable_user_speaking_frames_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            strategy = BaseUserTurnStopStrategy()
        captured = await self._capture_params(strategy)

        await strategy.trigger_user_turn_stopped()
        self.assertTrue(captured[0].enable_user_speaking_frames)


if __name__ == "__main__":
    unittest.main()
