#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
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
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
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


class MockAlwaysCompleteTurnAnalyzer(BaseTurnAnalyzer):
    """Turn analyzer that always returns COMPLETE for analyze_end_of_turn."""

    def __init__(self):
        super().__init__(sample_rate=16000)
        self._params = BaseTurnParams()

    @property
    def speech_triggered(self) -> bool:
        return False

    @property
    def params(self) -> BaseTurnParams:
        return self._params

    def append_audio(self, buffer, is_speech):
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        return EndOfTurnState.COMPLETE, None

    def clear(self):
        pass


class TestTurnAnalyzerUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_strategy(self, stt_timeout=STT_TIMEOUT):
        strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=MockAlwaysCompleteTurnAnalyzer())
        await strategy.setup(self.task_manager)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=stt_timeout)
        )
        return strategy

    async def test_set_finalized(self):
        """Finalized transcript after VAD stop triggers immediately."""
        strategy = await self._create_strategy()
        triggered = False

        @strategy.event_handler("on_user_turn_stopped")
        async def on_stopped(strategy, params):
            nonlocal triggered
            triggered = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await strategy.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="", finalized=True)
        )
        self.assertTrue(triggered)

    async def test_set_non_finalized_triggers_after_timeout(self):
        """Non-finalized transcript before timeout expires waits for timeout."""
        strategy = await self._create_strategy(stt_timeout=0.15)
        triggered = False

        @strategy.event_handler("on_user_turn_stopped")
        async def on_stopped(strategy, params):
            nonlocal triggered
            triggered = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp=""))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertFalse(triggered)

        await asyncio.sleep(0.25)
        self.assertTrue(triggered)

    async def test_transcript_after_expired_timeout_triggers_immediately(self):
        """Non-finalized transcript arriving after the STT timeout has expired
        triggers the turn stop immediately.

        Reproduces a race condition where Deepgram's aggressive endpointing
        (default 10ms) flushes the final transcript before the external VAD
        fires stop_secs later. By the time VAD stop arrives, Deepgram already
        delivered the is_final transcript. The strategy's internal timeout
        (stt_timeout - stop_secs) can be 0.0s, firing instantly before the
        TranscriptionFrame is processed. The late-arriving transcript then
        has no code path to re-check _maybe_trigger_user_turn_stopped,
        leaving the turn stuck until the 5s safety-net timeout.
        """
        strategy = await self._create_strategy(stt_timeout=STT_TIMEOUT)
        triggered = False

        @strategy.event_handler("on_user_turn_stopped")
        async def on_stopped(strategy, params):
            nonlocal triggered
            triggered = True

        # S - user starts speaking
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        # E - VAD stop fires; turn analyzer returns COMPLETE;
        # timeout = max(0, 0.0 - 0.0) = 0.0s → fires instantly
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Let the 0.0s timeout task complete
        await asyncio.sleep(0.05)
        self.assertFalse(triggered)

        # T - non-finalized transcript arrives after timeout expired
        await strategy.process_frame(
            TranscriptionFrame(text="Hi, Sarah.", user_id="", timestamp="")
        )

        # Turn should stop immediately — not wait for a 5s safety-net timeout
        self.assertTrue(triggered)

    async def test_transcript_while_speaking_does_not_trigger(self):
        """Transcript during active speech must not trigger turn stop."""
        strategy = await self._create_strategy()
        triggered = False

        @strategy.event_handler("on_user_turn_stopped")
        async def on_stopped(strategy, params):
            nonlocal triggered
            triggered = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp=""))
        self.assertFalse(triggered)

    async def test_transcript_before_vad_stop_waits_for_timeout(self):
        """Transcript arriving before VAD stop doesn't trigger prematurely."""
        strategy = await self._create_strategy(stt_timeout=0.15)
        triggered = False

        @strategy.event_handler("on_user_turn_stopped")
        async def on_stopped(strategy, params):
            nonlocal triggered
            triggered = True

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(TranscriptionFrame(text="Hello!", user_id="", timestamp=""))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Turn complete is set, text is set, but timeout is still running
        self.assertFalse(triggered)

        # After timeout expires, should trigger
        await asyncio.sleep(0.25)
        self.assertTrue(triggered)


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
