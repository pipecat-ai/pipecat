#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TurnAnalyzerUserTurnStopStrategy.

Reproduces the deadlock described in https://github.com/pipecat-ai/pipecat/issues/3988:

When an STT service delays finalized transcriptions for short utterances, the
stop strategy can never trigger because it only populates self._text from
TranscriptionFrame, ignoring InterimTranscriptionFrame. The VAD stop sets
_turn_complete=True, but _text stays empty, so _maybe_trigger_user_turn_stopped()
returns early and the bot goes silent.

Test naming convention follows the existing pattern in test_user_turn_stop_strategy.py:
  S = VADUserStartedSpeakingFrame
  I = InterimTranscriptionFrame
  E = VADUserStoppedSpeakingFrame
  T = TranscriptionFrame (finalized)
"""

import asyncio
import unittest
from typing import Optional, Tuple

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

AGGREGATION_TIMEOUT = 0.1
STT_TIMEOUT = 0.0


class MockTurnAnalyzer(BaseTurnAnalyzer):
    """Mock turn analyzer that always returns COMPLETE on analyze_end_of_turn."""

    def __init__(self):
        super().__init__()
        self._speech_triggered = False

    @property
    def speech_triggered(self) -> bool:
        return self._speech_triggered

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        return (EndOfTurnState.COMPLETE, None)

    def clear(self):
        self._speech_triggered = False


class TestTurnAnalyzerUserTurnStopStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_strategy(self):
        """Create strategy with mock turn analyzer and configure STT timeout."""
        strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=MockTurnAnalyzer())
        await strategy.setup(self.task_manager)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=STT_TIMEOUT)
        )
        return strategy

    async def test_sie_t_basic_flow(self):
        """S I E T: Basic flow — interim arrives during speech, finalized after VAD stop.

        Verifies the normal happy path works with TurnAnalyzerUserTurnStopStrategy.
        """
        strategy = await self._create_strategy()

        turn_stopped = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal turn_stopped
            turn_stopped = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(turn_stopped)

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Billing", user_id="user1", timestamp="")
        )
        self.assertIsNone(turn_stopped)

        # E — turn analyzer returns COMPLETE
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        self.assertIsNone(turn_stopped)

        # T (finalized) — triggers immediately
        await strategy.process_frame(
            TranscriptionFrame(text="Billing.", user_id="user1", timestamp="", finalized=True)
        )
        self.assertTrue(turn_stopped)

    async def test_sie_no_transcription_deadlock(self):
        """S I E (no T): Reproduces the deadlock from issue #3988.

        Scenario: User says a short word like "Billing." The STT emits an
        InterimTranscriptionFrame but delays the finalized TranscriptionFrame
        until more speech arrives. The VAD stop fires and the turn analyzer
        says COMPLETE, but _text is empty because process_frame only handles
        TranscriptionFrame. The turn never completes.

        Expected: on_user_turn_stopped should fire (via timeout or interim text),
        but currently it does NOT — this test documents the deadlock.
        """
        strategy = await self._create_strategy()

        turn_stopped = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal turn_stopped
            turn_stopped = True

        # S — user starts speaking
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        self.assertIsNone(turn_stopped)

        # I — STT emits interim transcription for the short utterance
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Billing", user_id="user1", timestamp="")
        )
        self.assertIsNone(turn_stopped)

        # E — VAD stop fires, turn analyzer returns COMPLETE
        # At this point _turn_complete=True but _text="" (no TranscriptionFrame received)
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Wait for STT timeout to elapse
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)

        # BUG: on_user_turn_stopped never fires because _text is empty.
        # The bot is now deadlocked — silent until the user speaks again.
        #
        # If this bug is fixed (e.g. by also populating _text from
        # InterimTranscriptionFrame), change assertIsNone to assertTrue.
        self.assertIsNone(turn_stopped)

    async def test_se_no_transcription_timeout(self):
        """S E (no T, no I): VAD fires but STT produces nothing.

        When there's no transcription at all (no interim, no finalized), the
        timeout fires but _maybe_trigger_user_turn_stopped returns early
        because _text is empty. This is arguably correct behavior (VAD false
        positive), but included to show the timeout alone doesn't help.
        """
        strategy = await self._create_strategy()

        turn_stopped = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal turn_stopped
            turn_stopped = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())

        # E — turn analyzer returns COMPLETE
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Wait for timeout
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)

        # No transcription at all — timeout fires but _text is empty, so no trigger.
        # This is expected: no speech was transcribed.
        self.assertIsNone(turn_stopped)

    async def test_sie_delayed_t_recovers(self):
        """S I E ... T: Delayed finalized transcription eventually recovers.

        Shows that if the finalized TranscriptionFrame eventually arrives
        (e.g. when user speaks again), the turn does complete. But the delay
        can be 15+ seconds in production — far too long.
        """
        strategy = await self._create_strategy()

        turn_stopped = None

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(strategy, params):
            nonlocal turn_stopped
            turn_stopped = True

        # S
        await strategy.process_frame(VADUserStartedSpeakingFrame())

        # I
        await strategy.process_frame(
            InterimTranscriptionFrame(text="Billing", user_id="user1", timestamp="")
        )

        # E — turn analyzer returns COMPLETE, timeout starts
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # Timeout elapses — still no trigger (no _text)
        await asyncio.sleep(AGGREGATION_TIMEOUT + 0.1)
        self.assertIsNone(turn_stopped)

        # T — finalized transcription finally arrives (e.g. 15s later in production)
        await strategy.process_frame(
            TranscriptionFrame(text="Billing.", user_id="user1", timestamp="", finalized=True)
        )

        # Now it triggers because _text is set and _turn_complete was already True
        self.assertTrue(turn_stopped)


if __name__ == "__main__":
    unittest.main()
