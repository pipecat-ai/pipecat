#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the fix of the stale transcription race condition ("one turn behind" bug).

The bug: when a late TranscriptionFrame arrived between turns, strategy._text
retained stale content into the next turn. With timeout=0 (common when
stt_p99 <= vad_stop_secs), the timeout fired with stale _text, triggering a
premature turn stop before the current TranscriptionFrame arrived.

Fix: reset _text in _handle_vad_user_started_speaking so the timeout gate
sees empty text and waits for the current turn's transcript.

Test naming convention:
  S = VADUserStartedSpeakingFrame
  E = VADUserStoppedSpeakingFrame
  T = TranscriptionFrame (finalized)
"""

import asyncio
import unittest
from typing import Optional, Tuple

import pytest

from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)
from pipecat.frames.frames import (
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

# With stt_p99=0 and stop_secs=0, timeout = max(0, 0-0) = 0.
# This matches production when stop_secs >= stt_p99 (e.g., 0.5s >= 0.35s).
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


def _tf(text: str) -> TranscriptionFrame:
    """Create a finalized TranscriptionFrame."""
    return TranscriptionFrame(text=text, user_id="user1", timestamp="", finalized=True)


# ---------------------------------------------------------------------------
# Strategy-level: _text not reset on turn start
# ---------------------------------------------------------------------------


class TestStrategyTextResetOnTurnStart(unittest.IsolatedAsyncioTestCase):
    """Verify _text is cleared on VAD start to prevent stale text gating premature stops."""

    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_strategy(self) -> TurnAnalyzerUserTurnStopStrategy:
        strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=MockTurnAnalyzer())
        await strategy.setup(self.task_manager)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=STT_TIMEOUT)
        )
        return strategy

    async def test_text_cleared_on_vad_start(self):
        """_text set by TranscriptionFrame is cleared by VADUserStartedSpeakingFrame.

        After a TranscriptionFrame sets _text, starting a new turn via VAD
        clears it so stale text cannot trigger a premature turn stop.
        """
        strategy = await self._create_strategy()

        # Set _text via a TranscriptionFrame
        await strategy.process_frame(_tf("Previous utterance"))
        self.assertEqual(strategy._text, "Previous utterance")

        # New turn starts
        await strategy.process_frame(VADUserStartedSpeakingFrame())

        # FIX: _text is cleared on turn start
        self.assertEqual(
            strategy._text,
            "",
            "_text should be '' after turn start to prevent stale text triggering premature stop",
        )

    @pytest.mark.xfail(reason="Bug fixed: _text is now cleared on VAD start", strict=True)
    async def test_stale_text_persists_after_vad_start(self):
        """REPRODUCTION: _text retains stale content after VADUserStartedSpeakingFrame.

        Before the fix, _text was not cleared on turn start, allowing stale
        content to gate a premature turn stop via timeout(0).
        """
        strategy = await self._create_strategy()

        await strategy.process_frame(_tf("Previous utterance"))
        self.assertEqual(strategy._text, "Previous utterance")

        await strategy.process_frame(VADUserStartedSpeakingFrame())

        # This assertion passes only with the bug present
        self.assertEqual(strategy._text, "Previous utterance")

    async def test_text_cleared_by_reset_only(self):
        """Only reset() clears _text — but reset() is called AFTER turn stop,
        not on turn start.
        """
        strategy = await self._create_strategy()

        await strategy.process_frame(_tf("Some text"))
        self.assertEqual(strategy._text, "Some text")

        await strategy.reset()
        self.assertEqual(strategy._text, "", "reset() correctly clears _text")


# ---------------------------------------------------------------------------
# Controller-level: full turn lifecycle showing the stale text race
# ---------------------------------------------------------------------------


class TestControllerStaleTextRaceFix(unittest.IsolatedAsyncioTestCase):
    """Full turn lifecycle verifying the stale text race is fixed.

    With the fix, _text is cleared on VAD start, so late TranscriptionFrames
    arriving between turns cannot gate a premature turn stop via timeout(0).
    """

    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_controller(
        self,
    ) -> Tuple[UserTurnController, TurnAnalyzerUserTurnStopStrategy]:
        strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=MockTurnAnalyzer())
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[strategy],
            ),
        )
        await controller.setup(self.task_manager)
        await strategy.process_frame(
            STTMetadataFrame(service_name="test", ttfs_p99_latency=STT_TIMEOUT)
        )
        return controller, strategy

    async def test_late_transcript_cleared_on_turn_start(self):
        """A late TranscriptionFrame's _text is cleared when the next turn starts.

        Flow:
        1. Turn 1: S E T → turn stops → reset() clears _text
        2. Late TranscriptionFrame arrives → _text = "Late text"
        3. Turn 2: S → _text cleared to "" (fix prevents stale gate)
        """
        controller, strategy = await self._create_controller()

        stop_count = 0

        @controller.event_handler("on_user_turn_stopped")
        async def on_stop(ctrl, strat, params):
            nonlocal stop_count
            stop_count += 1

        # --- Turn 1: normal S E T ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await controller.process_frame(_tf("Hello"))
        await asyncio.sleep(0.05)
        self.assertEqual(stop_count, 1)

        # After turn stop, controller calls reset() → _text = ""
        self.assertEqual(strategy._text, "")

        # Late TranscriptionFrame arrives between turns
        await controller.process_frame(_tf("Late text"))
        self.assertEqual(strategy._text, "Late text")

        # --- Turn 2 starts ---
        await controller.process_frame(VADUserStartedSpeakingFrame())

        # FIX: _text is cleared on turn start
        self.assertEqual(
            strategy._text,
            "",
            "_text cleared on turn start prevents stale text from gating premature stop",
        )

    async def test_timeout_waits_for_current_transcription(self):
        """timeout(0) does NOT trigger premature turn stop when _text is cleared on turn start.

        With the fix, after a late transcript sets _text, the next turn's VAD start
        clears _text. When timeout(0) fires, _maybe_trigger_user_turn_stopped sees
        empty _text and waits for the current TranscriptionFrame.
        """
        controller, strategy = await self._create_controller()

        stop_count = 0

        @controller.event_handler("on_user_turn_stopped")
        async def on_stop(ctrl, strat, params):
            nonlocal stop_count
            stop_count += 1

        # --- Turn 1: normal S E T ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await controller.process_frame(_tf("What time is it?"))
        await asyncio.sleep(0.05)
        self.assertEqual(stop_count, 1)

        # --- Late transcript arrives (e.g., Soniox second sentence endpoint) ---
        await controller.process_frame(_tf("I'll wait on site."))

        # --- Turn 2: S E (TranscriptionFrame not yet arrived) ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        # FIX: _text is cleared on turn start
        self.assertEqual(strategy._text, "")

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        # analyze_end_of_turn → COMPLETE, timeout(0) created

        # Let the timeout fire
        await asyncio.sleep(0.1)

        # FIX: Timeout fires but _text is empty → no premature turn stop
        self.assertEqual(
            stop_count,
            1,
            "No premature stop — timeout sees empty _text and waits for current transcript",
        )

        # Actual TranscriptionFrame arrives → triggers turn stop with correct content
        await controller.process_frame(_tf("John Smith"))
        await asyncio.sleep(0.05)
        self.assertEqual(stop_count, 2, "Turn stops with current transcript, not stale")

    async def test_three_turn_sequence_no_longer_one_behind(self):
        """Three-turn sequence where each turn correctly waits for its own transcript.

        With the fix, the "one turn behind" pattern is broken:
        - Turn 1: correctly processed
        - Late transcript: _text set but cleared on next turn start
        - Turn 2: waits for current transcript → correct
        - Turn 3: waits for current transcript → correct
        """
        controller, strategy = await self._create_controller()

        stop_count = 0

        @controller.event_handler("on_user_turn_stopped")
        async def on_stop(ctrl, strat, params):
            nonlocal stop_count
            stop_count += 1

        # --- Turn 1: "What time?" → correctly processed ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await controller.process_frame(_tf("What time?"))
        await asyncio.sleep(0.05)
        self.assertEqual(stop_count, 1)

        # --- Late transcript arrives between turns ---
        await controller.process_frame(_tf("I'll wait."))

        # --- Turn 2: User says "John Smith" ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(strategy._text, "", "_text cleared on turn start")

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.1)
        # FIX: no premature stop — timeout sees empty _text
        self.assertEqual(stop_count, 1, "no premature stop from stale _text")

        # Actual TranscriptionFrame arrives → correct turn stop
        await controller.process_frame(_tf("John Smith"))
        await asyncio.sleep(0.05)
        self.assertEqual(stop_count, 2, "turn 2 stops with correct transcript")

        # --- Turn 3: User says "No thanks" ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(strategy._text, "", "_text cleared on turn 3 start")

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.1)
        self.assertEqual(stop_count, 2, "no premature stop")

        await controller.process_frame(_tf("No thanks"))
        await asyncio.sleep(0.05)
        self.assertEqual(stop_count, 3, "turn 3 stops with correct transcript")


if __name__ == "__main__":
    unittest.main()
