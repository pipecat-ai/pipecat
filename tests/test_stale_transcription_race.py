#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for stale transcription race condition causing "one turn behind" behavior.

Reproduces a bug where the bot responds to the PREVIOUS user utterance instead
of the current one. Observed in production with Soniox STT + TurnAnalyzer.

Root cause: a race between the TurnAnalyzerUserTurnStopStrategy's STT timeout
(which fires with timeout=0 when stt_p99 <= vad_stop_secs) and the arrival of
the current turn's TranscriptionFrame.

Bug chain:
1. Turn N ends → push_aggregation → _aggregation cleared
2. Late TranscriptionFrame from STT arrives → accumulated in _aggregation
3. Turn N+1 starts → _aggregation is NOT cleared
4. VAD stop → turn analyzer COMPLETE → timeout(0) fires
5. Timeout checks strategy._text (stale, not reset on turn start) → non-empty
6. trigger_user_turn_stopped → push_aggregation pushes stale _aggregation
7. Strategy reset clears _turn_complete
8. Current TranscriptionFrame arrives → accumulated but _turn_complete=False → no push
9. Current text becomes stale for turn N+2 → cycle repeats

Contributing factors:
- _handle_vad_user_started_speaking does NOT reset self._text
- timeout = max(0, stt_p99 - vad_stop_secs) = 0 when stop_secs >= stt_p99

Test naming convention:
  S = VADUserStartedSpeakingFrame
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


class TestStrategyStaleTextNotResetOnTurnStart(unittest.IsolatedAsyncioTestCase):
    """_handle_vad_user_started_speaking does NOT reset self._text.

    This means _text carries stale content into the new turn, which the
    timeout(0) uses to trigger a premature turn stop.
    """

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

    async def test_text_persists_after_vad_start(self):
        """_text set by TranscriptionFrame is NOT cleared by VADUserStartedSpeakingFrame.

        After a TranscriptionFrame sets _text, starting a new turn via VAD
        should clear it — but it doesn't. This is the first enabler of the bug.
        """
        strategy = await self._create_strategy()

        # Set _text via a TranscriptionFrame
        await strategy.process_frame(_tf("Previous utterance"))
        self.assertEqual(strategy._text, "Previous utterance")

        # New turn starts
        await strategy.process_frame(VADUserStartedSpeakingFrame())

        # BUG: _text still has old content
        self.assertEqual(
            strategy._text,
            "Previous utterance",
            "_text should be '' after turn start but retains stale content (this is the bug)",
        )

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


class TestControllerStaleTextRace(unittest.IsolatedAsyncioTestCase):
    """Full turn lifecycle with UserTurnController showing the race condition.

    When a TranscriptionFrame arrives between turns (after turn stop reset
    but before the next turn start), _text gets set. On the next turn,
    _text is not reset by VAD start. When the timeout(0) fires, it sees
    stale _text and triggers a premature turn stop.
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

    async def test_late_transcript_sets_stale_text_between_turns(self):
        """A TranscriptionFrame arriving after turn stop sets _text to stale content.

        Flow:
        1. Turn 1: S E T → turn stops → reset() clears _text
        2. Late TranscriptionFrame arrives → _text = "Late text"
        3. Turn 2: S → _text still = "Late text" (not reset)
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

        # BUG: _text retains stale content from the late transcript
        self.assertEqual(
            strategy._text,
            "Late text",
            "Stale _text persists into turn 2 (root cause of one-turn-behind bug)",
        )

    async def test_timeout_fires_before_transcription_arrives(self):
        """timeout(0) triggers turn stop before the actual TranscriptionFrame arrives.

        This is the core reproduction: after a late transcript sets _text,
        the next turn's VAD stop creates timeout(0), which fires with stale
        _text and triggers a premature turn stop. The actual TranscriptionFrame
        for the current turn arrives too late.
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
        # At this point: _text = "I'll wait on site." (stale, not reset)
        self.assertEqual(strategy._text, "I'll wait on site.")

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        # analyze_end_of_turn → COMPLETE, timeout(0) created

        # Let the timeout fire
        await asyncio.sleep(0.1)

        # BUG: Timeout fires with stale _text → premature turn stop.
        # In the real pipeline, this pushes the stale _aggregation content
        # instead of waiting for the current TranscriptionFrame.
        self.assertEqual(
            stop_count,
            2,
            "Timeout fires with stale _text before actual TranscriptionFrame arrives",
        )

        # Now the actual TranscriptionFrame arrives — but the turn is already stopped.
        # _turn_complete was cleared by reset(), so this can't trigger another stop.
        await controller.process_frame(_tf("Kalle Anka"))
        self.assertEqual(strategy._text, "Kalle Anka")

        # "Kalle Anka" is now stale _text for turn 3 → the cycle continues.

    async def test_one_turn_behind_three_turn_sequence(self):
        """Three-turn sequence demonstrating the repeating "one turn behind" pattern.

        Matches the production log pattern:
        - Turn 1: correctly processed
        - Late transcript: poisons _text
        - Turn 2: premature stop (stale), actual text becomes stale
        - Turn 3: premature stop (stale from turn 2), actual text becomes stale
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

        # --- Late transcript poisons _text ---
        await controller.process_frame(_tf("I'll wait."))

        # --- Turn 2: User says "Kalle Anka" but stale text fires first ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(strategy._text, "I'll wait.", "stale _text enters turn 2")

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.1)
        self.assertEqual(stop_count, 2, "premature stop from stale _text")

        # Actual TranscriptionFrame arrives late → poisons _text for turn 3
        await controller.process_frame(_tf("Kalle Anka"))

        # --- Turn 3: same pattern repeats ---
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(strategy._text, "Kalle Anka", "stale _text from turn 2 enters turn 3")

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.1)
        self.assertEqual(stop_count, 3, "premature stop repeats — one turn behind continues")

        # Actual turn 3 transcript arrives late → becomes stale for turn 4
        await controller.process_frame(_tf("No thanks"))
        self.assertEqual(strategy._text, "No thanks", "stale for the next turn")


if __name__ == "__main__":
    unittest.main()
