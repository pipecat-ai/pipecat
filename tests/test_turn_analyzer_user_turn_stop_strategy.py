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
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.turns.user_idle_controller import UserIdleController
from pipecat.turns.user_start.transcription_user_turn_start_strategy import (
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

AGGREGATION_TIMEOUT = 0.1
STT_TIMEOUT = 0.0
# Short controller-level timeout for fast tests (normally 5.0s)
CONTROLLER_STOP_TIMEOUT = 0.2
# Short idle timeout for fast tests (normally seconds)
IDLE_TIMEOUT = 0.2


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


# ---------------------------------------------------------------------------
# Strategy-level tests: TurnAnalyzerUserTurnStopStrategy in isolation
# ---------------------------------------------------------------------------


class TestTurnAnalyzerStopStrategy(unittest.IsolatedAsyncioTestCase):
    """Tests for TurnAnalyzerUserTurnStopStrategy in isolation."""

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


# ---------------------------------------------------------------------------
# Controller-level tests: full turn lifecycle with UserTurnController
# ---------------------------------------------------------------------------


class TestControllerTimeoutWithInterimDeadlock(unittest.IsolatedAsyncioTestCase):
    """Tests for UserTurnController behavior when the stop strategy is stuck.

    Mark Backman noted that user_turn_stop_timeout (default 5s) should act as a
    safety net when no stop strategy triggers. These tests verify whether that
    timeout actually fires in the interim-only deadlock scenario.
    """

    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def _create_controller(self):
        """Create controller with VAD + Transcription start and TurnAnalyzer stop."""
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(),
                    TranscriptionUserTurnStartStrategy(enable_interruptions=True),
                ],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(turn_analyzer=MockTurnAnalyzer()),
                ],
            ),
            user_turn_stop_timeout=CONTROLLER_STOP_TIMEOUT,
        )
        await controller.setup(self.task_manager)
        return controller

    async def test_controller_timeout_fires_for_interim_only_turn(self):
        """Full scenario: Turn 1 completes normally, then a delayed interim
        starts a ghost turn that the stop strategy can't close.

        The controller's user_turn_stop_timeout should fire as a safety net,
        ending the stuck turn. This is the fallback Mark described.

        Sequence:
        1. Turn 1: S E T → completes normally via stop strategy
        2. Delayed I arrives → TranscriptionUserTurnStartStrategy fires new turn
        3. No VAD, no finalized T → stop strategy is stuck
        4. Controller timeout should fire after CONTROLLER_STOP_TIMEOUT
        """
        controller = await self._create_controller()

        turn_started_count = 0
        turn_stopped_count = 0
        timeout_fired = False

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal turn_started_count
            turn_started_count += 1

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal turn_stopped_count
            turn_stopped_count += 1

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            nonlocal timeout_fired
            timeout_fired = True

        # --- Turn 1: normal flow ---

        # S — VAD start triggers turn start
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(turn_started_count, 1)

        # I — interim arrives during speech
        await controller.process_frame(
            InterimTranscriptionFrame(text="I need help", user_id="user1", timestamp="")
        )

        # E — VAD stop
        await controller.process_frame(VADUserStoppedSpeakingFrame())

        # T — finalized transcription arrives, stop strategy fires
        await controller.process_frame(
            TranscriptionFrame(
                text="I need help with my order.", user_id="user1", timestamp="", finalized=True
            )
        )
        # Allow the strategy's trigger to propagate
        await asyncio.sleep(0.05)
        self.assertEqual(turn_stopped_count, 1)

        # --- Ghost turn: delayed interim triggers new turn ---

        # A delayed InterimTranscriptionFrame arrives (STT catching up)
        # TranscriptionUserTurnStartStrategy fires → new turn starts
        await controller.process_frame(
            InterimTranscriptionFrame(text="Billing", user_id="user1", timestamp="")
        )
        self.assertEqual(turn_started_count, 2)

        # No VAD start/stop, no finalized transcription — stop strategy is stuck.
        # The controller timeout (CONTROLLER_STOP_TIMEOUT) should fire.
        await asyncio.sleep(CONTROLLER_STOP_TIMEOUT + 0.1)

        self.assertTrue(timeout_fired, "user_turn_stop_timeout should fire as safety net")
        self.assertEqual(turn_stopped_count, 2, "timeout should end the stuck turn")


# ---------------------------------------------------------------------------
# Idle controller tests: user_idle_timeout blocked by stuck turn
# ---------------------------------------------------------------------------


class TestIdleControllerBlockedByStuckTurn(unittest.IsolatedAsyncioTestCase):
    """Tests for UserIdleController behavior during the interim deadlock.

    The user_idle_timeout timer only starts when BotStoppedSpeakingFrame arrives
    AND _user_turn_in_progress is False. When a ghost turn is stuck (never
    stopped), _user_turn_in_progress remains True and the idle timer can never
    start — removing another safety net.
    """

    async def asyncSetUp(self) -> None:
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_idle_timer_fires_when_no_turn_in_progress(self):
        """Baseline: idle timer fires normally when no turn is active."""
        idle_controller = UserIdleController(user_idle_timeout=IDLE_TIMEOUT)
        await idle_controller.setup(self.task_manager)

        idle_fired = False

        @idle_controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_fired
            idle_fired = True

        # Bot finishes speaking, no turn in progress → timer starts
        await idle_controller.process_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(IDLE_TIMEOUT + 0.1)
        self.assertTrue(idle_fired, "idle timer should fire when no turn is in progress")

    async def test_idle_timer_blocked_by_user_turn_in_progress(self):
        """Idle timer cannot start when _user_turn_in_progress is True.

        This reproduces the scenario where a ghost turn starts (from
        TranscriptionUserTurnStartStrategy) and never stops. The idle
        controller receives UserStartedSpeakingFrame but never
        UserStoppedSpeakingFrame, so _user_turn_in_progress stays True
        and the idle timer is permanently blocked.
        """
        idle_controller = UserIdleController(user_idle_timeout=IDLE_TIMEOUT)
        await idle_controller.setup(self.task_manager)

        idle_fired = False

        @idle_controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_fired
            idle_fired = True

        # Simulate: ghost turn starts but never stops
        await idle_controller.process_frame(UserStartedSpeakingFrame())
        # UserStoppedSpeakingFrame never arrives (_user_turn_in_progress stays True)

        # Bot stops speaking (was interrupted by the ghost turn)
        await idle_controller.process_frame(BotStoppedSpeakingFrame())

        # Wait well beyond the idle timeout
        await asyncio.sleep(IDLE_TIMEOUT + 0.1)

        # BUG: idle timer never started because _user_turn_in_progress is True.
        # This removes the last safety net — the bot stays silent indefinitely.
        self.assertFalse(idle_fired, "idle timer is blocked by stuck turn (documents the bug)")


if __name__ == "__main__":
    unittest.main()
