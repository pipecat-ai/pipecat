#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import warnings
from unittest.mock import MagicMock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnCompletionStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
    deferred,
)
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

USER_TURN_STOP_TIMEOUT = 0.2
TRANSCRIPTION_TIMEOUT = 0.1


class TestUserTurnController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_completion_dropped_while_user_speaking(self):
        """A completion arriving while the user speaks must not stop the turn.

        External completions (e.g. an LLM ✓) resolve with latency, so the user
        may have resumed speaking. The controller drops the finalization while
        the user is speaking and finalizes once they fall silent.
        """
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[ExternalUserTurnCompletionStopStrategy()],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(self.task_manager)

        stopped = False

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stopped
            stopped = True

        # Start a turn; the user is now speaking.
        await controller.process_frame(VADUserStartedSpeakingFrame())

        # A completion resolving while the user still speaks is stale: dropped.
        await controller.process_frame(UserTurnInferenceCompletedFrame())
        self.assertFalse(stopped)

        # Once the user stops, a fresh completion finalizes the turn.
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await controller.process_frame(UserTurnInferenceCompletedFrame())
        self.assertTrue(stopped)

        await controller.cleanup()

    async def test_default_user_turn_strategies(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )

        await controller.setup(self.task_manager)

        should_start = None
        should_stop = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)
        # Wait for user_speech_timeout to elapse
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertTrue(should_stop)

    async def test_update_strategies_does_not_accumulate_event_handlers(self):
        # Regression: re-applying strategies (e.g. realtime mode re-installing the
        # same instances, or repeated metadata broadcasts) must not accumulate
        # duplicate controller event handlers on the strategy instances —
        # _cleanup_strategies removes what _setup_strategies added.
        start = VADUserTurnStartStrategy()
        stop = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=[start], stop=[stop])
        )
        await controller.setup(self.task_manager)

        # Re-apply the same strategy instances several times.
        for _ in range(3):
            await controller.update_strategies(UserTurnStrategies(start=[start], stop=[stop]))

        for strategy in (start, stop):
            for name, handler in strategy._event_handlers.items():
                self.assertLessEqual(
                    len(handler.handlers), 1, f"{name} double-registered on {strategy}"
                )

    async def test_inference_triggered_fires_alongside_stopped(self):
        """Default strategies fire both inference-triggered and stopped, in order."""
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            )
        )

        await controller.setup(self.task_manager)

        events: list[str] = []

        @controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy):
            events.append("inference_triggered")

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            events.append("stopped")

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)

        self.assertEqual(events, ["inference_triggered", "stopped"])

    async def test_deferred_wrapper_skips_stopped(self):
        """A deferred() wrapper drops the inner strategy's on_user_turn_stopped event."""
        wrapped = deferred(
            SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        )
        controller = UserTurnController(user_turn_strategies=UserTurnStrategies(stop=[wrapped]))

        await controller.setup(self.task_manager)

        events: list[str] = []

        @controller.event_handler("on_user_turn_inference_triggered")
        async def on_user_turn_inference_triggered(controller, strategy):
            events.append("inference_triggered")

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            events.append("stopped")

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)

        # The inner strategy fires inference-triggered (forwarded by the
        # wrapper). Finalization is suppressed, but the controller's
        # stop watchdog eventually fires `stopped`.
        self.assertEqual(events[0], "inference_triggered")

    async def test_user_turn_start_reset(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)]
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        should_start = 0

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start += 1

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame(text="One", user_id="cat", timestamp=""))
        self.assertEqual(should_start, 0)

        await controller.process_frame(
            TranscriptionFrame(text="One two three!", user_id="cat", timestamp="")
        )
        self.assertEqual(should_start, 1)

        # Trigger user stop turn so we can trigger user start turn again.
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(TranscriptionFrame(text="Hi!", user_id="cat", timestamp=""))
        self.assertEqual(should_start, 1)

        await controller.process_frame(
            TranscriptionFrame(text="How are you?", user_id="cat", timestamp="")
        )
        self.assertEqual(should_start, 2)

    async def test_user_turn_stop_timeout_no_transcription(self):
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        should_start = None
        should_stop = None
        timeout = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            nonlocal timeout
            timeout = True

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)

        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertTrue(timeout)

    async def test_external_user_turn_strategies_no_timeout_while_speaking(self):
        """Test that timeout does not trigger when user is still speaking with external strategies."""
        controller = UserTurnController(
            user_turn_strategies=ExternalUserTurnStrategies(),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        should_start = None
        should_stop = None
        timeout = None

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal should_start
            should_start = True

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal should_stop
            should_stop = True

        @controller.event_handler("on_user_turn_stop_timeout")
        async def on_user_turn_stop_timeout(controller):
            nonlocal timeout
            timeout = True

        # Simulate external service (like Deepgram Flux) broadcasting UserStartedSpeakingFrame
        await controller.process_frame(UserStartedSpeakingFrame())
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        # User is still speaking, timeout should not trigger
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertTrue(should_start)
        self.assertFalse(should_stop)
        self.assertFalse(timeout)

        # Now external service broadcasts UserStoppedSpeakingFrame
        await controller.process_frame(UserStoppedSpeakingFrame())

        # But no transcription, so timeout should trigger
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)

        self.assertTrue(should_start)
        self.assertTrue(should_stop)
        self.assertTrue(timeout)

    async def test_late_transcription_between_turns_no_premature_stop(self):
        """Test that a late transcription arriving between turns does not cause a premature stop.

        Reproduces the bug from issue #4053: after turn 1 completes and reset()
        clears state, a late TranscriptionFrame sets _text to stale content. On
        the next turn, that stale _text gates a premature turn stop via timeout(0)
        before the current turn's transcript arrives.

        Uses only VADUserTurnStartStrategy (no TranscriptionUserTurnStartStrategy)
        so the late transcription doesn't trigger a spurious turn start.
        """
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )

        await controller.setup(self.task_manager)

        start_count = 0
        stop_count = 0

        @controller.event_handler("on_user_turn_started")
        async def on_user_turn_started(controller, strategy, params):
            nonlocal start_count
            start_count += 1

        @controller.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(controller, strategy, params):
            nonlocal stop_count
            stop_count += 1

        # === Turn 1: S-T-E ===
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(start_count, 1)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )

        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 1)

        # === Between turns: late transcription arrives ===
        # This sets _text on the stop strategy while _user_turn is False.
        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )

        # === Turn 2: S-T-E (transcription arrives during turn) ===
        # The fix resets stop strategies at turn start, clearing stale _text.
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(start_count, 2)

        await controller.process_frame(
            TranscriptionFrame(text="How are you?", user_id="", timestamp="now")
        )

        await controller.process_frame(VADUserStoppedSpeakingFrame())

        # Wait for user_speech_timeout to elapse — should get turn 2 stop
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(stop_count, 2)

    async def test_turn_start_notifies_start_and_stop_strategies(self):
        """Turn start calls handle_user_turn_started on both start and stop strategies."""
        started = []

        class SpyStart(VADUserTurnStartStrategy):
            async def handle_user_turn_started(self):
                started.append("start")

        class SpyStop(SpeechTimeoutUserTurnStopStrategy):
            async def handle_user_turn_started(self):
                await super().handle_user_turn_started()
                started.append("stop")

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[SpyStart()],
                stop=[SpyStop(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(self.task_manager)

        self.assertEqual(started, [])
        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(started, ["start", "stop"])

        await controller.cleanup()

    async def test_handle_user_turn_stopped_called_on_strategy_stop(self):
        """handle_user_turn_stopped() runs on a stop strategy when it ends the turn, never at start."""
        finalized = 0

        class SpyStopStrategy(SpeechTimeoutUserTurnStopStrategy):
            async def handle_user_turn_stopped(self):
                nonlocal finalized
                await super().handle_user_turn_stopped()
                finalized += 1

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpyStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(self.task_manager)

        await controller.process_frame(VADUserStartedSpeakingFrame())
        self.assertEqual(finalized, 0)

        await controller.process_frame(
            TranscriptionFrame(text="Hello!", user_id="", timestamp="now")
        )
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TRANSCRIPTION_TIMEOUT + 0.1)
        self.assertEqual(finalized, 1)

        await controller.cleanup()

    async def test_handle_user_turn_stopped_called_on_watchdog_stop(self):
        """handle_user_turn_stopped() also runs when the stop watchdog ends the turn."""
        finalized = 0

        class SpyNeverStopStrategy(ExternalUserTurnCompletionStopStrategy):
            async def handle_user_turn_stopped(self):
                nonlocal finalized
                await super().handle_user_turn_stopped()
                finalized += 1

        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpyNeverStopStrategy()],
            ),
            user_turn_stop_timeout=USER_TURN_STOP_TIMEOUT,
        )
        await controller.setup(self.task_manager)

        await controller.process_frame(VADUserStartedSpeakingFrame())
        await controller.process_frame(VADUserStoppedSpeakingFrame())
        self.assertEqual(finalized, 0)

        # No completion ever arrives; the watchdog finalizes the turn.
        await asyncio.sleep(USER_TURN_STOP_TIMEOUT + 0.1)
        self.assertEqual(finalized, 1)

        await controller.cleanup()

    async def test_start_strategy_not_reset_on_turn_stop(self):
        """A start strategy's handle_user_turn_stopped is a no-op (the deliberate asymmetry).

        Start strategies are armed on turn start but never reset on turn stop:
        their reset is turn-start semantic (e.g. WakePhrase refreshes its
        keepalive timeout), so a stop-side reset would be wrong.
        """
        resets = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            class LegacyStart(VADUserTurnStartStrategy):
                async def reset(self):  # the deprecated hook
                    nonlocal resets
                    resets += 1

        strategy = LegacyStart()

        # Arming on turn start bridges to reset()...
        await strategy.handle_user_turn_started()
        self.assertEqual(resets, 1)

        # ...but turn stop does not touch it.
        await strategy.handle_user_turn_stopped()
        self.assertEqual(resets, 1)

    async def test_stop_strategy_reset_bridged_on_both_callbacks(self):
        """A stop strategy's reset() is bridged on both turn boundaries (armed, then cleaned)."""
        resets = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            class LegacyStop(ExternalUserTurnCompletionStopStrategy):
                async def reset(self):  # the deprecated hook
                    nonlocal resets
                    resets += 1

        strategy = LegacyStop()
        await strategy.handle_user_turn_started()
        await strategy.handle_user_turn_stopped()
        self.assertEqual(resets, 2)

    async def test_overriding_reset_warns_at_class_definition(self):
        """Overriding the deprecated reset() warns when the class is defined."""
        with self.assertWarns(DeprecationWarning):

            class LegacyStop(ExternalUserTurnCompletionStopStrategy):
                async def reset(self):
                    pass

    async def test_subclass_of_concrete_strategy_overriding_reset_warns_but_is_not_bridged(self):
        """Subclassing a concrete strategy and overriding reset() warns, but reset() won't run.

        A concrete strategy's callback overrides don't route through the
        backward-compat bridge, so a legacy reset() on such a subclass is never
        invoked. __init_subclass__ still flags it loudly so the author migrates
        to the callbacks — no silent no-op.
        """
        reset_calls = 0

        with self.assertWarns(DeprecationWarning):

            class MyMinWords(MinWordsUserTurnStartStrategy):
                async def reset(self):
                    nonlocal reset_calls
                    reset_calls += 1

        strategy = MyMinWords(min_words=3)
        await strategy.handle_user_turn_started()
        self.assertEqual(reset_calls, 0)

    async def test_overriding_callbacks_does_not_warn_at_class_definition(self):
        """Defining a strategy that overrides the callbacks (not reset) doesn't warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            class Migrated(ExternalUserTurnCompletionStopStrategy):
                async def handle_user_turn_stopped(self):
                    pass

    async def test_migrated_strategy_does_not_warn_at_runtime(self):
        """A strategy overriding the callbacks never warns on the per-turn callbacks."""
        strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TRANSCRIPTION_TIMEOUT)
        await strategy.setup(self.task_manager)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            await strategy.handle_user_turn_started()
            await strategy.handle_user_turn_stopped()
        await strategy.cleanup()

    async def test_turn_analyzer_cleared_on_stop_but_not_start(self):
        """The analyzer is cleared when the turn ends, but not when it starts.

        Clearing on start would drop the continuously-fed pre-speech buffer;
        that's the whole reason the clear can't live in the shared reset.
        """
        analyzer = MagicMock()
        strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)

        await strategy.handle_user_turn_started()
        analyzer.clear.assert_not_called()

        await strategy.handle_user_turn_stopped()
        analyzer.clear.assert_called_once()

    async def test_deferred_forwards_both_callbacks(self):
        """The deferred() wrapper forwards both callbacks to its inner strategy."""
        analyzer = MagicMock()
        strategy = deferred(TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer))

        await strategy.handle_user_turn_started()
        analyzer.clear.assert_not_called()

        await strategy.handle_user_turn_stopped()
        analyzer.clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
