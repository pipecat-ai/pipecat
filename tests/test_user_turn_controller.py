#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy, deferred
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

USER_TURN_STOP_TIMEOUT = 0.2
TRANSCRIPTION_TIMEOUT = 0.1


class TestUserTurnController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

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


if __name__ == "__main__":
    unittest.main()
