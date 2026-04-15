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
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
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


if __name__ == "__main__":
    unittest.main()
