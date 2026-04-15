#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import unittest.mock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    UserIdleTimeoutUpdateFrame,
    UserStartedSpeakingFrame,
)
from pipecat.turns.user_idle_controller import UserIdleController
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

USER_IDLE_TIMEOUT = 0.2


class TestUserIdleController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_idle_after_bot_stops_speaking(self):
        """Test that idle event fires after BotStoppedSpeakingFrame + timeout."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStoppedSpeakingFrame())

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_user_speaking_cancels_timer(self):
        """Test that UserStartedSpeakingFrame cancels the idle timer."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT * 0.3)
        await controller.process_frame(UserStartedSpeakingFrame())

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_bot_speaking_cancels_timer(self):
        """Test that BotStartedSpeakingFrame cancels the idle timer."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT * 0.3)
        await controller.process_frame(BotStartedSpeakingFrame())

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_no_idle_before_bot_speaks(self):
        """Test that idle does not fire if no BotStoppedSpeakingFrame is received."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Wait without any frames
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_interruption_no_false_trigger(self):
        """Test that BotStoppedSpeakingFrame during a user turn does not start the timer."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # User starts speaking (interruption)
        await controller.process_frame(UserStartedSpeakingFrame())
        # Bot stops speaking due to interruption
        await controller.process_frame(BotStoppedSpeakingFrame())

        # Wait - timer should NOT have started because user turn is in progress
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_idle_cycle(self):
        """Test that idle fires, then can fire again after another bot speaking cycle."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_count = 0

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_count
            idle_count += 1

        # First cycle: bot stops → idle fires
        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertEqual(idle_count, 1)

        # Second cycle: bot starts → bot stops → idle fires again
        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertEqual(idle_count, 2)

        await controller.cleanup()

    async def test_cleanup_cancels_timer(self):
        """Test that cleanup cancels a pending idle timer."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT * 0.3)
        await controller.cleanup()

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

    async def test_function_call_cancels_timer(self):
        """Test normal ordering: BotStopped starts timer, FunctionCallsStarted cancels it."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Bot finishes speaking, timer starts
        await controller.process_frame(BotStoppedSpeakingFrame())
        # Function call starts shortly after, cancels the timer
        await asyncio.sleep(USER_IDLE_TIMEOUT * 0.3)
        await controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )

        # Wait longer than timeout — should not fire
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_function_call_suppresses_timer(self):
        """Test race condition: FunctionCallsStarted arrives before BotStopped.

        A race condition can cause FunctionCallsStarted to arrive before
        BotStoppedSpeaking. The counter guard prevents the timer from starting
        while a function call is in progress.
        """
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # LLM emits function call and "let me check" concurrently
        await controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())

        # Wait longer than timeout — should not fire (function call in progress)
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertFalse(idle_triggered)

        # Function call completes, bot speaks result
        await controller.process_frame(
            FunctionCallResultFrame(
                function_name="test", tool_call_id="123", arguments={}, result="ok"
            )
        )
        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())

        # Now the timer should start and fire
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_disabled_by_default(self):
        """Test that timeout=0 means idle detection is disabled."""
        controller = UserIdleController()
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_enable_via_frame(self):
        """Test enabling idle detection at runtime via UserIdleTimeoutUpdateFrame."""
        controller = UserIdleController()
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Initially disabled — no idle fires
        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertFalse(idle_triggered)

        # Enable idle detection
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))
        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_disable_via_frame(self):
        """Test disabling idle detection at runtime via UserIdleTimeoutUpdateFrame."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Start the timer
        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT * 0.3)

        # Disable — should cancel running timer
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=0))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()


if __name__ == "__main__":
    unittest.main()
