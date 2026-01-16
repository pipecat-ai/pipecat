#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    BotSpeakingFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
)
from pipecat.turns.user_idle_controller import UserIdleController
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

USER_IDLE_TIMEOUT = 0.2


class TestUserIdleController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    async def test_basic_idle_detection(self):
        """Test that idle event is triggered after timeout when no activity."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_idle")
        async def on_user_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Start conversation
        await controller.process_frame(UserStartedSpeakingFrame())

        # Wait for idle timeout
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_user_speaking_resets_idle_timer(self):
        """Test that continuous UserSpeakingFrame frames reset the idle timer."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_idle")
        async def on_user_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Start conversation
        await controller.process_frame(UserStartedSpeakingFrame())

        # Send UserSpeakingFrame continuously to reset timer
        for _ in range(5):
            await asyncio.sleep(USER_IDLE_TIMEOUT * 0.5)  # 50% of timeout period
            await controller.process_frame(UserSpeakingFrame())

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_bot_speaking_resets_idle_timer(self):
        """Test that BotSpeakingFrame frames reset the idle timer."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_idle")
        async def on_user_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Start conversation
        await controller.process_frame(UserStartedSpeakingFrame())

        # Bot speaking should reset timer
        for _ in range(5):
            await asyncio.sleep(USER_IDLE_TIMEOUT * 0.6)  # 60% of timeout
            await controller.process_frame(BotSpeakingFrame())

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_function_call_prevents_idle(self):
        """Test that function calls in progress prevent idle event."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_idle")
        async def on_user_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Start conversation
        await controller.process_frame(UserStartedSpeakingFrame())

        # Start function call
        await controller.process_frame(FunctionCallsStartedFrame(function_calls=[]))

        # Wait longer than idle timeout
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        # Should not trigger idle because function call is in progress
        self.assertFalse(idle_triggered)

        # Complete function call
        await controller.process_frame(
            FunctionCallResultFrame(
                function_name="test",
                tool_call_id="123",
                arguments={},
                result=None,
                run_llm=False,
            )
        )

        # Now idle should trigger
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_no_idle_before_conversation_starts(self):
        """Test that idle monitoring doesn't start before first conversation activity."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_idle")
        async def on_user_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Wait without starting conversation
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_idle_starts_with_bot_speaking(self):
        """Test that monitoring starts with BotSpeakingFrame, not just user speech."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_triggered = False

        @controller.event_handler("on_user_idle")
        async def on_user_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Start conversation with bot speaking
        await controller.process_frame(BotSpeakingFrame())

        # Wait for idle timeout
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_multiple_idle_events(self):
        """Test that idle event can trigger multiple times."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(self.task_manager)

        idle_count = 0

        @controller.event_handler("on_user_idle")
        async def on_user_idle(controller):
            nonlocal idle_count
            idle_count += 1

        # Start conversation
        await controller.process_frame(UserStartedSpeakingFrame())

        # First idle
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        first_count = idle_count
        self.assertGreaterEqual(first_count, 1)

        # Second idle
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        second_count = idle_count
        self.assertGreater(second_count, first_count)

        # User activity resets timer
        await controller.process_frame(UserSpeakingFrame())

        # Give a moment for the timer to reset
        await asyncio.sleep(0.1)

        # Third idle
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        third_count = idle_count
        self.assertGreater(third_count, second_count)

        await controller.cleanup()
