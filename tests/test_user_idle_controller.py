#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import unittest.mock

from pipecat.frames.frames import (
    BackgroundWorkFinishedFrame,
    BackgroundWorkStartedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    LLMMarkerResponseFrame,
    UserIdleTimeoutUpdateFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    UserTurnIdleFrame,
    UserWaitingFrame,
    UserWaitingTimeoutUpdateFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_idle_controller import UserIdleController
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.utils.asyncio.task_manager import TaskManager
from tests.frame_processor_helpers import frame_processor_setup

USER_IDLE_TIMEOUT = 0.2
USER_WAITING_TIMEOUT = 0.2


class TestUserIdleController(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.task_manager = TaskManager()

    async def test_idle_after_bot_stops_speaking(self):
        """Test that idle event fires after BotStoppedSpeakingFrame + timeout."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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

    async def test_intermediate_result_keeps_function_call_in_progress(self):
        """An async tool's intermediate result does not settle the call.

        An async function call (``cancel_on_interruption=False``) can report
        progress with ``is_final=False`` results while it keeps running. The
        bot may speak each update and stop again, but the user is still
        waiting on the tool, so the idle timer must stay off until the final
        result arrives.
        """
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Async tool starts; bot acknowledges it and stops speaking
        await controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())

        # Tool reports progress while still running; bot speaks the update
        await controller.process_frame(
            FunctionCallResultFrame(
                function_name="test",
                tool_call_id="123",
                arguments={},
                result="halfway",
                properties=FunctionCallResultProperties(is_final=False),
            )
        )
        self.assertTrue(controller.function_calls_in_progress)
        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())

        # Wait longer than timeout — should not fire (tool still running)
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertFalse(idle_triggered)

        # Final result settles the call; bot speaks it
        await controller.process_frame(
            FunctionCallResultFrame(
                function_name="test", tool_call_id="123", arguments={}, result="done"
            )
        )
        self.assertFalse(controller.function_calls_in_progress)
        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())

        # Now the timer should start and fire
        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)
        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_disabled_by_default(self):
        """Test that timeout=0 means idle detection is disabled."""
        controller = UserIdleController()
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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
        await controller.setup(frame_processor_setup(self.task_manager))

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

    async def test_update_applies_to_running_timer(self):
        """Test that a timeout update restarts a running timer with the new duration."""
        controller = UserIdleController(user_idle_timeout=10)
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Timer starts with the long timeout
        await controller.process_frame(BotStoppedSpeakingFrame())
        # Retune to a short timeout while the timer is running
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_enable_via_frame_arms_while_idle(self):
        """Test that enabling via frame arms the timer when everything is already idle."""
        controller = UserIdleController()
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # Bot finished speaking while idle detection was disabled
        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())

        # Enable idle detection — no further speaking events will re-arm
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_update_while_bot_speaking_does_not_arm(self):
        """Test that a timeout update while the bot is speaking does not arm the timer."""
        controller = UserIdleController()
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_update_before_first_bot_turn_does_not_arm(self):
        """Test that a timeout update before the bot has spoken does not arm the timer."""
        controller = UserIdleController()
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_update_while_user_turn_in_progress_does_not_arm(self):
        """Test that a timeout update during a user turn does not arm the timer."""
        controller = UserIdleController()
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(UserStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_update_while_awaiting_bot_response_does_not_arm(self):
        """Test that a timeout update between a user turn and the bot's response does not arm."""
        controller = UserIdleController()
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # User finished a turn; the bot's response is still in flight.
        await controller.process_frame(UserStartedSpeakingFrame())
        await controller.process_frame(UserStoppedSpeakingFrame())
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_update_while_function_call_in_progress_does_not_arm(self):
        """Test that a timeout update during a function call does not arm the timer."""
        controller = UserIdleController()
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())
        await controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_wait_for_user_arms_timer(self):
        """Test that wait_for_user arms the timer after a user turn."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        # The user speaks while the bot is idle: the timer is cancelled and
        # nothing else would restart it.
        await controller.process_frame(BotStoppedSpeakingFrame())
        await controller.process_frame(UserStartedSpeakingFrame())
        await controller.process_frame(UserStoppedSpeakingFrame())
        await controller.wait_for_user()

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertTrue(idle_triggered)

        await controller.cleanup()

    async def test_wait_for_user_while_bot_speaking_does_not_arm(self):
        """Test that wait_for_user does nothing while the bot is speaking."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.wait_for_user()

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_wait_for_user_during_function_call_does_not_arm(self):
        """Test that wait_for_user does nothing while a function call is pending."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))

        idle_triggered = False

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            nonlocal idle_triggered
            idle_triggered = True

        await controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await controller.wait_for_user()

        await asyncio.sleep(USER_IDLE_TIMEOUT + 0.1)

        self.assertFalse(idle_triggered)

        await controller.cleanup()

    async def test_waiting_for_user(self):
        """Test that waiting_for_user follows the bot finishing and the user speaking."""
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))

        self.assertFalse(controller.waiting_for_user)
        await controller.process_frame(BotStoppedSpeakingFrame())
        self.assertTrue(controller.waiting_for_user)
        await controller.process_frame(UserStartedSpeakingFrame())
        self.assertFalse(controller.waiting_for_user)
        await controller.process_frame(UserStoppedSpeakingFrame())
        self.assertFalse(controller.waiting_for_user)

        await controller.cleanup()


if __name__ == "__main__":
    unittest.main()


class TestUserWaiting(unittest.IsolatedAsyncioTestCase):
    """The waiting timer: the same silence as idle, with background work in progress."""

    async def asyncSetUp(self):
        self.task_manager = TaskManager()
        self.controller = UserIdleController(
            user_idle_timeout=USER_IDLE_TIMEOUT, user_waiting_timeout=USER_WAITING_TIMEOUT
        )
        await self.controller.setup(frame_processor_setup(self.task_manager))
        self.events: list[str] = []

        @self.controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            self.events.append("idle")

        @self.controller.event_handler("on_user_waiting")
        async def on_user_waiting(controller):
            self.events.append("waiting")

    async def asyncTearDown(self):
        await self.controller.cleanup()

    async def _let_timers_run(self):
        await asyncio.sleep(USER_WAITING_TIMEOUT + 0.1)

    async def test_bot_stopping_with_work_in_flight_starts_the_waiting_timer(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()

        self.assertEqual(self.events, ["waiting"])

    async def test_bot_stopping_with_nothing_in_flight_starts_the_idle_timer(self):
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()

        self.assertEqual(self.events, ["idle"])

    async def test_an_open_async_call_counts_as_work_in_flight(self):
        """The call's work frames bracket it; its pending result alone suppresses idle."""
        await self.controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="call_1"))
        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()

        self.assertEqual(self.events, ["waiting"])
        self.assertFalse(self.controller.waiting_for_user)

    async def test_a_pending_synchronous_call_starts_no_timer(self):
        await self.controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()

        self.assertEqual(self.events, [])

    async def test_work_starting_in_a_quiet_conversation_starts_the_waiting_timer(self):
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT * 0.3)
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self._let_timers_run()

        self.assertEqual(self.events, ["waiting"])

    async def test_work_starting_while_the_bot_speaks_waits_for_it_to_stop(self):
        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self._let_timers_run()
        self.assertEqual(self.events, [])

        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()
        self.assertEqual(self.events, ["waiting"])

    async def test_the_last_work_finishing_hands_over_to_the_idle_timer(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w2"))
        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_WAITING_TIMEOUT * 0.3)
        await self.controller.process_frame(BackgroundWorkFinishedFrame(work_id="w2"))
        await asyncio.sleep(USER_WAITING_TIMEOUT * 0.3)
        self.assertEqual(self.events, [])

        await self.controller.process_frame(BackgroundWorkFinishedFrame(work_id="w1"))
        await self._let_timers_run()

        self.assertEqual(self.events, ["idle"])
        self.assertTrue(self.controller.waiting_for_user)

    async def test_a_finish_for_unknown_work_is_ignored(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BackgroundWorkFinishedFrame(work_id="other"))
        await self.controller.process_frame(BackgroundWorkFinishedFrame(work_id="w1"))
        await self.controller.process_frame(BackgroundWorkFinishedFrame(work_id="w1"))
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()

        self.assertEqual(self.events, ["idle"])

    async def test_the_waiting_event_fires_once_per_start(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()
        await self._let_timers_run()
        self.assertEqual(self.events, ["waiting"])

        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()
        self.assertEqual(self.events, ["waiting", "waiting"])

    async def test_a_silent_reply_starts_the_waiting_timer_over(self):
        """A reply the LLM left unspoken ends its turn like speech would."""
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()
        self.assertEqual(self.events, ["waiting"])

        await self.controller.process_frame(LLMMarkerResponseFrame(raw="∅", kind="silent"))
        await self._let_timers_run()
        self.assertEqual(self.events, ["waiting", "waiting"])

    async def test_a_silent_reply_starts_the_idle_timer_over(self):
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()
        self.assertEqual(self.events, ["idle"])

        await self.controller.process_frame(LLMMarkerResponseFrame(raw="∅", kind="silent"))
        await self._let_timers_run()
        self.assertEqual(self.events, ["idle", "idle"])
        self.assertTrue(self.controller.waiting_for_user)

    async def test_a_silent_reply_starts_no_timer_while_someone_speaks(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(LLMMarkerResponseFrame(raw="∅", kind="silent"))
        await self._let_timers_run()
        self.assertEqual(self.events, [])

        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self.controller.process_frame(UserStartedSpeakingFrame())
        await self.controller.process_frame(LLMMarkerResponseFrame(raw="∅", kind="silent"))
        await self._let_timers_run()
        self.assertEqual(self.events, [])

    async def test_other_marker_reports_start_no_timer(self):
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()
        await self.controller.process_frame(LLMMarkerResponseFrame(raw="● Hi", kind="complete"))
        await self._let_timers_run()
        self.assertEqual(self.events, ["idle"])

    async def test_a_waiting_timeout_update_applies_at_once(self):
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))
        events = []

        @controller.event_handler("on_user_waiting")
        async def on_user_waiting(controller):
            events.append("waiting")

        await controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await controller.process_frame(BotStoppedSpeakingFrame())
        await controller.process_frame(UserWaitingTimeoutUpdateFrame(timeout=USER_WAITING_TIMEOUT))
        await self._let_timers_run()
        self.assertEqual(events, ["waiting"])

        await controller.process_frame(BotStartedSpeakingFrame())
        await controller.process_frame(BotStoppedSpeakingFrame())
        await controller.process_frame(UserWaitingTimeoutUpdateFrame(timeout=0))
        await self._let_timers_run()
        self.assertEqual(events, ["waiting"])
        await controller.cleanup()

    async def test_quiet_secs_counts_from_when_the_conversation_went_quiet(self):
        self.assertEqual(self.controller.quiet_secs, 0.0)
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(0.1)
        self.assertGreaterEqual(self.controller.quiet_secs, 0.1)

        await self.controller.process_frame(UserStartedSpeakingFrame())
        self.assertEqual(self.controller.quiet_secs, 0.0)
        await self.controller.process_frame(UserStoppedSpeakingFrame())
        await asyncio.sleep(0.05)
        self.assertGreaterEqual(self.controller.quiet_secs, 0.05)
        self.assertLess(self.controller.quiet_secs, 0.1)

    async def test_quiet_goes_on_through_a_silent_reply(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()
        await self.controller.process_frame(LLMMarkerResponseFrame(raw="∅", kind="silent"))
        self.assertGreaterEqual(self.controller.quiet_secs, USER_WAITING_TIMEOUT)

    async def test_a_call_settling_with_work_in_flight_starts_the_waiting_timer_over(self):
        """The bot speaks, then calls a function, and says nothing on its result."""
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self.controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await self.controller.process_frame(
            FunctionCallResultFrame(
                function_name="test", tool_call_id="123", arguments={}, result="ok"
            )
        )
        await self._let_timers_run()

        self.assertEqual(self.events, ["waiting"])

    async def test_a_call_settling_while_the_bot_speaks_starts_no_timer(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(
            FunctionCallsStartedFrame(function_calls=[unittest.mock.Mock()])
        )
        await self.controller.process_frame(BotStartedSpeakingFrame())
        await self.controller.process_frame(
            FunctionCallResultFrame(
                function_name="test", tool_call_id="123", arguments={}, result="ok"
            )
        )
        await self._let_timers_run()

        self.assertEqual(self.events, [])

    async def test_speech_cancels_the_waiting_timer(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_WAITING_TIMEOUT * 0.3)
        await self.controller.process_frame(UserStartedSpeakingFrame())
        await self._let_timers_run()

        self.assertEqual(self.events, [])

    async def test_wait_for_user_with_work_in_flight_starts_the_waiting_timer(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self.controller.process_frame(UserStartedSpeakingFrame())
        await self.controller.process_frame(UserStoppedSpeakingFrame())
        await self.controller.wait_for_user()
        await self._let_timers_run()

        self.assertEqual(self.events, ["waiting"])

    async def test_an_idle_timeout_update_does_not_arm_during_work(self):
        await self.controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self.controller.process_frame(BotStoppedSpeakingFrame())
        await self.controller.process_frame(UserIdleTimeoutUpdateFrame(timeout=USER_IDLE_TIMEOUT))
        await self._let_timers_run()

        self.assertEqual(self.events, ["waiting"])

    async def test_work_starting_cancels_an_idle_timer_even_with_no_waiting_timeout(self):
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))
        events = []

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            events.append("idle")

        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_IDLE_TIMEOUT * 0.3)
        await controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await self._let_timers_run()

        self.assertEqual(events, [])
        await controller.cleanup()

    async def test_work_finishing_cancels_a_waiting_timer_even_with_no_idle_timeout(self):
        controller = UserIdleController(user_waiting_timeout=USER_WAITING_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))
        events = []

        @controller.event_handler("on_user_waiting")
        async def on_user_waiting(controller):
            events.append("waiting")

        await controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await controller.process_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(USER_WAITING_TIMEOUT * 0.3)
        await controller.process_frame(BackgroundWorkFinishedFrame(work_id="w1"))
        await self._let_timers_run()

        self.assertEqual(events, [])
        await controller.cleanup()

    async def test_waiting_detection_is_off_by_default(self):
        controller = UserIdleController(user_idle_timeout=USER_IDLE_TIMEOUT)
        await controller.setup(frame_processor_setup(self.task_manager))
        events = []

        @controller.event_handler("on_user_waiting")
        async def on_user_waiting(controller):
            events.append("waiting")

        @controller.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(controller):
            events.append("idle")

        await controller.process_frame(BackgroundWorkStartedFrame(work_id="w1"))
        await controller.process_frame(BotStoppedSpeakingFrame())
        await self._let_timers_run()

        self.assertEqual(events, [])
        await controller.cleanup()


class TestUserWaitingHosts(unittest.IsolatedAsyncioTestCase):
    """The hosts expose the waiting timeout and event beside the idle pair, and broadcast both silences."""

    def _waiting_frames(self, frames: list) -> list[UserWaitingFrame]:
        return [f for f in frames if isinstance(f, UserWaitingFrame)]

    def _idle_frames(self, frames: list) -> list[UserTurnIdleFrame]:
        return [f for f in frames if isinstance(f, UserTurnIdleFrame)]

    async def test_the_user_aggregator_emits_on_user_waiting(self):
        user, assistant = LLMContextAggregatorPair(
            LLMContext(),
            user_params=LLMUserAggregatorParams(user_waiting_timeout=USER_WAITING_TIMEOUT),
        )
        events = []

        @user.event_handler("on_user_waiting")
        async def on_user_waiting(aggregator):
            events.append("waiting")

        received_down, received_up = await run_test(
            Pipeline([user, assistant]),
            frames_to_send=[
                BackgroundWorkStartedFrame(work_id="w1"),
                BotStoppedSpeakingFrame(),
                SleepFrame(USER_WAITING_TIMEOUT + 0.2),
            ],
        )

        self.assertEqual(events, ["waiting"])
        # The frame goes both ways, saying how long it has been quiet.
        (down,) = self._waiting_frames(received_down)
        (up,) = self._waiting_frames(received_up)
        self.assertGreaterEqual(down.quiet_secs, USER_WAITING_TIMEOUT)
        self.assertEqual(up.quiet_secs, down.quiet_secs)

    async def test_the_user_aggregator_broadcasts_the_idle_frame(self):
        user, assistant = LLMContextAggregatorPair(
            LLMContext(),
            user_params=LLMUserAggregatorParams(user_idle_timeout=USER_IDLE_TIMEOUT),
        )

        received_down, received_up = await run_test(
            Pipeline([user, assistant]),
            frames_to_send=[BotStoppedSpeakingFrame(), SleepFrame(USER_IDLE_TIMEOUT + 0.2)],
        )

        (down,) = self._idle_frames(received_down)
        (up,) = self._idle_frames(received_up)
        self.assertGreaterEqual(down.quiet_secs, USER_IDLE_TIMEOUT)
        self.assertEqual(up.quiet_secs, down.quiet_secs)

    async def test_the_user_turn_processor_emits_on_user_waiting(self):
        processor = UserTurnProcessor(user_waiting_timeout=USER_WAITING_TIMEOUT)
        events = []

        @processor.event_handler("on_user_waiting")
        async def on_user_waiting(processor):
            events.append("waiting")

        received_down, received_up = await run_test(
            processor,
            frames_to_send=[
                BackgroundWorkStartedFrame(work_id="w1"),
                BotStoppedSpeakingFrame(),
                SleepFrame(USER_WAITING_TIMEOUT + 0.2),
            ],
        )

        self.assertEqual(events, ["waiting"])
        self.assertEqual(len(self._waiting_frames(received_down)), 1)
        self.assertEqual(len(self._waiting_frames(received_up)), 1)

    async def test_the_user_turn_processor_broadcasts_the_idle_frame(self):
        processor = UserTurnProcessor(user_idle_timeout=USER_IDLE_TIMEOUT)

        received_down, received_up = await run_test(
            processor,
            frames_to_send=[BotStoppedSpeakingFrame(), SleepFrame(USER_IDLE_TIMEOUT + 0.2)],
        )

        self.assertEqual(len(self._idle_frames(received_down)), 1)
        self.assertEqual(len(self._idle_frames(received_up)), 1)
