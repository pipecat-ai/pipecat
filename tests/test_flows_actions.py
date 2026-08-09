#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test suite for ActionManager functionality.

This module tests the ActionManager class which handles execution of actions
during conversation flows. Tests cover:
- Built-in actions (TTS, end conversation)
- Custom action registration and execution
- Error handling and validation
- Action sequencing
- TTS service integration
- Frame queueing

The tests use unittest.IsolatedAsyncioTestCase for async support and include
mocked dependencies for PipelineTask.
"""

import asyncio
import unittest
import warnings
from typing import Any
from unittest.mock import AsyncMock, patch

from pipecat.flows.actions import ActionFinishedFrame, ActionManager, FunctionActionFrame
from pipecat.flows.exceptions import ActionError
from pipecat.frames.frames import TTSSpeakFrame, UninterruptibleFrame
from pipecat.utils.frame_queue import FrameQueue
from tests.flows_test_helpers import (
    assert_end_frame_queued,
    assert_tts_speak_frames_queued,
    get_queued_tts_speak_frames,
    make_mock_task,
)


class TestActionManager(unittest.IsolatedAsyncioTestCase):
    """Test suite for ActionManager class.

    Tests functionality of ActionManager including:
    - Built-in action handlers:
        - TTS speech synthesis
        - Conversation ending
    - Custom action registration
    - Action execution sequencing
    - Error handling:
        - Missing TTS service
        - Invalid actions
        - Failed handlers
    - Multiple action execution
    - Frame queueing validation

    Each test uses mocked dependencies to verify:
    - Correct frame generation
    - Proper service calls
    - Error handling behavior
    - Action sequencing
    """

    def setUp(self):
        """Set up test fixtures before each test.

        Creates:
        - Mock PipelineTask for frame queueing
        - ActionManager instance with mocked dependencies
        """
        self.mock_task = make_mock_task()
        self.mock_flow_manager = AsyncMock()
        self.action_manager = ActionManager(self.mock_task, self.mock_flow_manager)

    async def test_initialization(self):
        """Test ActionManager initialization and default handlers."""
        # Verify built-in action handlers are registered
        self.assertIn("tts_say", self.action_manager._action_handlers)
        self.assertIn("end_conversation", self.action_manager._action_handlers)

    async def test_tts_action(self):
        """Test basic TTS action execution."""
        action = {"type": "tts_say", "text": "Hello"}
        await self.action_manager.execute_actions([action])
        assert_tts_speak_frames_queued(self.mock_task, ["Hello"])

    async def test_end_conversation_action(self):
        """Test basic end conversation action."""
        action = {"type": "end_conversation"}
        await self.action_manager.execute_actions([action])

        # Verify EndFrame was queued
        assert_end_frame_queued(self.mock_task)

    async def test_end_conversation_with_goodbye(self):
        """Test end conversation action with goodbye message."""
        action = {"type": "end_conversation", "text": "Goodbye!"}
        await self.action_manager.execute_actions([action])

        # Verify TTSSpeakFrame
        assert_tts_speak_frames_queued(self.mock_task, ["Goodbye!"])

        # Verify EndFrame
        assert_end_frame_queued(self.mock_task)

    async def test_tts_action_append_text_to_context(self):
        """Test that tts_say maps append_text_to_context onto the TTSSpeakFrame."""
        # Explicitly True
        await self.action_manager.execute_actions(
            [{"type": "tts_say", "text": "Hello", "append_text_to_context": True}]
        )
        frames = get_queued_tts_speak_frames(self.mock_task)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0].append_to_context, True)

        # Explicitly False
        self.mock_task.queue_frame.reset_mock()
        await self.action_manager.execute_actions(
            [{"type": "tts_say", "text": "Hello", "append_text_to_context": False}]
        )
        frames = get_queued_tts_speak_frames(self.mock_task)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0].append_to_context, False)

        # Omitted: Flows applies its own default of True (and never passes None,
        # so no append_to_context deprecation warning fires).
        self.mock_task.queue_frame.reset_mock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await self.action_manager.execute_actions([{"type": "tts_say", "text": "Hello"}])
        frames = get_queued_tts_speak_frames(self.mock_task)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0].append_to_context, True)
        self.assertEqual(
            [w for w in caught if "append_to_context" in str(w.message)],
            [],
            "Flows must not pass None to TTSSpeakFrame",
        )

    async def test_end_conversation_append_text_to_context(self):
        """Test that end_conversation maps append_text_to_context onto its goodbye frame."""
        # Explicitly False
        await self.action_manager.execute_actions(
            [{"type": "end_conversation", "text": "Goodbye!", "append_text_to_context": False}]
        )
        frames = get_queued_tts_speak_frames(self.mock_task)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0].append_to_context, False)
        assert_end_frame_queued(self.mock_task)

        # Explicitly True
        self.mock_task.queue_frame.reset_mock()
        await self.action_manager.execute_actions(
            [{"type": "end_conversation", "text": "Goodbye!", "append_text_to_context": True}]
        )
        frames = get_queued_tts_speak_frames(self.mock_task)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0].append_to_context, True)

        # Omitted: Flows applies its own default of True (and never passes None,
        # so no append_to_context deprecation warning fires).
        self.mock_task.queue_frame.reset_mock()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await self.action_manager.execute_actions(
                [{"type": "end_conversation", "text": "Goodbye!"}]
            )
        frames = get_queued_tts_speak_frames(self.mock_task)
        self.assertEqual(len(frames), 1)
        self.assertIs(frames[0].append_to_context, True)
        self.assertEqual(
            [w for w in caught if "append_to_context" in str(w.message)],
            [],
            "Flows must not pass None to TTSSpeakFrame",
        )

    async def test_function_actions(self):
        """Test executing function actions."""
        results = []

        async def first_function(action, flow_manager):
            results.append("first_start")
            await asyncio.sleep(0.25)
            results.append("first_end")

        async def second_function(action, flow_manager):
            results.append("second_start")
            results.append("second_end")

        actions = [
            {"type": "function", "handler": first_function},
            {"type": "function", "handler": second_function},
        ]

        await self.action_manager.execute_actions(actions)

        # Validate the order
        self.assertEqual(
            results,
            ["first_start", "first_end", "second_start", "second_end"],
        )

    async def test_action_handler_signatures(self):
        """Test both legacy and modern action handler signatures."""

        # Test legacy single-parameter handler
        async def legacy_handler(action: dict):
            self.assertEqual(action["data"], "legacy")

        self.action_manager._register_action("legacy", legacy_handler)
        await self.action_manager.execute_actions([{"type": "legacy", "data": "legacy"}])

        # Test modern two-parameter handler
        async def modern_handler(action: dict, flow_manager: Any):
            self.assertEqual(action["data"], "modern")
            self.assertEqual(flow_manager, self.mock_flow_manager)

        self.action_manager._register_action("modern", modern_handler)
        await self.action_manager.execute_actions([{"type": "modern", "data": "modern"}])

    async def test_invalid_action(self):
        """Test handling invalid actions."""
        # Test missing type
        with self.assertRaises(ActionError) as context:
            await self.action_manager.execute_actions([{}])
        self.assertIn("missing required 'type' field", str(context.exception))

        # Test unknown action type
        with self.assertRaises(ActionError) as context:
            await self.action_manager.execute_actions([{"type": "invalid"}])
        self.assertIn("No handler registered", str(context.exception))

    async def test_multiple_actions(self):
        """Test executing multiple actions in sequence."""
        actions = [
            {"type": "tts_say", "text": "First"},
            {"type": "tts_say", "text": "Second"},
        ]
        await self.action_manager.execute_actions(actions)

        # Verify TTS was called twice in correct order
        assert_tts_speak_frames_queued(self.mock_task, ["First", "Second"])

    def test_register_invalid_handler(self):
        """Test registering invalid action handlers."""
        # Test non-callable handler
        with self.assertRaises(ValueError) as context:
            self.action_manager._register_action("invalid", "not_callable")
        self.assertIn("must be callable", str(context.exception))

        # Test None handler
        with self.assertRaises(ValueError) as context:
            self.action_manager._register_action("invalid", None)
        self.assertIn("must be callable", str(context.exception))

    async def test_none_or_empty_actions(self):
        """Test handling None or empty action lists."""
        # Test None actions
        await self.action_manager.execute_actions(None)
        self.mock_task.queue_frame.assert_not_called()

        # Test empty list
        await self.action_manager.execute_actions([])
        self.mock_task.queue_frame.assert_not_called()

    @patch("loguru.logger.error")
    async def test_action_error_handling(self, mock_logger):
        """Test error handling during action execution."""
        # Configure task mock to raise an error
        self.mock_task.queue_frame = AsyncMock(side_effect=Exception("Frame error"))

        action = {"type": "tts_say", "text": "Hello"}
        await self.action_manager.execute_actions([action])

        # Verify error was logged
        mock_logger.assert_called_with("TTS error: Frame error")

    async def test_action_execution_error_handling(self):
        """Test error handling during action execution."""
        action_manager = ActionManager(self.mock_task, self.mock_flow_manager)

        # Test action with missing handler
        with self.assertRaises(ActionError):
            await action_manager.execute_actions([{"type": "nonexistent_action"}])

        # Test action handler that raises an exception
        async def failing_handler(action):
            raise Exception("Handler error")

        action_manager._register_action("failing_action", failing_handler)

        with self.assertRaises(ActionError):
            await action_manager.execute_actions([{"type": "failing_action"}])

    def test_action_frames_are_uninterruptible(self):
        """Test that ActionFinishedFrame and FunctionActionFrame are uninterruptible.

        These frames carry the ongoing-actions count's decrement signal, so they have to
        survive an interruption's queue flush (see FrameQueue.reset()).
        """

        async def dummy(action, flow_manager):
            pass

        self.assertIsInstance(ActionFinishedFrame(), UninterruptibleFrame)
        self.assertIsInstance(FunctionActionFrame(action={}, function=dummy), UninterruptibleFrame)

    async def test_action_frames_survive_frame_queue_reset(self):
        """Test that the action frames survive a real FrameQueue.reset().

        An interruptible frame queued ahead of a completion frame is dropped, while the
        completion frame itself stays in the queue.
        """

        async def dummy(action, flow_manager):
            pass

        for completion_frame in (
            ActionFinishedFrame(),
            FunctionActionFrame(action={}, function=dummy),
        ):
            with self.subTest(frame=type(completion_frame).__name__):
                queue = FrameQueue()
                queue.put_nowait(TTSSpeakFrame(text="Hello"))
                queue.put_nowait(completion_frame)

                queue.reset()

                remaining = []
                while not queue.empty():
                    remaining.append(queue.get_nowait())
                self.assertEqual(remaining, [completion_frame])

    async def test_tts_action_completes_when_interrupted(self):
        """Test that a tts_say action completes even if interrupted.

        The TTSSpeakFrame is dropped by the interruption flush but the ActionFinishedFrame
        survives, so execute_actions() returns and the ongoing actions count/event resync.
        """
        self.mock_task.interrupted = True

        await asyncio.wait_for(
            self.action_manager.execute_actions([{"type": "tts_say", "text": "Hello"}]), 1.0
        )

        self.assertEqual(self.action_manager._ongoing_actions_count, 0)
        self.assertTrue(self.action_manager._ongoing_actions_finished_event.is_set())

    async def test_function_action_completes_when_interrupted(self):
        """Test that a function action's handler still runs even if interrupted.

        The FunctionActionFrame survives the interruption flush, so the handler runs,
        execute_actions() returns, and the ongoing actions count/event resync.
        """
        ran = False

        async def handler(action, flow_manager):
            nonlocal ran
            ran = True

        self.mock_task.interrupted = True

        await asyncio.wait_for(
            self.action_manager.execute_actions([{"type": "function", "handler": handler}]), 1.0
        )

        self.assertTrue(ran)
        self.assertEqual(self.action_manager._ongoing_actions_count, 0)
        self.assertTrue(self.action_manager._ongoing_actions_finished_event.is_set())

    async def test_function_action_raising_handler_resets_count(self):
        """Test count consistency when a raising handler's exception unwinds synchronously.

        The mock task delivers frames inline, so the handler's exception propagates back
        through execute_actions() as an ActionError. The count must land back at 0 without
        execute_actions()'s snapshot-based guard double-decrementing.
        """

        async def failing_handler(action, flow_manager):
            raise RuntimeError("boom")

        with self.assertRaises(ActionError):
            await self.action_manager.execute_actions(
                [{"type": "function", "handler": failing_handler}]
            )

        self.assertEqual(self.action_manager._ongoing_actions_count, 0)
        self.assertTrue(self.action_manager._ongoing_actions_finished_event.is_set())

    async def test_raising_function_handler_decrements_count_on_dispatch(self):
        """Test that a raising handler can't leave the count stuck in real frame dispatch.

        In a real pipeline, on_frame_reached_downstream runs in its own task and handler
        exceptions are logged and swallowed (BaseObject._run_handler) rather than
        propagating back to execute_actions(), so the decrement around the handler
        invocation is the only thing that resyncs the count. Invoking the captured handler
        directly exercises that path in isolation.
        """

        async def failing_handler(action, flow_manager):
            raise RuntimeError("boom")

        frame = FunctionActionFrame(action={}, function=failing_handler)
        self.action_manager._increment_ongoing_actions_count()

        with self.assertRaises(RuntimeError):
            await self.mock_task.on_frame_reached_downstream(self.mock_task, frame)

        self.assertEqual(self.action_manager._ongoing_actions_count, 0)
        self.assertTrue(self.action_manager._ongoing_actions_finished_event.is_set())
