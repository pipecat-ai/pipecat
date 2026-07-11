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

from pipecat.flows.actions import ActionManager
from pipecat.flows.exceptions import ActionError
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
