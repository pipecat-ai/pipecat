#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock

from pipecat.extensions.ivr.ivr_navigator import IVRProcessor
from pipecat.frames.frames import (
    ErrorFrame,
    LLMMessagesUpdateFrame,
    LLMTextFrame,
    OutputDTMFUrgentFrame,
    StartFrame,
    TextFrame,
    VADParamsUpdateFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import run_test


class TestIVRNavigation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock LLM service
        self.mock_llm = AsyncMock(spec=LLMService)

        # Test prompts
        self.ivr_prompt = "Navigate to the billing department"
        self.conversation_prompt = "You are a helpful customer service agent"

    async def test_switch_to_ivr_mode(self):
        """Test switching to IVR mode from conversation mode."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="conversation",
        )

        frames_to_send = [
            # LLM responds with DTMF command
            LLMTextFrame(text="<ivr>detected</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the conversation prompt
            VADParamsUpdateFrame,  # Initialize the conversation VAD parameters
            LLMMessagesUpdateFrame,  # Switch to the ivr prompt
            VADParamsUpdateFrame,  # Switch to the ivr VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_basic_dtmf_navigation(self):
        """Test basic DTMF tone generation from LLM responses."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        frames_to_send = [
            LLMTextFrame(text="<dtmf>1</dtmf>"),
        ]

        expected_down_frames = [
            OutputDTMFUrgentFrame,
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_multiple_dtmf_navigation(self):
        """Test basic DTMF tone generation from LLM responses."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        frames_to_send = [
            LLMTextFrame(text="<dtmf>1</dtmf>"),
            LLMTextFrame(text="<dtmf>2</dtmf>"),
            LLMTextFrame(text="<dtmf>3</dtmf>"),
            LLMTextFrame(text="<dtmf>4</dtmf>"),
        ]

        expected_down_frames = [
            OutputDTMFUrgentFrame,
            TextFrame,  # Context frame with skip_tts=True
            OutputDTMFUrgentFrame,
            TextFrame,  # Context frame with skip_tts=True
            OutputDTMFUrgentFrame,
            TextFrame,  # Context frame with skip_tts=True
            OutputDTMFUrgentFrame,
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_ivr_wait(self):
        """Test basic DTMF tone generation from LLM responses."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        frames_to_send = [
            LLMTextFrame(text="<ivr>wait</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_ivr_stuck(self):
        """Test that on_ivr_stuck event handler is called when stuck."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        # Mock event handler
        stuck_handler_called = False
        received_processor = None

        async def mock_stuck_handler(ivr_processor):
            nonlocal stuck_handler_called, received_processor
            stuck_handler_called = True
            received_processor = ivr_processor

        # Register the event handler
        processor.add_event_handler("on_ivr_stuck", mock_stuck_handler)

        frames_to_send = [
            LLMTextFrame(text="<ivr>stuck</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify the event handler was called
        self.assertTrue(stuck_handler_called, "on_ivr_stuck event handler should have been called")
        self.assertEqual(
            received_processor, processor, "Event handler should receive the IVRProcessor instance"
        )

    async def test_ivr_completed(self):
        """Test that on_ivr_completed event handler is called when completed."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        # Mock event handler
        completed_handler_called = False
        received_processor = None

        async def mock_completed_handler(ivr_processor):
            nonlocal completed_handler_called, received_processor
            completed_handler_called = True
            received_processor = ivr_processor

        # Register the event handler
        processor.add_event_handler("on_ivr_completed", mock_completed_handler)

        frames_to_send = [
            LLMTextFrame(text="<ivr>completed</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify the event handler was called
        self.assertTrue(
            completed_handler_called, "on_ivr_completed event handler should have been called"
        )
        self.assertEqual(
            received_processor, processor, "Event handler should receive the IVRProcessor instance"
        )

    async def test_ivr_completed_with_start_conversation(self):
        """Test that start_conversation() works when called from event handler."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        # Mock event handler that calls start_conversation
        conversation_started = False

        async def mock_completed_handler(ivr_processor):
            nonlocal conversation_started
            await ivr_processor.start_conversation()
            conversation_started = True

        # Register the event handler
        processor.add_event_handler("on_ivr_completed", mock_completed_handler)

        frames_to_send = [
            LLMTextFrame(text="<ivr>completed</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
            LLMMessagesUpdateFrame,  # Start conversation mode
            VADParamsUpdateFrame,  # Conversation VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify conversation was started
        self.assertTrue(conversation_started, "start_conversation() should have been called")

    async def test_ivr_completed_without_conversation_prompt(self):
        """Test that start_conversation() fails gracefully without conversation_prompt."""
        # Create processor without conversation prompt
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=None,  # No conversation prompt
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        # Mock event handler that tries to call start_conversation
        handler_called = False
        conversation_attempted = False

        async def mock_completed_handler(ivr_processor):
            nonlocal handler_called, conversation_attempted
            handler_called = True
            await ivr_processor.start_conversation()
            conversation_attempted = True

        # Register the event handler
        processor.add_event_handler("on_ivr_completed", mock_completed_handler)

        frames_to_send = [
            LLMTextFrame(text="<ivr>completed</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
            # No conversation frames should be sent
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify handler was called but conversation didn't start
        self.assertTrue(handler_called, "Event handler should have been called")
        self.assertTrue(conversation_attempted, "start_conversation() should have been attempted")

    async def test_normal_text_passthrough(self):
        """Test that normal LLM text (no XML) passes through unchanged."""
        processor = IVRProcessor(
            ivr_prompt=self.ivr_prompt,
            conversation_prompt=self.conversation_prompt,
            ivr_response_delay=2.0,
            conversation_response_delay=0.8,
            initial_mode="ivr",
        )

        frames_to_send = [
            LLMTextFrame(text="Hello, I'm trying to reach billing."),
        ]

        expected_down_frames = [
            LLMTextFrame,  # Should pass through unchanged
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the IVR prompt
            VADParamsUpdateFrame,  # Initialize the VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
