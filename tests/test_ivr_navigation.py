#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest
from unittest.mock import AsyncMock

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.extensions.ivr.ivr_navigator import IVRProcessor
from pipecat.frames.frames import (
    LLMMessagesUpdateFrame,
    LLMTextFrame,
    OutputDTMFUrgentFrame,
    TextFrame,
    VADParamsUpdateFrame,
)
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import run_test


class TestIVRNavigation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock LLM service
        self.mock_llm = AsyncMock(spec=LLMService)

        # Test prompts
        self.classifier_prompt = "Classify as IVR or conversation"
        self.ivr_prompt = "Navigate to the billing department"

        # VAD parameters
        self.ivr_vad_params = VADParams(stop_secs=2.0)

    async def test_switch_to_ivr_mode(self):
        """Test switching to IVR mode from conversation mode."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
        )

        frames_to_send = [
            LLMTextFrame(text="<mode>ivr</mode>"),
        ]

        expected_down_frames = [
            # No frames expected - mode selection doesn't push TextFrame
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
            LLMMessagesUpdateFrame,  # Switch to the ivr prompt
            VADParamsUpdateFrame,  # Switch to the ivr VAD parameters
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_switch_to_conversation_mode(self):
        """Test switching to conversation mode when conversation is detected."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
        )

        # Mock event handler
        conversation_handler_called = False
        received_processor = None
        received_history = None

        async def mock_conversation_handler(ivr_processor, conversation_history):
            nonlocal conversation_handler_called, received_processor, received_history
            conversation_handler_called = True
            received_processor = ivr_processor
            received_history = conversation_history

        # Register the event handler
        processor.add_event_handler("on_conversation_detected", mock_conversation_handler)

        frames_to_send = [
            LLMTextFrame(text="<mode>conversation</mode>"),
        ]

        expected_down_frames = [
            # No frames expected - mode selection doesn't push TextFrame
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify the event handler was called
        self.assertTrue(
            conversation_handler_called,
            "on_conversation_detected event handler should have been called",
        )
        self.assertEqual(
            received_processor, processor, "Event handler should receive the IVRProcessor instance"
        )
        self.assertEqual(
            received_history,
            [],
            "Event handler should receive empty conversation history initially",
        )

    async def test_basic_dtmf_navigation(self):
        """Test basic DTMF tone generation from LLM responses."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
        )

        frames_to_send = [
            LLMTextFrame(text="<dtmf>1</dtmf>"),
        ]

        expected_down_frames = [
            OutputDTMFUrgentFrame,
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
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
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
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
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
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
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
        )

        frames_to_send = [
            LLMTextFrame(text="<ivr>wait</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

    async def test_ivr_stuck(self):
        """Test that on_ivr_status_changed event handler is called when stuck."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
        )

        # Mock event handler
        stuck_handler_called = False
        received_processor = None
        received_status = None

        async def mock_status_handler(ivr_processor, status):
            nonlocal stuck_handler_called, received_processor, received_status
            if status.value == "stuck":
                stuck_handler_called = True
                received_processor = ivr_processor
                received_status = status

        # Register the event handler
        processor.add_event_handler("on_ivr_status_changed", mock_status_handler)

        frames_to_send = [
            LLMTextFrame(text="<ivr>stuck</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify the event handler was called
        self.assertTrue(
            stuck_handler_called,
            "on_ivr_status_changed event handler should have been called for stuck status",
        )
        self.assertEqual(
            received_processor, processor, "Event handler should receive the IVRProcessor instance"
        )
        self.assertEqual(
            received_status.value, "stuck", "Event handler should receive IVRStatus.STUCK"
        )

    async def test_ivr_completed(self):
        """Test that on_ivr_status_changed event handler is called when completed."""
        # Create just the IVR processor to test in isolation
        processor = IVRProcessor(
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
        )

        # Mock event handler
        completed_handler_called = False
        received_processor = None
        received_status = None

        async def mock_status_handler(ivr_processor, status):
            nonlocal completed_handler_called, received_processor, received_status
            if status.value == "completed":
                completed_handler_called = True
                received_processor = ivr_processor
                received_status = status

        # Register the event handler
        processor.add_event_handler("on_ivr_status_changed", mock_status_handler)

        frames_to_send = [
            LLMTextFrame(text="<ivr>completed</ivr>"),
        ]

        expected_down_frames = [
            TextFrame,  # Context frame with skip_tts=True
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify the event handler was called
        self.assertTrue(
            completed_handler_called,
            "on_ivr_status_changed event handler should have been called for completed status",
        )
        self.assertEqual(
            received_processor, processor, "Event handler should receive the IVRProcessor instance"
        )
        self.assertEqual(
            received_status.value, "completed", "Event handler should receive IVRStatus.COMPLETED"
        )

    async def test_normal_text_passthrough(self):
        """Test that normal LLM text (no XML) passes through unchanged."""
        processor = IVRProcessor(
            classifier_prompt=self.classifier_prompt,
            ivr_prompt=self.ivr_prompt,
            ivr_vad_params=self.ivr_vad_params,
        )

        frames_to_send = [
            LLMTextFrame(text="Hello, I'm trying to reach billing."),
        ]

        expected_down_frames = [
            LLMTextFrame,  # Should pass through unchanged
        ]

        expected_up_frames = [
            LLMMessagesUpdateFrame,  # Initialize the mode detection prompt
        ]

        await run_test(
            processor,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
