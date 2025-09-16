#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for ServiceSwitcher and related components."""

import unittest

from pipecat.frames.frames import (
    Frame,
    ManuallySwitchServiceFrame,
    TextFrame,
)
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyManual
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import run_test


class MockFrameProcessor(FrameProcessor):
    """A test frame processor that tracks which frames it has processed."""

    def __init__(self, test_name: str, **kwargs):
        """Initialize the test processor with a name.

        Args:
            test_name: A unique name for this processor instance.
            **kwargs: Additional arguments passed to the parent FrameProcessor.
        """
        super().__init__(name=test_name, **kwargs)
        self.test_name = test_name
        self.processed_frames = []
        self.frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process an incoming frame and track it.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)
        self.processed_frames.append(frame)
        self.frame_count += 1
        await self.push_frame(frame, direction)

    def reset_counters(self):
        """Reset the frame tracking counters."""
        self.processed_frames = []
        self.frame_count = 0


class TestServiceSwitcherStrategyManual(unittest.IsolatedAsyncioTestCase):
    """Test cases for ServiceSwitcherStrategyManual."""

    def setUp(self):
        """Set up test fixtures."""
        self.service1 = MockFrameProcessor("service1")
        self.service2 = MockFrameProcessor("service2")
        self.service3 = MockFrameProcessor("service3")
        self.services = [self.service1, self.service2, self.service3]

    def test_init_with_services(self):
        """Test initialization with a list of services."""
        strategy = ServiceSwitcherStrategyManual(self.services)

        self.assertEqual(strategy.services, self.services)
        self.assertEqual(strategy.active_service, self.service1)  # First service should be active

    def test_init_with_empty_services(self):
        """Test initialization with an empty list of services."""
        strategy = ServiceSwitcherStrategyManual([])

        self.assertEqual(strategy.services, [])
        self.assertIsNone(strategy.active_service)

    def test_handle_manually_switch_service_frame(self):
        """Test manual service switching with ManuallySwitchServiceFrame."""
        strategy = ServiceSwitcherStrategyManual(self.services)

        # Initially service1 should be active
        self.assertEqual(strategy.active_service, self.service1)
        self.assertNotEqual(strategy.active_service, self.service2)

        # Switch to service2
        switch_frame = ManuallySwitchServiceFrame(service=self.service2)
        strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)

        self.assertNotEqual(strategy.active_service, self.service1)
        self.assertEqual(strategy.active_service, self.service2)
        self.assertNotEqual(strategy.active_service, self.service3)

        # Switch to service3
        switch_frame = ManuallySwitchServiceFrame(service=self.service3)
        strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)

        self.assertNotEqual(strategy.active_service, self.service1)
        self.assertNotEqual(strategy.active_service, self.service2)
        self.assertEqual(strategy.active_service, self.service3)

    def test_handle_frame_invalid_service(self):
        """Test that switching to an invalid service raises an error."""
        strategy = ServiceSwitcherStrategyManual(self.services)
        invalid_service = MockFrameProcessor("invalid")

        switch_frame = ManuallySwitchServiceFrame(service=invalid_service)

        with self.assertRaises(ValueError) as context:
            strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)

        self.assertIn("Service", str(context.exception))
        self.assertIn("is not in the list of available services", str(context.exception))

    def test_handle_frame_unsupported_frame_type(self):
        """Test that unsupported frame types raise an error."""
        strategy = ServiceSwitcherStrategyManual(self.services)
        unsupported_frame = TextFrame(text="test")  # Not a ServiceSwitcherFrame

        with self.assertRaises(ValueError) as context:
            strategy.handle_frame(unsupported_frame, FrameDirection.DOWNSTREAM)

        self.assertIn("Unsupported frame type", str(context.exception))


class TestServiceSwitcher(unittest.IsolatedAsyncioTestCase):
    """Test cases for ServiceSwitcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.service1 = MockFrameProcessor("service1")
        self.service2 = MockFrameProcessor("service2")
        self.service3 = MockFrameProcessor("service3")
        self.services = [self.service1, self.service2, self.service3]

    def test_init_with_manual_strategy(self):
        """Test initialization with manual strategy."""
        switcher = ServiceSwitcher(self.services, ServiceSwitcherStrategyManual)

        self.assertEqual(switcher.services, self.services)
        self.assertIsInstance(switcher.strategy, ServiceSwitcherStrategyManual)
        self.assertEqual(switcher.strategy.services, self.services)

    async def test_default_active_service(self):
        """Test that the initially-active service receives frames while others don't."""
        switcher = ServiceSwitcher(self.services, ServiceSwitcherStrategyManual)

        # Reset counters
        for service in self.services:
            service.reset_counters()

        # Send some test frames
        frames_to_send = [
            TextFrame(text="Hello 1"),
            TextFrame(text="Hello 2"),
            TextFrame(text="Hello 3"),
        ]

        await run_test(
            switcher,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame, TextFrame, TextFrame],
        )

        # Only service1 should have processed the text frames
        # Note: The service also receives StartFrame and EndFrame, so count those too
        text_frames = [f for f in self.service1.processed_frames if isinstance(f, TextFrame)]
        self.assertEqual(len(text_frames), 3)

        # Check that other services don't receive text frames (they might get StartFrame/EndFrame)
        service2_text_frames = [
            f for f in self.service2.processed_frames if isinstance(f, TextFrame)
        ]
        service3_text_frames = [
            f for f in self.service3.processed_frames if isinstance(f, TextFrame)
        ]
        self.assertEqual(len(service2_text_frames), 0)
        self.assertEqual(len(service3_text_frames), 0)

        # Verify the actual text frames processed
        for i, frame in enumerate(text_frames):
            self.assertEqual(frame.text, f"Hello {i + 1}")

    async def test_service_switching(self):
        """Test that after service switching using ManuallySwitchServiceFrame, the new active service receives frames while others don't."""
        switcher = ServiceSwitcher(self.services, ServiceSwitcherStrategyManual)

        # Reset counters
        for service in self.services:
            service.reset_counters()

        # Send a test frame, a switch frame, and another test frame
        await run_test(
            switcher,
            frames_to_send=[
                TextFrame("Hello 1"),
                ManuallySwitchServiceFrame(service=self.service2),
                TextFrame("Hello 2"),
            ],
            expected_down_frames=[TextFrame, ManuallySwitchServiceFrame, TextFrame],
        )

        # Verify service2 received the frame
        service1_text_frames = [
            f for f in self.service1.processed_frames if isinstance(f, TextFrame)
        ]
        service2_text_frames = [
            f for f in self.service2.processed_frames if isinstance(f, TextFrame)
        ]
        service3_text_frames = [
            f for f in self.service3.processed_frames if isinstance(f, TextFrame)
        ]

        self.assertEqual(len(service1_text_frames), 1)
        self.assertEqual(len(service2_text_frames), 1)
        self.assertEqual(len(service3_text_frames), 0)

        self.assertEqual(service1_text_frames[0].text, "Hello 1")
        self.assertEqual(service2_text_frames[0].text, "Hello 2")


if __name__ == "__main__":
    unittest.main()
