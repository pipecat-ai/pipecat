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
from pipecat.pipeline.pipeline import Pipeline
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
            expected_up_frames=[],  # Expect no error frames
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
            expected_up_frames=[],  # Expect no error frames
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

    async def test_multi_service_switcher_targeting(self):
        """Test that ManuallySwitchServiceFrame targets the correct ServiceSwitcher in a multi-switcher pipeline."""
        # Create services for first switcher
        switcher1_service1 = MockFrameProcessor("switcher1_service1")
        switcher1_service2 = MockFrameProcessor("switcher1_service2")
        switcher1_services = [switcher1_service1, switcher1_service2]

        # Create services for second switcher
        switcher2_service1 = MockFrameProcessor("switcher2_service1")
        switcher2_service2 = MockFrameProcessor("switcher2_service2")
        switcher2_services = [switcher2_service1, switcher2_service2]

        # Create two service switchers
        switcher1 = ServiceSwitcher(switcher1_services, ServiceSwitcherStrategyManual)
        switcher2 = ServiceSwitcher(switcher2_services, ServiceSwitcherStrategyManual)

        # Create a pipeline with both switchers: switcher1 -> switcher2
        pipeline = Pipeline([switcher1, switcher2])

        # Reset counters
        for service in switcher1_services + switcher2_services:
            service.reset_counters()

        # Initially, both switchers should use their first services
        self.assertEqual(switcher1.strategy.active_service, switcher1_service1)
        self.assertEqual(switcher2.strategy.active_service, switcher2_service1)

        # Send frames to test the pipeline:
        # 1. Text frame (should go through both switchers' active services)
        # 2. Switch frame targeting switcher1's second service
        # 3. Text frame (should go through switcher1's new service and switcher2's original service)
        # 4. Switch frame targeting switcher2's second service
        # 5. Text frame (should go through switcher1's current service and switcher2's new service)
        await run_test(
            pipeline,
            frames_to_send=[
                TextFrame("Before any switches"),
                ManuallySwitchServiceFrame(service=switcher1_service2),  # Switch first switcher
                TextFrame("After switching first switcher"),
                ManuallySwitchServiceFrame(service=switcher2_service2),  # Switch second switcher
                TextFrame("After switching second switcher"),
            ],
            expected_down_frames=[
                TextFrame,
                ManuallySwitchServiceFrame,
                TextFrame,
                ManuallySwitchServiceFrame,
                TextFrame,
            ],
            expected_up_frames=[],  # Expect no error frames
        )

        # Verify the active services changed correctly
        self.assertEqual(switcher1.strategy.active_service, switcher1_service2)
        self.assertEqual(switcher2.strategy.active_service, switcher2_service2)

        # Verify frame distribution:
        # First text frame should go through switcher1_service1 and switcher2_service1
        switcher1_service1_texts = [
            f for f in switcher1_service1.processed_frames if isinstance(f, TextFrame)
        ]
        switcher2_service1_texts = [
            f for f in switcher2_service1.processed_frames if isinstance(f, TextFrame)
        ]

        # Second text frame should go through switcher1_service2 and switcher2_service1
        switcher1_service2_texts = [
            f for f in switcher1_service2.processed_frames if isinstance(f, TextFrame)
        ]

        # Third text frame should go through switcher1_service2 and switcher2_service2
        switcher2_service2_texts = [
            f for f in switcher2_service2.processed_frames if isinstance(f, TextFrame)
        ]

        # Verify frame counts and content
        self.assertEqual(len(switcher1_service1_texts), 1)
        self.assertEqual(switcher1_service1_texts[0].text, "Before any switches")

        self.assertEqual(len(switcher1_service2_texts), 2)
        self.assertEqual(switcher1_service2_texts[0].text, "After switching first switcher")
        self.assertEqual(switcher1_service2_texts[1].text, "After switching second switcher")

        self.assertEqual(len(switcher2_service1_texts), 2)
        self.assertEqual(switcher2_service1_texts[0].text, "Before any switches")
        self.assertEqual(switcher2_service1_texts[1].text, "After switching first switcher")

        self.assertEqual(len(switcher2_service2_texts), 1)
        self.assertEqual(switcher2_service2_texts[0].text, "After switching second switcher")


if __name__ == "__main__":
    unittest.main()
