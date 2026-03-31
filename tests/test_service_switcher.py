#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for ServiceSwitcher and related components."""

import asyncio
import unittest
from dataclasses import dataclass

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    ManuallySwitchServiceFrame,
    ServiceMetadataFrame,
    ServiceSwitcherRequestMetadataFrame,
    StartFrame,
    SystemFrame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import (
    ServiceSwitcher,
    ServiceSwitcherStrategy,
    ServiceSwitcherStrategyFailover,
    ServiceSwitcherStrategyManual,
)
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


@dataclass
class MockMetadataFrame(ServiceMetadataFrame):
    """A mock metadata frame for testing ServiceMetadataFrame handling."""

    pass


class MockMetadataService(FrameProcessor):
    """A mock service that emits ServiceMetadataFrame like STT services.

    Pushes MockMetadataFrame on StartFrame and ServiceSwitcherRequestMetadataFrame.
    """

    def __init__(self, test_name: str, **kwargs):
        super().__init__(name=test_name, **kwargs)
        self.test_name = test_name
        self.processed_frames = []
        self.metadata_push_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.processed_frames.append(frame)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._push_metadata()
        elif isinstance(frame, ServiceSwitcherRequestMetadataFrame):
            await self._push_metadata()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _push_metadata(self):
        self.metadata_push_count += 1
        await self.push_frame(MockMetadataFrame(service_name=self.test_name))

    def reset_counters(self):
        self.processed_frames = []
        self.metadata_push_count = 0


class ErrorInjectorProcessor(FrameProcessor):
    """A downstream processor that pushes an ErrorFrame upstream on receiving a TextFrame.

    Simulates an error from a service outside the ServiceSwitcher (e.g. TTS
    erroring while propagating upstream through an LLM switcher).
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_error("downstream service error")
        await self.push_frame(frame, direction)


class ErrorOnTextService(FrameProcessor):
    """A mock service that pushes an error on the first TextFrame it receives.

    Simulates a managed service inside a ServiceSwitcher that encounters an error.
    """

    def __init__(self, test_name: str, **kwargs):
        super().__init__(name=test_name, **kwargs)
        self._errored = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and not self._errored:
            self._errored = True
            await self.push_error("service connection lost")
        await self.push_frame(frame, direction)


@dataclass
class DummySystemFrame(SystemFrame):
    """A dummy system frame for testing purposes."""

    text: str = ""


class TestServiceSwitcherStrategy(unittest.IsolatedAsyncioTestCase):
    """Test cases for the base ServiceSwitcherStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.service1 = MockFrameProcessor("service1")
        self.service2 = MockFrameProcessor("service2")
        self.service3 = MockFrameProcessor("service3")
        self.services = [self.service1, self.service2, self.service3]

    def test_init_with_services(self):
        """Test initialization with a list of services."""
        strategy = ServiceSwitcherStrategy(self.services)

        self.assertEqual(strategy.services, self.services)
        self.assertEqual(strategy.active_service, self.service1)

    async def test_handle_frame_returns_none_for_manual_switch(self):
        """Test that base strategy does not handle ManuallySwitchServiceFrame."""
        strategy = ServiceSwitcherStrategy(self.services)

        switch_frame = ManuallySwitchServiceFrame(service=self.service2)
        result = await strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)

        self.assertIsNone(result)
        self.assertEqual(strategy.active_service, self.service1)

    async def test_handle_frame_returns_none_for_unsupported_frame(self):
        """Test that unsupported frame types return None."""
        strategy = ServiceSwitcherStrategy(self.services)
        unsupported_frame = TextFrame(text="test")

        result = await strategy.handle_frame(unsupported_frame, FrameDirection.DOWNSTREAM)

        self.assertIsNone(result)

    async def test_handle_error_returns_none(self):
        """Test that handle_error returns None by default."""
        strategy = ServiceSwitcherStrategy(self.services)

        result = await strategy.handle_error(ErrorFrame(error="error"))

        self.assertIsNone(result)
        self.assertEqual(strategy.active_service, self.service1)


class TestServiceSwitcherStrategyManual(unittest.IsolatedAsyncioTestCase):
    """Test cases for ServiceSwitcherStrategyManual."""

    def setUp(self):
        """Set up test fixtures."""
        self.service1 = MockFrameProcessor("service1")
        self.service2 = MockFrameProcessor("service2")
        self.service3 = MockFrameProcessor("service3")
        self.services = [self.service1, self.service2, self.service3]

    def test_is_subclass_of_base_strategy(self):
        """Test that ServiceSwitcherStrategyManual is a subclass of ServiceSwitcherStrategy."""
        strategy = ServiceSwitcherStrategyManual(self.services)
        self.assertIsInstance(strategy, ServiceSwitcherStrategy)

    async def test_handle_manually_switch_service_frame(self):
        """Test manual service switching with ManuallySwitchServiceFrame."""
        strategy = ServiceSwitcherStrategyManual(self.services)

        # Initially service1 should be active
        self.assertEqual(strategy.active_service, self.service1)

        # Switch to service2
        switch_frame = ManuallySwitchServiceFrame(service=self.service2)
        await strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(strategy.active_service, self.service2)

        # Switch to service3
        switch_frame = ManuallySwitchServiceFrame(service=self.service3)
        await strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(strategy.active_service, self.service3)

    async def test_on_service_switched_event(self):
        """Test that on_service_switched event fires with correct arguments."""
        strategy = ServiceSwitcherStrategyManual(self.services)

        switched_events = []

        @strategy.event_handler("on_service_switched")
        async def on_service_switched(strategy, service):
            switched_events.append((strategy, service))

        switch_frame = ManuallySwitchServiceFrame(service=self.service2)
        await strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        self.assertEqual(len(switched_events), 1)
        self.assertIsInstance(switched_events[0][0], ServiceSwitcherStrategyManual)
        self.assertEqual(switched_events[0][1], self.service2)

    async def test_unknown_service_ignored(self):
        """Test that switching to an unknown service is ignored."""
        strategy = ServiceSwitcherStrategyManual(self.services)

        switched_events = []

        @strategy.event_handler("on_service_switched")
        async def on_service_switched(strategy, service):
            switched_events.append(service)

        unknown_service = MockFrameProcessor("unknown")
        switch_frame = ManuallySwitchServiceFrame(service=unknown_service)
        result = await strategy.handle_frame(switch_frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0)

        self.assertIsNone(result)
        self.assertEqual(len(switched_events), 0)
        self.assertEqual(strategy.active_service, self.service1)


class TestServiceSwitcher(unittest.IsolatedAsyncioTestCase):
    """Test cases for ServiceSwitcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.service1 = MockFrameProcessor("service1")
        self.service2 = MockFrameProcessor("service2")
        self.service3 = MockFrameProcessor("service3")
        self.services = [self.service1, self.service2, self.service3]

    def test_init_with_default_strategy(self):
        """Test initialization with default strategy."""
        switcher = ServiceSwitcher(self.services)

        self.assertEqual(switcher.services, self.services)
        self.assertIsInstance(switcher.strategy, ServiceSwitcherStrategyManual)
        self.assertEqual(switcher.strategy.services, self.services)

    async def test_default_active_service(self):
        """Test that the initially-active service receives frames while others don't."""
        switcher = ServiceSwitcher(self.services)

        # Reset counters
        for service in self.services:
            service.reset_counters()

        # Send some test frames
        frames_to_send = [
            TextFrame(text="Hello 1"),
            DummySystemFrame(text="System Message 1"),
            TextFrame(text="Hello 2"),
            DummySystemFrame(text="System Message 2"),
            TextFrame(text="Hello 3"),
        ]

        await run_test(
            switcher,
            frames_to_send=frames_to_send,
            expected_down_frames=[
                DummySystemFrame,
                DummySystemFrame,
                TextFrame,
                TextFrame,
                TextFrame,
            ],
            expected_up_frames=[],  # Expect no error frames
        )

        # Only service1 should have processed the text frames
        # Note: The service also receives StartFrame and EndFrame, so count those too
        text_frames = [f for f in self.service1.processed_frames if isinstance(f, TextFrame)]
        self.assertEqual(len(text_frames), 3)

        # Only service1 should have processed the system frames
        system_frames = [
            f for f in self.service1.processed_frames if isinstance(f, DummySystemFrame)
        ]
        self.assertEqual(len(system_frames), 2)

        # Check that other services don't receive text frames (they still get StartFrame/EndFrame)
        service2_text_frames = [
            f for f in self.service2.processed_frames if isinstance(f, TextFrame)
        ]
        service3_text_frames = [
            f for f in self.service3.processed_frames if isinstance(f, TextFrame)
        ]
        self.assertEqual(len(service2_text_frames), 0)
        self.assertEqual(len(service3_text_frames), 0)

        # Check that other services don't receive dummy system frames (they still get StartFrame/EndFrame)
        service2_system_frames = [
            f for f in self.service2.processed_frames if isinstance(f, DummySystemFrame)
        ]
        service3_system_frames = [
            f for f in self.service3.processed_frames if isinstance(f, DummySystemFrame)
        ]
        self.assertEqual(len(service2_system_frames), 0)
        self.assertEqual(len(service3_system_frames), 0)

        # Verify the actual text frames processed
        for i, frame in enumerate(text_frames):
            self.assertEqual(frame.text, f"Hello {i + 1}")

        # Verify the actual system frames processed
        for i, frame in enumerate(system_frames):
            self.assertEqual(frame.text, f"System Message {i + 1}")

    async def test_service_switching(self):
        """Test that after service switching using ManuallySwitchServiceFrame, the new active service receives frames while others don't."""
        switcher = ServiceSwitcher(self.services)

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
            expected_down_frames=[TextFrame, TextFrame],
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
        switcher1 = ServiceSwitcher(switcher1_services)
        switcher2 = ServiceSwitcher(switcher2_services)

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
                TextFrame,
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


class TestServiceSwitcherMetadata(unittest.IsolatedAsyncioTestCase):
    """Test cases for ServiceMetadataFrame handling in ServiceSwitcher."""

    def setUp(self):
        """Set up test fixtures with mock metadata services."""
        self.service1 = MockMetadataService("service1")
        self.service2 = MockMetadataService("service2")
        self.services = [self.service1, self.service2]

    async def test_only_active_service_metadata_at_startup(self):
        """Test that only the active service's metadata leaves the ServiceSwitcher at startup."""
        switcher = ServiceSwitcher(self.services)

        # Run the pipeline (StartFrame triggers metadata emission)
        output_frames = []

        async def capture_frame(frame: Frame):
            output_frames.append(frame)

        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[MockMetadataFrame, TextFrame],
            expected_up_frames=[],
        )

        # Both services push metadata internally on StartFrame, but only the
        # active service's metadata passes through the filter
        self.assertEqual(self.service1.metadata_push_count, 1)  # StartFrame (passes filter)
        self.assertEqual(self.service2.metadata_push_count, 1)  # StartFrame (blocked by filter)

    async def test_metadata_emitted_on_service_switch(self):
        """Test that switching services triggers metadata emission from the new active service."""
        switcher = ServiceSwitcher(self.services)

        # Reset counters after startup
        self.service1.reset_counters()
        self.service2.reset_counters()

        await run_test(
            switcher,
            frames_to_send=[
                TextFrame(text="before switch"),
                ManuallySwitchServiceFrame(service=self.service2),
                TextFrame(text="after switch"),
            ],
            expected_down_frames=[
                MockMetadataFrame,  # From startup (service1)
                TextFrame,
                MockMetadataFrame,  # From service2 after switch
                TextFrame,
            ],
            expected_up_frames=[],
        )

        # service2 should have received ServiceSwitcherRequestMetadataFrame after becoming active
        request_frames = [
            f
            for f in self.service2.processed_frames
            if isinstance(f, ServiceSwitcherRequestMetadataFrame)
        ]
        self.assertEqual(len(request_frames), 1)

    async def test_inactive_service_metadata_blocked(self):
        """Test that metadata from inactive services is blocked."""
        switcher = ServiceSwitcher(self.services)

        # Run and collect output frames
        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[MockMetadataFrame, TextFrame],
            expected_up_frames=[],
        )

        # service2 pushed metadata on StartFrame, but it should have been blocked
        self.assertGreaterEqual(self.service2.metadata_push_count, 1)
        # Only one MockMetadataFrame should have left (from service1)


class TestServiceSwitcherStrategyFailover(unittest.IsolatedAsyncioTestCase):
    """Test cases for ServiceSwitcherStrategyFailover."""

    def setUp(self):
        """Set up test fixtures."""
        self.service1 = MockFrameProcessor("service1")
        self.service2 = MockFrameProcessor("service2")
        self.service3 = MockFrameProcessor("service3")
        self.services = [self.service1, self.service2, self.service3]

    def test_init_defaults(self):
        """Test that default values are set correctly."""
        strategy = ServiceSwitcherStrategyFailover(self.services)
        self.assertEqual(strategy.active_service, self.service1)

    async def test_error_switches_to_next_service(self):
        """Test that an error on the active service switches to the next one."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        error = ErrorFrame(error="connection lost")
        result = await strategy.handle_error(error)

        self.assertEqual(result, self.service2)
        self.assertEqual(strategy.active_service, self.service2)

    async def test_consecutive_errors_cycle_through_services(self):
        """Test that repeated errors cycle through all services."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        # First error: service1 -> service2
        await strategy.handle_error(ErrorFrame(error="error 1"))
        self.assertEqual(strategy.active_service, self.service2)

        # Second error: service2 -> service3
        await strategy.handle_error(ErrorFrame(error="error 2"))
        self.assertEqual(strategy.active_service, self.service3)

        # Third error: service3 -> service1 (wraps around)
        await strategy.handle_error(ErrorFrame(error="error 3"))
        self.assertEqual(strategy.active_service, self.service1)

    async def test_single_service_returns_none(self):
        """Test that handle_error returns None with only one service."""
        strategy = ServiceSwitcherStrategyFailover([self.service1])

        result = await strategy.handle_error(ErrorFrame(error="error"))
        self.assertIsNone(result)

    async def test_manual_switch_still_works(self):
        """Test that ManuallySwitchServiceFrame is still handled."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        frame = ManuallySwitchServiceFrame(service=self.service3)
        result = await strategy.handle_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(result, self.service3)
        self.assertEqual(strategy.active_service, self.service3)

    async def test_passthrough_error_does_not_trigger_failover(self):
        """Test that an error propagating upstream from a downstream processor does not trigger failover.

        This reproduces the bug where an ErrorFrame from e.g. TTS propagates
        upstream through an LLM ServiceSwitcher and incorrectly triggers
        failover even though neither LLM service produced the error.
        """
        switcher = ServiceSwitcher(
            [self.service1, self.service2],
            strategy_type=ServiceSwitcherStrategyFailover,
        )
        error_injector = ErrorInjectorProcessor()
        pipeline = Pipeline([switcher, error_injector])

        await run_test(
            pipeline,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[ErrorFrame],
        )

        # Active service should NOT have changed — the error came from outside
        self.assertEqual(switcher.strategy.active_service, self.service1)

    async def test_managed_service_error_triggers_failover(self):
        """Test that an error from a managed service inside the switcher triggers failover."""
        error_service = ErrorOnTextService("error_service")
        backup_service = MockFrameProcessor("backup_service")
        switcher = ServiceSwitcher(
            [error_service, backup_service],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[ErrorFrame],
        )

        # Active service SHOULD have changed — the error came from a managed service
        self.assertEqual(switcher.strategy.active_service, backup_service)

    async def test_on_service_switched_event_fires_on_error(self):
        """Test that on_service_switched event fires when an error triggers a switch."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        switched_events = []

        @strategy.event_handler("on_service_switched")
        async def on_service_switched(strategy, service):
            switched_events.append(service)

        await strategy.handle_error(ErrorFrame(error="error"))
        await asyncio.sleep(0)

        self.assertEqual(len(switched_events), 1)
        self.assertEqual(switched_events[0], self.service2)


if __name__ == "__main__":
    unittest.main()
