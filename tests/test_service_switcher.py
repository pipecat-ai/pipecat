#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for ServiceSwitcher and related components."""

import asyncio
import unittest
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMUpdateSettingsFrame,
    ManuallySwitchServiceFrame,
    ServiceMetadataFrame,
    ServiceSwitcherRequestMetadataFrame,
    ServiceUpdateSettingsFrame,
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
from pipecat.services.settings import LLMSettings
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory


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

    Simulates a managed service inside a ServiceSwitcher that encounters an
    error. ``becomes_unusable`` chooses between an error the service can carry
    on from and one that ends its usefulness.
    """

    def __init__(
        self,
        test_name: str,
        becomes_unusable: bool = True,
        category: ErrorCategory | None = None,
        **kwargs,
    ):
        super().__init__(name=test_name, **kwargs)
        self._becomes_unusable = becomes_unusable
        self._category = category
        self._errored = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and not self._errored:
            self._errored = True
            await self.push_error(
                "service connection lost",
                category=self._category,
                force_treat_as_permanent=self._becomes_unusable,
            )
        await self.push_frame(frame, direction)


class RepeatedlyErroringService(FrameProcessor):
    """A mock service that goes on erroring after the switcher has moved off it.

    Simulates a websocket service whose reconnect loop keeps reporting: the
    errors after the first come from a background task rather than from frame
    processing, so they arrive once the service is no longer the active one.
    """

    def __init__(self, test_name: str, follow_up_errors: int = 2, **kwargs):
        super().__init__(name=test_name, **kwargs)
        self._follow_up_errors = follow_up_errors
        self.errors = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and self.errors == 0:
            self.errors += 1
            await self.push_error("service connection lost", force_treat_as_permanent=True)
            self.create_task(self._retry_loop(), name="retry")
        await self.push_frame(frame, direction)

    async def _retry_loop(self):
        for attempt in range(self._follow_up_errors):
            await asyncio.sleep(0.02)
            self.errors += 1
            await self.push_error(
                f"reconnection attempt {attempt + 1} failed", force_treat_as_permanent=True
            )


class SlowMockSettingsService(FrameProcessor):
    """A settings-aware service that blocks on a TextFrame, like an LLM mid-inference.

    Frames queued behind the text wait for it, which is what makes the ordering
    of a settings update passing through the switcher observable.
    """

    def __init__(self, test_name: str, text_delay: float = 0.0, **kwargs):
        super().__init__(name=test_name, **kwargs)
        self._text_delay = text_delay

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            await asyncio.sleep(self._text_delay)
        await self.push_frame(frame, direction)


class MockSettingsService(FrameProcessor):
    """A mock service that records the settings updates it receives and applies.

    It applies an update the way a real service does — unless the update is
    addressed to a different service — but forwards every frame either way, so
    that tests can see what leaves the switcher.
    """

    def __init__(self, test_name: str, **kwargs):
        super().__init__(name=test_name, **kwargs)
        self.test_name = test_name
        self.received_settings = []
        self.applied_settings = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ServiceUpdateSettingsFrame):
            self.received_settings.append(frame)
            if frame.service is None or frame.service is self:
                self.applied_settings.append(frame)
        await self.push_frame(frame, direction)

    @property
    def applied_models(self) -> list[str | None]:
        """The models carried by the settings updates this service applied."""
        return [f.delta.model for f in self.applied_settings if f.delta]


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


class TestServiceSwitcherSettingsUpdates(unittest.IsolatedAsyncioTestCase):
    """Test cases for ServiceUpdateSettingsFrame handling in ServiceSwitcher."""

    def setUp(self):
        """Set up test fixtures with mock settings-aware services."""
        self.service1 = MockSettingsService("service1")
        self.service2 = MockSettingsService("service2")
        self.service3 = MockSettingsService("service3")
        self.services = [self.service1, self.service2, self.service3]

    async def test_settings_update_applies_to_the_active_service_alone_by_default(self):
        """A settings update reaches the active service alone unless it opts in.

        Settings values are often specific to one provider: a voice id for one
        TTS service means nothing to the next.
        """
        switcher = ServiceSwitcher(self.services)

        await run_test(
            switcher,
            frames_to_send=[LLMUpdateSettingsFrame(delta=LLMSettings(model="new-model"))],
            expected_down_frames=[LLMUpdateSettingsFrame],
            expected_up_frames=[],
        )

        self.assertEqual(self.service1.applied_models, ["new-model"])
        self.assertEqual(self.service2.received_settings, [])
        self.assertEqual(self.service3.received_settings, [])

    async def test_settings_update_reaches_every_service(self):
        """An update marked reach_inactive_services is applied by inactive services too.

        Their branch filters otherwise gate the update, leaving them to take over
        a session without a setting the rest of the pipeline assumes is in place.
        """
        switcher = ServiceSwitcher(self.services)

        await run_test(
            switcher,
            frames_to_send=[
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(model="new-model"), reach_inactive_services=True
                )
            ],
            # A single copy leaves the switcher, not one per service.
            expected_down_frames=[LLMUpdateSettingsFrame],
            expected_up_frames=[],
        )

        for service in self.services:
            self.assertEqual(service.applied_models, ["new-model"])
            # An untargeted update stays untargeted for every service it reaches:
            # the inactive ones aren't handed an update addressed to them.
            self.assertEqual([f.service for f in service.received_settings], [None])

    async def test_settings_update_travelling_upstream_reaches_every_service(self):
        """An untargeted settings update pushed upstream is applied by inactive services too."""
        switcher = ServiceSwitcher(self.services)

        await run_test(
            switcher,
            frames_to_send=[
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(model="new-model"), reach_inactive_services=True
                )
            ],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_down_frames=[],
            expected_up_frames=[LLMUpdateSettingsFrame],
        )

        for service in self.services:
            self.assertEqual(service.applied_models, ["new-model"])

    async def test_settings_update_addressed_to_inactive_service(self):
        """A settings update addressed to an inactive service is applied by it.

        The ``service`` field is the way to configure one specific service, so it
        has to work for a service that isn't the active one.
        """
        switcher = ServiceSwitcher(self.services)

        await run_test(
            switcher,
            frames_to_send=[
                LLMUpdateSettingsFrame(service=self.service3, delta=LLMSettings(model="new-model"))
            ],
            expected_down_frames=[LLMUpdateSettingsFrame],
            expected_up_frames=[],
        )

        self.assertEqual(self.service3.applied_models, ["new-model"])
        self.assertEqual(self.service1.applied_models, [])
        self.assertEqual(self.service2.applied_models, [])

    async def test_settings_update_addressed_to_active_service(self):
        """A settings update addressed to the active service is applied by it alone."""
        switcher = ServiceSwitcher(self.services)

        await run_test(
            switcher,
            frames_to_send=[
                LLMUpdateSettingsFrame(service=self.service1, delta=LLMSettings(model="new-model"))
            ],
            expected_down_frames=[LLMUpdateSettingsFrame],
            expected_up_frames=[],
        )

        self.assertEqual(self.service1.applied_models, ["new-model"])
        self.assertEqual(self.service2.applied_models, [])
        self.assertEqual(self.service3.applied_models, [])

    async def test_settings_update_for_another_switcher_passes_through_unchanged(self):
        """A settings update travels through a switcher to the service it's addressed to."""
        switcher1_service1 = MockSettingsService("switcher1_service1")
        switcher1_service2 = MockSettingsService("switcher1_service2")
        switcher2_service1 = MockSettingsService("switcher2_service1")
        switcher2_service2 = MockSettingsService("switcher2_service2")

        switcher1 = ServiceSwitcher([switcher1_service1, switcher1_service2])
        switcher2 = ServiceSwitcher([switcher2_service1, switcher2_service2])
        pipeline = Pipeline([switcher1, switcher2])

        await run_test(
            pipeline,
            frames_to_send=[
                LLMUpdateSettingsFrame(
                    service=switcher2_service2, delta=LLMSettings(model="new-model")
                )
            ],
            expected_down_frames=[LLMUpdateSettingsFrame],
            expected_up_frames=[],
        )

        self.assertEqual(switcher2_service2.applied_models, ["new-model"])
        self.assertEqual(switcher2_service1.applied_models, [])
        # The first switcher passes the update along still addressed to its
        # recipient, and leaves its own inactive services out of it.
        self.assertEqual(switcher1_service1.applied_models, [])
        self.assertEqual(
            [f.service for f in switcher1_service1.received_settings], [switcher2_service2]
        )
        self.assertEqual(switcher1_service2.received_settings, [])

    async def test_settings_update_keeps_its_place_in_the_stream(self):
        """An update crossing a switcher leaves it in the order it arrived.

        An idle service is free to handle its copy of the update at once, while
        the active service still has earlier frames in flight.
        """
        active = SlowMockSettingsService("active", text_delay=0.2)
        inactive = SlowMockSettingsService("inactive")
        switcher = ServiceSwitcher([active, inactive])

        await run_test(
            switcher,
            frames_to_send=[
                TextFrame("turn text"),
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(model="new-model"), reach_inactive_services=True
                ),
            ],
            expected_down_frames=[TextFrame, LLMUpdateSettingsFrame],
            expected_up_frames=[],
        )

    async def test_inactive_service_is_configured_before_failover(self):
        """A failover lands on a service that already has the latest settings."""
        switcher = ServiceSwitcher(self.services, strategy_type=ServiceSwitcherStrategyFailover)

        await run_test(
            switcher,
            frames_to_send=[
                LLMUpdateSettingsFrame(
                    delta=LLMSettings(model="new-model"), reach_inactive_services=True
                ),
                ManuallySwitchServiceFrame(service=self.service2),
            ],
            expected_down_frames=[LLMUpdateSettingsFrame],
            expected_up_frames=[],
        )

        self.assertEqual(switcher.strategy.active_service, self.service2)
        self.assertEqual(self.service2.applied_models, ["new-model"])


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
        """Test that an error costing us the active service switches to the next one."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        await self.service1.set_usable(False)
        error = ErrorFrame(error="connection lost")
        result = await strategy.handle_error(error)

        self.assertEqual(result, self.service2)
        self.assertEqual(strategy.active_service, self.service2)

    async def test_recoverable_error_does_not_switch(self):
        """Test that an error the active service can carry on from is ignored."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        result = await strategy.handle_error(ErrorFrame(error="transient failure"))

        self.assertIsNone(result)
        self.assertEqual(strategy.active_service, self.service1)

    async def test_consecutive_errors_cycle_through_services(self):
        """Test that repeated errors cycle through all services."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        # First error: service1 -> service2
        await self.service1.set_usable(False)
        await strategy.handle_error(ErrorFrame(error="error 1"))
        self.assertEqual(strategy.active_service, self.service2)

        # Second error: service2 -> service3
        await self.service2.set_usable(False)
        await strategy.handle_error(ErrorFrame(error="error 2"))
        self.assertEqual(strategy.active_service, self.service3)

        # Third error: service3 -> service1 (wraps around), service1 having
        # been brought back in the meantime.
        await self.service1.set_usable(True)
        await self.service3.set_usable(False)
        await strategy.handle_error(ErrorFrame(error="error 3"))
        self.assertEqual(strategy.active_service, self.service1)

    async def test_single_service_returns_none(self):
        """Test that handle_error returns None with only one service."""
        strategy = ServiceSwitcherStrategyFailover([self.service1])

        await self.service1.set_usable(False)
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
            expected_up_frames=[],
        )

        # Active service SHOULD have changed — the error came from a managed service
        self.assertEqual(switcher.strategy.active_service, backup_service)

    async def test_failover_absorbs_the_error(self):
        """Test that an error the switcher recovered from goes no further.

        The switcher went on doing its job by moving work to another service,
        so there is nothing left for the rest of the pipeline to act on.
        """
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
            expected_up_frames=[],
        )

        self.assertTrue(switcher.is_usable)

    async def test_a_failed_service_that_keeps_erroring_is_answered_for_once(self):
        """Test that a service the switcher moved off doesn't reach the pipeline.

        The rest of the pipeline deals with the switcher, and judges it by the
        processor an error names. A service the switcher has already recovered
        from would have the pipeline write the switcher off for a failure it
        survived.
        """
        error_service = RepeatedlyErroringService("error_service")
        backup_service = MockFrameProcessor("backup_service")
        switcher = ServiceSwitcher(
            [error_service, backup_service],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test"), SleepFrame(sleep=0.2)],
            expected_down_frames=[TextFrame],
            expected_up_frames=[],
        )

        self.assertEqual(error_service.errors, 3)
        self.assertEqual(switcher.strategy.active_service, backup_service)
        self.assertTrue(switcher.is_usable)

    async def test_an_error_from_a_service_in_reserve_goes_no_further(self):
        """Test that a service the switcher isn't using can error unnoticed.

        It isn't being given work, so nothing about it bears on whether the
        switcher can do its job.
        """
        active_service = MockFrameProcessor("active_service")
        reserve_service = ErrorOnTextService("reserve_service", becomes_unusable=False)
        switcher = ServiceSwitcher(
            [active_service, reserve_service],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[],
        )

        await reserve_service.push_error("service connection lost")

        self.assertEqual(switcher.strategy.active_service, active_service)

    async def test_a_switcher_with_nothing_left_answers_for_every_error(self):
        """Test that a spent switcher reports its service's errors as its own.

        With nowhere to move the work, the failed service stays active and
        goes on erroring, and each of those errors is the switcher's to
        report.
        """
        error_service = RepeatedlyErroringService("error_service")
        switcher = ServiceSwitcher(
            [error_service],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        _, up_frames = await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test"), SleepFrame(sleep=0.2)],
            expected_down_frames=[TextFrame],
            expected_up_frames=[ErrorFrame, ErrorFrame, ErrorFrame],
        )

        self.assertEqual(error_service.errors, 3)
        self.assertFalse(switcher.is_usable)
        self.assertTrue(all(frame.processor is switcher for frame in up_frames))

    async def test_recoverable_error_does_not_trigger_failover(self):
        """Test that an error the service can carry on from costs no failover."""
        error_service = ErrorOnTextService("error_service", becomes_unusable=False)
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

        self.assertEqual(switcher.strategy.active_service, error_service)

    async def test_strategy_sees_every_error_from_the_active_service(self):
        """Test that the strategy is given errors the active service can carry on from.

        Which errors are worth switching away from is the strategy's decision,
        so it hears about them all, not only the ones that end a service.
        """
        seen: list[ErrorFrame] = []

        class RecordingStrategy(ServiceSwitcherStrategyManual):
            async def handle_error(self, error: ErrorFrame) -> FrameProcessor | None:
                seen.append(error)
                return None

        error_service = ErrorOnTextService("error_service", becomes_unusable=False)
        backup_service = MockFrameProcessor("backup_service")
        switcher = ServiceSwitcher(
            [error_service, backup_service],
            strategy_type=RecordingStrategy,
        )

        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].processor, error_service)
        self.assertTrue(error_service.is_usable)

    async def test_error_with_no_service_left_is_reported(self):
        """Test that running out of services is reported as the switcher's own error."""
        error_service = ErrorOnTextService("error_service")
        switcher = ServiceSwitcher(
            [error_service],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        _, up = await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].processor, switcher)
        self.assertIn("service connection lost", up[0].error)
        self.assertFalse(switcher.is_usable)

    async def test_a_lost_service_does_not_write_off_the_switcher(self):
        """Test that a switcher with a service left over is still reported as working.

        The pipeline deals with the switcher, not with the services inside it,
        so an error that costs one service must not read as the switcher being
        spent while it still has somewhere to send work. A manual strategy
        never switches on its own, which is what leaves the pair in this state.
        """
        error_service = ErrorOnTextService("error_service")
        backup_service = MockFrameProcessor("backup_service")
        switcher = ServiceSwitcher(
            [error_service, backup_service],
            strategy_type=ServiceSwitcherStrategyManual,
        )

        _, up = await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].processor, switcher)
        self.assertTrue(up[0].processor.is_usable)
        # The failing service is named, so the report still leads somewhere.
        self.assertIn("error_service", up[0].error)

    async def test_a_rejected_service_does_not_misconfigure_the_switcher(self):
        """Test that a service's rejected configuration is not read as the switcher's.

        Inheriting the category would write the switcher off for good, taking
        its remaining services with it.
        """
        error_service = ErrorOnTextService("error_service", category=ErrorCategory.AUTHENTICATION)
        backup_service = MockFrameProcessor("backup_service")
        switcher = ServiceSwitcher(
            [error_service, backup_service],
            strategy_type=ServiceSwitcherStrategyManual,
        )

        _, up = await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.UNKNOWN)
        self.assertTrue(switcher.is_usable)

    async def test_failover_skips_services_that_cannot_work(self):
        """Test that failover passes over a service that can't be given work."""
        error_service = ErrorOnTextService("error_service")
        spent_service = MockFrameProcessor("spent_service")
        backup_service = MockFrameProcessor("backup_service")
        await spent_service.set_usable(False)

        switcher = ServiceSwitcher(
            [error_service, spent_service, backup_service],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            expected_up_frames=[],
        )

        self.assertEqual(switcher.strategy.active_service, backup_service)

    async def test_switcher_is_usable_while_any_service_is(self):
        """Test that a switcher outlives the services it has lost."""
        first = MockFrameProcessor("first")
        second = MockFrameProcessor("second")
        switcher = ServiceSwitcher([first, second])

        await first.set_usable(False)
        self.assertTrue(switcher.is_usable)

        await second.set_usable(False)
        self.assertFalse(switcher.is_usable)

        # Bringing one back brings the switcher back with it.
        await second.set_usable(True)
        self.assertTrue(switcher.is_usable)

    async def test_switcher_announces_its_own_usability(self):
        """Test that a switcher reports the changes its services cause in it."""
        first = MockFrameProcessor("first")
        second = MockFrameProcessor("second")
        switcher = ServiceSwitcher([first, second])

        announced = []
        heard = asyncio.Event()

        @switcher.event_handler("on_usable_changed")
        async def on_usable_changed(switcher, is_usable):
            announced.append(is_usable)
            heard.set()

        async def wait_for_announcement():
            async with asyncio.timeout(5):
                await heard.wait()
            heard.clear()

        # Losing one service of two changes nothing the switcher can't absorb.
        await first.set_usable(False)
        # Losing the last one does.
        await second.set_usable(False)
        await wait_for_announcement()
        self.assertEqual(announced, [False])

        # And getting one back brings the switcher back with it.
        await first.set_usable(True)
        await wait_for_announcement()
        self.assertEqual(announced, [False, True])

    async def test_setting_the_switcher_usable_is_ignored(self):
        """Test that a switcher's usability can only be moved by its services.

        The switcher reports a reading of its services, so setting it directly
        would claim something the services don't say.
        """
        first = MockFrameProcessor("first")
        switcher = ServiceSwitcher([first])

        announced = []

        @switcher.event_handler("on_usable_changed")
        async def on_usable_changed(switcher, is_usable):
            announced.append(is_usable)

        messages = []
        handler_id = logger.add(messages.append, level="DEBUG", format="{message}")
        try:
            await switcher.set_usable(False)
            await asyncio.sleep(0.1)
        finally:
            logger.remove(handler_id)

        self.assertTrue(switcher.is_usable)
        self.assertEqual(announced, [])
        # Silently doing nothing would leave the caller to work that out.
        self.assertTrue(
            any("ignoring set_usable" in message for message in messages),
            f"the switcher did not report that it ignored the call: {messages}",
        )

        # And a service that can't be given work isn't overridden either.
        await first.set_usable(False)
        await switcher.set_usable(True)
        self.assertFalse(switcher.is_usable)

    async def test_manual_switch_refuses_a_service_that_cannot_work(self):
        """Test that a service that can't be given work is never made active."""
        first = MockFrameProcessor("first")
        second = MockFrameProcessor("second")
        strategy = ServiceSwitcherStrategyManual([first, second])
        await second.set_usable(False)

        switched = await strategy.handle_frame(
            ManuallySwitchServiceFrame(service=second), FrameDirection.DOWNSTREAM
        )

        self.assertIsNone(switched)
        self.assertEqual(strategy.active_service, first)

    async def test_on_service_switched_event_fires_on_error(self):
        """Test that on_service_switched event fires when an error triggers a switch."""
        strategy = ServiceSwitcherStrategyFailover(self.services)

        switched_events = []

        @strategy.event_handler("on_service_switched")
        async def on_service_switched(strategy, service):
            switched_events.append(service)

        await self.service1.set_usable(False)
        await strategy.handle_error(ErrorFrame(error="error"))
        await asyncio.sleep(0)

        self.assertEqual(len(switched_events), 1)
        self.assertEqual(switched_events[0], self.service2)


class TestServiceSwitcherSetupFailure(unittest.IsolatedAsyncioTestCase):
    """Test cases for a service that fails while the pipeline is setting up."""

    class FailsToConnectService(MockFrameProcessor):
        """A service whose connection attempt fails while it is set up."""

        async def setup(self, setup):
            await super().setup(setup)
            await asyncio.sleep(0.01)
            raise RuntimeError(f"{self.name} could not connect")

    async def test_failover_moves_off_a_service_that_cannot_be_set_up(self):
        """A service that fails to set up is one the switcher moves off.

        Services connect while the pipeline is setting up, so a service can
        fail before a single frame has been pushed. Setting up is not attempted
        again, so the service is finished rather than having a bad moment, and
        the switcher settles on the backup before the pipeline starts.
        """
        failing_service = self.FailsToConnectService("failing_service")
        backup_service = MockFrameProcessor("backup_service")
        switcher = ServiceSwitcher(
            [failing_service, backup_service],
            strategy_type=ServiceSwitcherStrategyFailover,
        )

        await run_test(
            switcher,
            frames_to_send=[TextFrame(text="test")],
            expected_down_frames=[TextFrame],
            # The switcher recovered on its own, so the error goes no further.
            expected_up_frames=[],
        )

        self.assertFalse(failing_service.is_usable)
        self.assertIs(switcher.strategy.active_service, backup_service)
        self.assertTrue(switcher.is_usable)

        # The work reached the backup, never the service that failed.
        self.assertIn(TextFrame, [type(f) for f in backup_service.processed_frames])
        self.assertNotIn(TextFrame, [type(f) for f in failing_service.processed_frames])


if __name__ == "__main__":
    unittest.main()
