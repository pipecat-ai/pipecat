#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass, field
from typing import List

from loguru import logger

from pipecat.frames.frames import (
    DataFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputTransportMessageUrgentFrame,
    StopFrame,
    SystemFrame,
    TextFrame,
    UninterruptibleFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import (
    INTERRUPTION_COMPLETION_TIMEOUT,
    FrameDirection,
    FrameProcessor,
)
from pipecat.tests.utils import SleepFrame, run_test


@dataclass
class BroadcastTestFrame(DataFrame):
    """Test frame with init fields for broadcast testing."""

    text: str = ""
    value: int = 0
    items: List[str] = field(default_factory=list)


class TestFrameProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_before_after_events(self):
        identity = IdentityFilter()

        before_process_called = False
        after_process_called = False
        before_push_called = False
        after_push_called = False

        @identity.event_handler("on_before_process_frame")
        async def on_before_process_frame(filter, frame):
            nonlocal before_process_called
            before_process_called = True

        @identity.event_handler("on_after_process_frame")
        async def on_after_process_frame(filter, frame):
            nonlocal after_process_called
            after_process_called = True

        @identity.event_handler("on_before_push_frame")
        async def on_before_push_frame(filter, frame):
            nonlocal before_push_called
            before_push_called = True

        @identity.event_handler("on_after_push_frame")
        async def on_after_push_frame(filter, frame):
            nonlocal after_push_called
            after_push_called = True

        pipeline = Pipeline([identity])

        frames_to_send = [TextFrame(text="Hello cat!")]

        expected_down_frames = [TextFrame]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert before_process_called
        assert after_process_called
        assert before_push_called
        assert after_push_called

    async def test_interruption_and_wait(self):
        class DelayFrameProcessor(FrameProcessor):
            """This processors just gives time to the event loop to change
            between tasks. Otherwise things happen to fast."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                await asyncio.sleep(0.1)
                await self.push_frame(frame, direction)

        class InterruptFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, TextFrame):
                    await self.push_interruption_task_frame_and_wait()
                    await self.push_frame(OutputTransportMessageUrgentFrame(message=frame.text))
                else:
                    await self.push_frame(frame, direction)

        pipeline = Pipeline([DelayFrameProcessor(), InterruptFrameProcessor()])

        frames_to_send = [
            # Just a random interruption to make sure we don't clear anything
            # before the actual `InterruptionTaskFrame` interruption.
            InterruptionFrame(),
            # This will generate an `InterruptionTaskFrame` and will wait for an
            # `InterruptionFrame`.
            TextFrame(text="Hello from Pipecat!"),
            # Just give time for everything to complete.
            SleepFrame(sleep=0.5),
            EndFrame(),
        ]
        expected_down_frames = [
            InterruptionFrame,
            InterruptionFrame,
            OutputTransportMessageUrgentFrame,
            EndFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            send_end_frame=False,
        )

    async def test_interruptible_frames(self):
        @dataclass
        class TestInterruptibleFrame(DataFrame):
            text: str

        class DelayTestFrameProcessor(FrameProcessor):
            """This processor just delays processing frames so we have time to
            try to interrupt them.
            """

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if not isinstance(frame, SystemFrame):
                    # Sleep more than SleepFrame default.
                    await asyncio.sleep(0.4)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([DelayTestFrameProcessor()])

        frames_to_send = [
            TestInterruptibleFrame(text="Hello from Pipecat!"),
            # Make sure we hit the DelayTestFrameProcessor first.
            SleepFrame(),
            # Just a random interruption. This should cause the interruption of
            # TestInterruptibleFrame.
            InterruptionFrame(),
        ]
        expected_down_frames = [InterruptionFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_uninterruptible_frames(self):
        @dataclass
        class TestUninterruptibleFrame(DataFrame, UninterruptibleFrame):
            text: str

        class DelayTestFrameProcessor(FrameProcessor):
            """This processor just delays processing non-InterruptionFrame so we
            have time to try to interrupt them.

            """

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if not isinstance(frame, SystemFrame):
                    # Sleep more than SleepFrame default.
                    await asyncio.sleep(0.4)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([DelayTestFrameProcessor()])

        frames_to_send = [
            TestUninterruptibleFrame(text="Hello from Pipecat!"),
            # Make sure we hit the DelayTestFrameProcessor first.
            SleepFrame(),
            # Just a random interruption. This should not cause the interruption
            # of TestUninterruptibleFrame.
            InterruptionFrame(),
        ]
        expected_down_frames = [
            InterruptionFrame,
            TestUninterruptibleFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_broadcast_frame(self):
        """Test that broadcast_frame creates two separate frames with fresh IDs."""
        downstream_frames: List[Frame] = []
        upstream_frames: List[Frame] = []

        class BroadcastTestProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.broadcast_frame(
                        BroadcastTestFrame, text="hello", value=42, items=["a", "b"]
                    )
                else:
                    await self.push_frame(frame, direction)

        class CaptureProcessor(FrameProcessor):
            def __init__(self, capture_list: List[Frame], direction: FrameDirection):
                super().__init__()
                self._capture_list = capture_list
                self._capture_direction = direction

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if direction == self._capture_direction and isinstance(frame, BroadcastTestFrame):
                    self._capture_list.append(frame)
                await self.push_frame(frame, direction)

        up_capture = CaptureProcessor(upstream_frames, FrameDirection.UPSTREAM)
        broadcaster = BroadcastTestProcessor()
        down_capture = CaptureProcessor(downstream_frames, FrameDirection.DOWNSTREAM)

        pipeline = Pipeline([up_capture, broadcaster, down_capture])

        frames_to_send = [TextFrame(text="trigger")]
        expected_down_frames = [BroadcastTestFrame]
        expected_up_frames = [BroadcastTestFrame]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify we got one frame in each direction
        self.assertEqual(len(downstream_frames), 1)
        self.assertEqual(len(upstream_frames), 1)

        down_frame = downstream_frames[0]
        up_frame = upstream_frames[0]

        # Verify the frames have different IDs (they are separate instances)
        self.assertNotEqual(down_frame.id, up_frame.id)

        # Verify the frames have the correct field values
        self.assertEqual(down_frame.text, "hello")
        self.assertEqual(down_frame.value, 42)
        self.assertEqual(down_frame.items, ["a", "b"])
        self.assertEqual(up_frame.text, "hello")
        self.assertEqual(up_frame.value, 42)
        self.assertEqual(up_frame.items, ["a", "b"])

        # Verify the items lists are shared references (no deep copy)
        self.assertIs(down_frame.items, up_frame.items)

    async def test_broadcast_frame_instance(self):
        """Test that broadcast_frame_instance shallow-copies all fields except id and name."""
        downstream_frames: List[Frame] = []
        upstream_frames: List[Frame] = []
        original_frame: List[Frame] = []

        class BroadcastInstanceTestProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, BroadcastTestFrame):
                    # Set some non-init fields on the frame
                    frame.pts = 12345
                    frame.metadata = {"key": "value", "nested": {"a": 1}}
                    original_frame.append(frame)
                    await self.broadcast_frame_instance(frame)
                else:
                    await self.push_frame(frame, direction)

        class CaptureProcessor(FrameProcessor):
            def __init__(self, capture_list: List[Frame], direction: FrameDirection):
                super().__init__()
                self._capture_list = capture_list
                self._capture_direction = direction

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if direction == self._capture_direction and isinstance(frame, BroadcastTestFrame):
                    self._capture_list.append(frame)
                await self.push_frame(frame, direction)

        up_capture = CaptureProcessor(upstream_frames, FrameDirection.UPSTREAM)
        broadcaster = BroadcastInstanceTestProcessor()
        down_capture = CaptureProcessor(downstream_frames, FrameDirection.DOWNSTREAM)

        pipeline = Pipeline([up_capture, broadcaster, down_capture])

        # Create a frame with mutable fields to test shallow copying
        test_frame = BroadcastTestFrame(text="test", value=99, items=["x", "y", "z"])

        frames_to_send = [test_frame]
        expected_down_frames = [BroadcastTestFrame]
        expected_up_frames = [BroadcastTestFrame]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )

        # Verify we got one frame in each direction
        self.assertEqual(len(downstream_frames), 1)
        self.assertEqual(len(upstream_frames), 1)
        self.assertEqual(len(original_frame), 1)

        orig = original_frame[0]
        down_frame = downstream_frames[0]
        up_frame = upstream_frames[0]

        # Verify the frames have different IDs and names (fresh values)
        self.assertNotEqual(down_frame.id, orig.id)
        self.assertNotEqual(up_frame.id, orig.id)
        self.assertNotEqual(down_frame.id, up_frame.id)
        self.assertNotEqual(down_frame.name, orig.name)
        self.assertNotEqual(up_frame.name, orig.name)

        # Verify init fields are copied correctly
        self.assertEqual(down_frame.text, "test")
        self.assertEqual(down_frame.value, 99)
        self.assertEqual(down_frame.items, ["x", "y", "z"])
        self.assertEqual(up_frame.text, "test")
        self.assertEqual(up_frame.value, 99)
        self.assertEqual(up_frame.items, ["x", "y", "z"])

        # Verify non-init fields (except id/name) are copied
        self.assertEqual(down_frame.pts, 12345)
        self.assertEqual(down_frame.metadata, {"key": "value", "nested": {"a": 1}})
        self.assertEqual(up_frame.pts, 12345)
        self.assertEqual(up_frame.metadata, {"key": "value", "nested": {"a": 1}})

        # Verify mutable fields are shallow-copied (shared references)
        self.assertIs(down_frame.items, orig.items)
        self.assertIs(up_frame.items, orig.items)
        self.assertIs(down_frame.metadata, orig.metadata)
        self.assertIs(up_frame.metadata, orig.metadata)

    async def test_terminal_frames_survive_interruption(self):
        """Test that EndFrame survives interruption (it is uninterruptible).

        This test simulates issue #3524 where an InterruptionFrame during slow
        processing would cause terminal frames to be lost, freezing the pipeline.
        """
        received_frames: List[Frame] = []

        class DelayAndInterruptProcessor(FrameProcessor):
            """This processor delays processing and then generates an interruption.

            When processing a TextFrame, it sleeps and then pushes an
            InterruptionFrame to simulate what happens when interruption occurs
            while a terminal frame is in the queue.
            """

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    # Delay to allow EndFrame to be queued
                    await asyncio.sleep(0.1)
                    # Push interruption - this should NOT discard the EndFrame
                    await self.push_frame(InterruptionFrame(), direction)
                await self.push_frame(frame, direction)

        class CaptureFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                received_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([DelayAndInterruptProcessor(), CaptureFrameProcessor()])

        frames_to_send = [
            TextFrame(text="trigger"),
        ]
        expected_down_frames = [
            InterruptionFrame,
            TextFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Verify EndFrame was received by our capture processor (survived interruption)
        # Note: run_test filters EndFrame from expected_down_frames when send_end_frame=True,
        # but our capture processor sees it before that filtering.
        end_frames = [f for f in received_frames if isinstance(f, EndFrame)]
        self.assertEqual(len(end_frames), 1, "EndFrame should survive interruption")

    async def test_stop_frame_survives_interruption(self):
        """Test that StopFrame survives interruption (it is uninterruptible).

        Similar to test_terminal_frames_survive_interruption but specifically
        for StopFrame.
        """
        received_frames: List[Frame] = []

        class DelayAndInterruptProcessor(FrameProcessor):
            """This processor delays processing and then generates an interruption."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    # Delay to allow StopFrame to be queued
                    await asyncio.sleep(0.1)
                    # Push interruption - this should NOT discard the StopFrame
                    await self.push_frame(InterruptionFrame(), direction)
                await self.push_frame(frame, direction)

        class CaptureFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                received_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([DelayAndInterruptProcessor(), CaptureFrameProcessor()])

        frames_to_send = [
            TextFrame(text="trigger"),
            StopFrame(),
        ]
        expected_down_frames = [
            InterruptionFrame,
            TextFrame,
            StopFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            send_end_frame=False,
        )

        # Verify StopFrame was received (survived interruption)
        stop_frames = [f for f in received_frames if isinstance(f, StopFrame)]
        self.assertEqual(len(stop_frames), 1, "StopFrame should survive interruption")

    async def test_interruption_frame_complete_sets_event(self):
        """Test that InterruptionFrame.complete() sets the event."""
        event = asyncio.Event()
        frame = InterruptionFrame(event=event)
        self.assertFalse(event.is_set())
        frame.complete()
        self.assertTrue(event.is_set())

    async def test_interruption_frame_complete_without_event(self):
        """Test that InterruptionFrame.complete() is safe without an event."""
        frame = InterruptionFrame()
        frame.complete()  # Should not raise

    async def test_interruption_event_set_at_pipeline_sink(self):
        """Test that the event from push_interruption_task_frame_and_wait()
        is set when the InterruptionFrame reaches the pipeline sink."""
        event_was_set = False

        class InterruptOnTextProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                nonlocal event_was_set

                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.push_interruption_task_frame_and_wait()

                    event_was_set = True
                    await self.push_frame(OutputTransportMessageUrgentFrame(message="done"))
                else:
                    await self.push_frame(frame, direction)

        pipeline = Pipeline([InterruptOnTextProcessor()])

        frames_to_send = [
            TextFrame(text="trigger"),
        ]
        expected_down_frames = [
            InterruptionFrame,
            OutputTransportMessageUrgentFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        self.assertTrue(event_was_set, "Event should be set after InterruptionFrame completes")

    async def test_interruption_completion_timeout_warning(self):
        """Test that a warning is logged when an InterruptionFrame is blocked
        and never reaches the pipeline sink."""
        warnings = []
        handler_id = logger.add(
            lambda msg: warnings.append(str(msg)), level="WARNING", format="{message}"
        )

        try:

            class BlockInterruptionProcessor(FrameProcessor):
                """Blocks InterruptionFrames, completing them after a delay."""

                async def process_frame(self, frame: Frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)
                    if isinstance(frame, InterruptionFrame):
                        # Complete after the timeout so the warning fires
                        # but the test doesn't hang.
                        async def delayed_complete():
                            await asyncio.sleep(INTERRUPTION_COMPLETION_TIMEOUT + 1.0)
                            frame.complete()

                        asyncio.create_task(delayed_complete())
                        return
                    await self.push_frame(frame, direction)

            class InterruptOnTextProcessor(FrameProcessor):
                async def process_frame(self, frame: Frame, direction: FrameDirection):
                    await super().process_frame(frame, direction)
                    if isinstance(frame, TextFrame):
                        await self.push_interruption_task_frame_and_wait()
                        await self.push_frame(OutputTransportMessageUrgentFrame(message="done"))
                    else:
                        await self.push_frame(frame, direction)

            pipeline = Pipeline([BlockInterruptionProcessor(), InterruptOnTextProcessor()])

            frames_to_send = [
                TextFrame(text="trigger"),
            ]
            expected_down_frames = [
                OutputTransportMessageUrgentFrame,
            ]
            await run_test(
                pipeline,
                frames_to_send=frames_to_send,
                expected_down_frames=expected_down_frames,
            )
        finally:
            logger.remove(handler_id)

        self.assertTrue(
            any("InterruptionFrame has not completed" in w for w in warnings),
            "Expected a timeout warning about InterruptionFrame not completing",
        )


if __name__ == "__main__":
    unittest.main()
