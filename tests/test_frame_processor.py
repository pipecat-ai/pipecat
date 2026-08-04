#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    CancelFrame,
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
    FrameDirection,
    FrameProcessor,
)
from pipecat.tests.utils import SleepFrame, run_test


@dataclass
class BroadcastTestFrame(DataFrame):
    """Test frame with init fields for broadcast testing."""

    text: str = ""
    value: int = 0
    items: list[str] = field(default_factory=list)


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

    async def test_broadcast_interruption(self):
        """Test that broadcast_interruption() pushes InterruptionFrame both
        directions and allows subsequent code to run."""

        class InterruptFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, TextFrame):
                    await self.broadcast_interruption()
                    await self.push_frame(OutputTransportMessageUrgentFrame(message=frame.text))
                else:
                    await self.push_frame(frame, direction)

        pipeline = Pipeline([InterruptFrameProcessor()])

        frames_to_send = [
            TextFrame(text="Hello from Pipecat!"),
            SleepFrame(sleep=0.5),
        ]
        expected_down_frames = [
            InterruptionFrame,
            OutputTransportMessageUrgentFrame,
        ]
        expected_up_frames = [
            InterruptionFrame,
        ]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
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
        downstream_frames: list[Frame] = []
        upstream_frames: list[Frame] = []

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
            def __init__(self, capture_list: list[Frame], direction: FrameDirection):
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
        downstream_frames: list[Frame] = []
        upstream_frames: list[Frame] = []
        original_frame: list[Frame] = []

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
            def __init__(self, capture_list: list[Frame], direction: FrameDirection):
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
        received_frames: list[Frame] = []

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
        received_frames: list[Frame] = []

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

    async def test_broadcast_interruption_allows_subsequent_code(self):
        """Test that broadcast_interruption() returns immediately, allowing the
        caller to run code afterwards (e.g. push an urgent frame)."""
        code_after_ran = False

        class InterruptOnTextProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                nonlocal code_after_ran

                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.broadcast_interruption()

                    code_after_ran = True
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
        self.assertTrue(code_after_ran, "Code after broadcast_interruption() should execute")

    async def test_lifecycle_endframe_bypasses_processing_pause(self):
        """EndFrame must bypass a processing pause so teardown is not delayed.

        A processor that has called pause_processing_frames() (e.g. a TTS
        service pausing to avoid audio overlap) must still process EndFrame
        immediately. Otherwise, when an upstream ParallelPipeline's EndFrame
        synchronization buffers frames until all branches deliver EndFrame, an
        EndFrame stuck behind the pause never reaches the sync sink and the
        pipeline deadlocks (or stalls until a watchdog force-resumes).
        """
        received_frames: list[Frame] = []

        class PauseOnTextProcessor(FrameProcessor):
            """Pauses frame processing when it sees a TextFrame trigger."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.pause_processing_frames()
                await self.push_frame(frame, direction)

        class CaptureFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                received_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([PauseOnTextProcessor(), CaptureFrameProcessor()])

        frames_to_send = [
            TextFrame(text="trigger_pause"),
            # Give the pause time to latch before EndFrame arrives.
            SleepFrame(sleep=0.05),
            # EndFrame arrives while processing is paused; it must bypass the
            # pause and reach the capture processor immediately.
            EndFrame(),
        ]

        # If EndFrame does NOT bypass the pause, run_test will hang waiting for
        # the pipeline to finish (EndFrame never reaches the sink). The
        # asyncio.wait_for timeout catches that. We skip expected_down_frames
        # validation because EndFrame is not filtered when send_end_frame=False.
        await asyncio.wait_for(
            run_test(
                pipeline,
                frames_to_send=frames_to_send,
                send_end_frame=False,
            ),
            timeout=3.0,
        )

        end_frames = [f for f in received_frames if isinstance(f, EndFrame)]
        self.assertEqual(len(end_frames), 1, "EndFrame should bypass the processing pause")

    async def test_regular_dataframe_respects_processing_pause(self):
        """A regular DataFrame must remain gated behind a processing pause.

        This is the counterpart to the lifecycle-frame bypass: non-lifecycle
        frames must still honor pause_processing_frames() and only flow once
        resume_processing_frames() is called.
        """
        received_frames: list[Frame] = []
        resumed = asyncio.Event()

        class PauseAndResumeProcessor(FrameProcessor):
            """Pauses on a trigger, then resumes after a short delay."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.pause_processing_frames()

                    # Resume shortly after pausing so the test can complete.
                    async def resume_later():
                        await asyncio.sleep(0.1)
                        await self.resume_processing_frames()
                        resumed.set()

                    self.create_task(resume_later())
                await self.push_frame(frame, direction)

        class CaptureFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                received_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([PauseAndResumeProcessor(), CaptureFrameProcessor()])

        # MarkerDataFrame is a regular (non-lifecycle) DataFrame; it must NOT
        # arrive at the capture processor until after resume.
        @dataclass
        class MarkerDataFrame(DataFrame):
            label: str = ""

        frames_to_send = [
            TextFrame(text="trigger_pause"),
            # Give the pause time to latch before the marker frame is queued.
            SleepFrame(sleep=0.05),
            MarkerDataFrame(label="gated"),
        ]

        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=[TextFrame, MarkerDataFrame],
        )

        # The marker frame must have arrived — but only after resume.
        self.assertTrue(resumed.is_set(), "resume_processing_frames should have been called")
        marker_frames = [f for f in received_frames if isinstance(f, MarkerDataFrame)]
        self.assertEqual(len(marker_frames), 1, "MarkerDataFrame should arrive after resume")

    async def test_cancelframe_bypasses_processing_pause(self):
        """CancelFrame must bypass the frame processing pause.

        CancelFrame is a SystemFrame, so it is handled by the input frame task
        handler and gated by the system-frame pause. Like EndFrame, it must
        bypass that pause so cancellation can flow through promptly during
        teardown — a ParallelPipeline's CancelFrame synchronization has the
        same buffering dependency as its EndFrame synchronization.
        """
        received_frames: list[Frame] = []

        class PauseSystemOnTextProcessor(FrameProcessor):
            """Pauses system-frame processing when it sees a TextFrame trigger."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    await self.pause_processing_system_frames()
                await self.push_frame(frame, direction)

        class CaptureFrameProcessor(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                received_frames.append(frame)
                await self.push_frame(frame, direction)

        pipeline = Pipeline([PauseSystemOnTextProcessor(), CaptureFrameProcessor()])

        frames_to_send = [
            TextFrame(text="trigger_pause"),
            # Give the pause time to latch before CancelFrame arrives.
            SleepFrame(sleep=0.05),
            # CancelFrame is a SystemFrame; it must bypass the system-frame
            # pause and reach the capture processor immediately.
            CancelFrame(),
        ]

        # If CancelFrame does NOT bypass the pause, the pipeline will hang
        # (CancelFrame never reaches the sink, so the worker's cancel wait
        # never completes). We pass expected_down_frames=None because CancelFrame
        # is a high-priority SystemFrame that may reach the sink before the
        # regular TextFrame is processed, so ordering is not guaranteed.
        await asyncio.wait_for(
            run_test(
                pipeline,
                frames_to_send=frames_to_send,
                send_end_frame=False,
            ),
            timeout=3.0,
        )

        cancel_frames = [f for f in received_frames if isinstance(f, CancelFrame)]
        self.assertEqual(
            len(cancel_frames), 1, "CancelFrame should bypass the system-frame processing pause"
        )


if __name__ == "__main__":
    unittest.main()
