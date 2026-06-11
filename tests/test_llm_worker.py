#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from unittest.mock import MagicMock

from pipecat.frames.frames import LLMMessagesAppendFrame, PipelineFlushFrame
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.workers.llm import LLMWorker, tool


def _create_task():
    """Create a StubLLMTask with mocked parent queue_frame for testing."""

    class StubLLMTask(LLMWorker):
        @tool
        async def fast_tool(self, params):
            """A quick tool."""
            await params.result_callback("done")

        @tool
        async def slow_tool(self, params, delay: float):
            """A tool that blocks on an event for coordination."""
            await params.result_callback("done")

    llm = MagicMock()
    llm._register_direct_function = MagicMock()
    task = StubLLMTask("test_task", llm=llm, bridged=())

    # Capture frames passed to PipelineWorker.queue_frame (i.e. super().queue_frame).
    delivered: list[tuple] = []
    original_pt_queue_frame = PipelineWorker.queue_frame

    async def class_replacement(self, frame, direction=FrameDirection.DOWNSTREAM):
        # Only intercept for this specific instance; otherwise fall through.
        if self is task:
            delivered.append((frame, direction))
            return
        await original_pt_queue_frame(self, frame, direction)

    PipelineWorker.queue_frame = class_replacement

    # flush_pipeline() does a real round-trip through the pipeline source/sink,
    # which this stubbed task has no running pipeline to service. Complete it
    # instantly so the deferral logic under test isn't blocked on the probe.
    async def _instant_flush(timeout: float = 5.0) -> bool:
        return True

    task.flush_pipeline = _instant_flush
    task._restore_pt_queue_frame = lambda: setattr(
        PipelineWorker, "queue_frame", original_pt_queue_frame
    )

    task._delivered_frames = delivered
    return task


def _get_delivered_frames(task):
    """Extract non-flush frames delivered to the underlying pipeline."""
    return [
        (frame, direction)
        for frame, direction in task._delivered_frames
        if not isinstance(frame, PipelineFlushFrame)
    ]


def _make_frame(content: str, run_llm: bool = True) -> LLMMessagesAppendFrame:
    return LLMMessagesAppendFrame(messages=[{"role": "user", "content": content}], run_llm=run_llm)


class TestToolCallTracking(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tasks = []

    def _track(self, task):
        self._tasks.append(task)
        return task

    def tearDown(self):
        for task in self._tasks:
            restore = getattr(task, "_restore_pt_queue_frame", None)
            if restore:
                restore()

    async def test_tool_call_active_initially_false(self):
        task = self._track(_create_task())
        self.assertFalse(task.tool_call_active)

    async def test_tool_call_active_during_execution(self):
        """tool_call_active is True while a tool is running."""
        task = self._track(_create_task())
        observed = []

        @tool
        async def gated_tool(self, params):
            """Waits on gate."""
            observed.append(task.tool_call_active)

        wrapped = task._track_tool_call(gated_tool.__get__(task))
        params = MagicMock()
        await wrapped(params)

        self.assertTrue(observed[0])
        self.assertFalse(task.tool_call_active)

    async def test_queue_frame_delivers_immediately_when_idle(self):
        """queue_frame delivers immediately when no tools are in-flight."""
        task = self._track(_create_task())
        frame = _make_frame("hello")

        await task.queue_frame(frame)

        delivered = _get_delivered_frames(task)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)

    async def test_queue_frame_defers_when_tool_active(self):
        """queue_frame defers delivery when a tool is in-flight."""
        task = self._track(_create_task())
        task._tool_call_inflight = 1

        frame = _make_frame("deferred")
        await task.queue_frame(frame)

        delivered = _get_delivered_frames(task)
        self.assertEqual(len(delivered), 0)
        self.assertEqual(len(task._deferred_frames), 1)
        self.assertEqual(task._deferred_frames[0], (frame, FrameDirection.DOWNSTREAM))

    async def test_deferred_frames_flush_when_tool_completes(self):
        """Frames deferred during a tool call are delivered when it finishes."""
        task = self._track(_create_task())
        gate = asyncio.Event()
        frame = _make_frame("event data")

        @tool
        async def blocking_tool(self, params):
            """Blocks until gate is set."""
            await gate.wait()

        wrapped = task._track_tool_call(blocking_tool.__get__(task))
        params = MagicMock()

        runner_task = asyncio.create_task(wrapped(params))
        await asyncio.sleep(0)

        await task.queue_frame(frame)
        self.assertEqual(len(_get_delivered_frames(task)), 0)

        gate.set()
        await runner_task

        delivered = _get_delivered_frames(task)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)

    async def test_concurrent_tools_flush_only_when_all_done(self):
        """With two parallel tools, flush happens only when the last one completes."""
        task = self._track(_create_task())
        gate_a = asyncio.Event()
        gate_b = asyncio.Event()

        @tool
        async def tool_a(self, params):
            """First tool."""
            await gate_a.wait()

        @tool
        async def tool_b(self, params):
            """Second tool."""
            await gate_b.wait()

        wrapped_a = task._track_tool_call(tool_a.__get__(task))
        wrapped_b = task._track_tool_call(tool_b.__get__(task))
        params = MagicMock()

        task_a = asyncio.create_task(wrapped_a(params))
        task_b = asyncio.create_task(wrapped_b(params))
        await asyncio.sleep(0)

        self.assertEqual(task._tool_call_inflight, 2)

        frame = _make_frame("queued")
        await task.queue_frame(frame)

        # First tool finishes — frame still deferred (second tool running)
        gate_a.set()
        await task_a
        self.assertEqual(task._tool_call_inflight, 1)
        self.assertEqual(len(_get_delivered_frames(task)), 0)

        # Second tool finishes — NOW flush
        gate_b.set()
        await task_b
        self.assertEqual(task._tool_call_inflight, 0)

        delivered = _get_delivered_frames(task)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)

    async def test_queue_frame_preserves_frame_attributes(self):
        """Frame attributes like run_llm are preserved through defer and flush."""
        task = self._track(_create_task())
        gate = asyncio.Event()

        @tool
        async def blocking_tool(self, params):
            """Blocks."""
            await gate.wait()

        wrapped = task._track_tool_call(blocking_tool.__get__(task))
        params = MagicMock()

        runner_task = asyncio.create_task(wrapped(params))
        await asyncio.sleep(0)

        frame = _make_frame("no inference", run_llm=False)
        await task.queue_frame(frame)

        gate.set()
        await runner_task

        delivered = _get_delivered_frames(task)
        self.assertEqual(len(delivered), 1)
        self.assertFalse(delivered[0][0].run_llm)

    async def test_multiple_deferred_frames_flush_in_order(self):
        """Multiple deferred frames are delivered in FIFO order."""
        task = self._track(_create_task())
        gate = asyncio.Event()

        @tool
        async def blocking_tool(self, params):
            """Blocks."""
            await gate.wait()

        wrapped = task._track_tool_call(blocking_tool.__get__(task))
        params = MagicMock()

        runner_task = asyncio.create_task(wrapped(params))
        await asyncio.sleep(0)

        frame_a = _make_frame("first", run_llm=False)
        frame_b = _make_frame("second", run_llm=True)
        await task.queue_frame(frame_a)
        await task.queue_frame(frame_b)

        gate.set()
        await runner_task

        delivered = _get_delivered_frames(task)
        self.assertEqual(len(delivered), 2)
        self.assertIs(delivered[0][0], frame_a)
        self.assertIs(delivered[1][0], frame_b)

    async def test_tool_error_still_decrements_and_flushes(self):
        """If a tool raises, the counter still decrements and deferred frames flush."""
        task = self._track(_create_task())

        @tool
        async def failing_tool(self, params):
            """Always fails."""
            raise ValueError("boom")

        wrapped = task._track_tool_call(failing_tool.__get__(task))
        params = MagicMock()

        frame = _make_frame("recover")
        task._tool_call_inflight = 1
        await task.queue_frame(frame)
        task._tool_call_inflight = 0

        with self.assertRaises(ValueError):
            await wrapped(params)

        self.assertFalse(task.tool_call_active)
        delivered = _get_delivered_frames(task)
        self.assertEqual(len(delivered), 1)
        self.assertIs(delivered[0][0], frame)


if __name__ == "__main__":
    unittest.main()
