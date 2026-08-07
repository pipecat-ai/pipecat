#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy, WorkerParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory

RUN_TIMEOUT_SECS = 10


class ErroringProcessor(FrameProcessor):
    """Processor that reports an error every time it sees a `TextFrame`."""

    def __init__(self, category: ErrorCategory = ErrorCategory.AUTHENTICATION, **kwargs):
        super().__init__(**kwargs)
        self._category = category

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_error_frame(ErrorFrame("service failed", category=self._category))

        # Forward everything, so lifecycle frames reach the sink and every
        # processor in the pipeline gets a chance to report its own error.
        await self.push_frame(frame, direction)


class UnusableProcessorTestCase(unittest.IsolatedAsyncioTestCase):
    def collect_unusable(self, worker: PipelineWorker) -> list[ErrorFrame]:
        """Collect the errors that cost their processor its usefulness.

        This is how application code tells the two kinds of error apart, with
        no dedicated event of its own.
        """
        unusable: list[ErrorFrame] = []

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame):
            if frame.processor and not frame.processor.is_usable:
                unusable.append(frame)

        return unusable

    async def run_worker(
        self,
        processor: FrameProcessor,
        policy: ProcessorUnusablePolicy,
        frames: list[Frame],
    ) -> tuple[list[ErrorFrame], list[Frame]]:
        """Run a one-processor pipeline, returning its errors and terminal frames."""
        worker = PipelineWorker(Pipeline([processor]), processor_unusable_policy=policy)

        unusable = self.collect_unusable(worker)
        finished: list[Frame] = []

        @worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(worker, frame):
            finished.append(frame)

        await worker.queue_frames(frames)

        async with asyncio.timeout(RUN_TIMEOUT_SECS):
            await worker.run(WorkerParams(task_manager=TaskManager()))

        return unusable, finished


class TestProcessorUnusablePolicy(UnusableProcessorTestCase):
    async def test_continue_keeps_the_pipeline_running(self):
        unusable, finished = await self.run_worker(
            ErroringProcessor(),
            ProcessorUnusablePolicy.CONTINUE,
            [TextFrame("hello"), EndFrame()],
        )

        self.assertEqual(len(unusable), 1)
        self.assertEqual(unusable[0].category, ErrorCategory.AUTHENTICATION)
        # The pipeline ran until the EndFrame we queued, not because of the error.
        self.assertTrue(any(isinstance(frame, EndFrame) for frame in finished))

    async def test_end_stops_the_pipeline(self):
        unusable, finished = await self.run_worker(
            ErroringProcessor(),
            ProcessorUnusablePolicy.END,
            [TextFrame("hello")],
        )

        self.assertEqual(len(unusable), 1)
        self.assertTrue(any(isinstance(frame, EndFrame) for frame in finished))

    async def test_cancel_stops_the_pipeline(self):
        unusable, finished = await self.run_worker(
            ErroringProcessor(),
            ProcessorUnusablePolicy.CANCEL,
            [TextFrame("hello")],
        )

        self.assertEqual(len(unusable), 1)
        self.assertTrue(any(isinstance(frame, CancelFrame) for frame in finished))

    async def test_default_policy_is_continue(self):
        worker = PipelineWorker(Pipeline([]))
        self.assertEqual(worker._processor_unusable_policy, ProcessorUnusablePolicy.CONTINUE)


class TestErrorsThePipelineActsOn(UnusableProcessorTestCase):
    async def test_a_processor_is_acted_on_once(self):
        processor = ErroringProcessor()
        worker = PipelineWorker(
            Pipeline([processor]), processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE
        )
        self.collect_unusable(worker)

        await worker.queue_frames(
            [TextFrame("one"), TextFrame("two"), TextFrame("three"), EndFrame()]
        )

        async with asyncio.timeout(RUN_TIMEOUT_SECS):
            await worker.run(WorkerParams(task_manager=TaskManager()))

        # Every error is reported, but the policy is applied to the first.
        self.assertEqual(worker._unusable_processors, {processor})

    async def test_each_processor_is_acted_on_separately(self):
        first = ErroringProcessor(name="first")
        second = ErroringProcessor(name="second")
        worker = PipelineWorker(
            Pipeline([first, second]),
            processor_unusable_policy=ProcessorUnusablePolicy.CONTINUE,
        )

        # The TextFrame reaches both processors, so both report an error.
        await worker.queue_frames([TextFrame("hello"), EndFrame()])

        async with asyncio.timeout(RUN_TIMEOUT_SECS):
            await worker.run(WorkerParams(task_manager=TaskManager()))

        self.assertEqual(worker._unusable_processors, {first, second})

    async def test_transient_errors_leave_the_pipeline_running(self):
        unusable, finished = await self.run_worker(
            ErroringProcessor(category=ErrorCategory.SERVER),
            ProcessorUnusablePolicy.END,
            [TextFrame("hello"), EndFrame()],
        )

        self.assertEqual(unusable, [])
        self.assertTrue(any(isinstance(frame, EndFrame) for frame in finished))

    async def test_unclassified_errors_leave_the_pipeline_running(self):
        unusable, _ = await self.run_worker(
            ErroringProcessor(category=ErrorCategory.UNKNOWN),
            ProcessorUnusablePolicy.END,
            [TextFrame("hello"), EndFrame()],
        )

        self.assertEqual(unusable, [])
