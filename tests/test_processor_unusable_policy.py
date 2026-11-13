#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
import warnings

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FatalErrorFrame,
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


class PermanentlyFailingProcessor(FrameProcessor):
    """Processor whose failure keeps recurring, whatever its category says."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_error(
                "service failed for good",
                category=ErrorCategory.SERVER,
                force_treat_as_permanent=True,
            )

        await self.push_frame(frame, direction)


class FatalErroringProcessor(FrameProcessor):
    """Processor reporting errors through the deprecated ``fatal`` flag."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_error("service failed", fatal=True)

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

    async def test_permanent_errors_are_acted_on_whatever_the_category(self):
        unusable, finished = await self.run_worker(
            PermanentlyFailingProcessor(),
            ProcessorUnusablePolicy.CANCEL,
            [TextFrame("hello")],
        )

        self.assertEqual(len(unusable), 1)
        self.assertTrue(any(isinstance(frame, CancelFrame) for frame in finished))


class TestDeprecatedFatalFlag(UnusableProcessorTestCase):
    def assert_warns_fatal(self, warnings_raised, subject: str):
        """Assert the deprecation was reported once, naming both replacements."""
        self.assertEqual(len(warnings_raised), 1)
        message = str(warnings_raised[0].message)
        self.assertIs(warnings_raised[0].category, DeprecationWarning)
        self.assertIn(subject, message)
        self.assertIn("force_treat_as_permanent=True", message)
        self.assertIn("EndWorkerFrame", message)

    def test_error_frame_warns_when_fatal(self):
        with warnings.catch_warnings(record=True) as raised:
            ErrorFrame("service failed", fatal=True)

        self.assert_warns_fatal(raised, "`ErrorFrame.fatal`")

    def test_error_frame_stays_quiet_without_fatal(self):
        with warnings.catch_warnings(record=True) as raised:
            ErrorFrame("service failed")
            ErrorFrame("service failed", fatal=False)

        self.assertEqual(raised, [])

    def test_fatal_error_frame_warns_about_itself_only(self):
        with warnings.catch_warnings(record=True) as raised:
            FatalErrorFrame("service failed")

        self.assertEqual(len(raised), 1)
        self.assertIn("`FatalErrorFrame` is deprecated", str(raised[0].message))

    async def test_push_error_warns_when_fatal_and_still_cancels(self):
        """The flag keeps cancelling until it goes away, whatever the policy."""
        with warnings.catch_warnings(record=True) as raised:
            unusable, finished = await self.run_worker(
                FatalErroringProcessor(),
                ProcessorUnusablePolicy.CONTINUE,
                [TextFrame("hello")],
            )

        self.assert_warns_fatal(raised, "`push_error(fatal=True)`")
        self.assertEqual(unusable, [])
        self.assertTrue(any(isinstance(frame, CancelFrame) for frame in finished))

    async def test_push_error_stays_quiet_when_permanent(self):
        with warnings.catch_warnings(record=True) as raised:
            await self.run_worker(
                PermanentlyFailingProcessor(),
                ProcessorUnusablePolicy.CANCEL,
                [TextFrame("hello")],
            )

        self.assertEqual(raised, [])
