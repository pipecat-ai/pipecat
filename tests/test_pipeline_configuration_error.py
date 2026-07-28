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
from pipecat.pipeline.worker import ConfigurationErrorPolicy, PipelineWorker, WorkerParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory

RUN_TIMEOUT_SECS = 10


class ErroringProcessor(FrameProcessor):
    """Processor that reports an error every time it sees a `TextFrame`."""

    def __init__(
        self,
        category: ErrorCategory = ErrorCategory.AUTHENTICATION,
        handled: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._category = category
        self._handled = handled

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            await self.push_error_frame(
                ErrorFrame("service failed", category=self._category, handled=self._handled)
            )

        # Forward everything, so lifecycle frames reach the sink and every
        # processor in the pipeline gets a chance to report its own error.
        await self.push_frame(frame, direction)


class ConfigurationErrorTestCase(unittest.IsolatedAsyncioTestCase):
    async def run_worker(
        self,
        processor: FrameProcessor,
        policy: ConfigurationErrorPolicy,
        frames: list[Frame],
    ) -> tuple[list[ErrorFrame], list[Frame]]:
        """Run a one-processor pipeline, returning reported errors and terminal frames."""
        worker = PipelineWorker(Pipeline([processor]), on_configuration_error=policy)

        reported: list[ErrorFrame] = []
        finished: list[Frame] = []

        @worker.event_handler("on_pipeline_configuration_error")
        async def on_configuration_error(worker, frame):
            reported.append(frame)

        @worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(worker, frame):
            finished.append(frame)

        await worker.queue_frames(frames)

        async with asyncio.timeout(RUN_TIMEOUT_SECS):
            await worker.run(WorkerParams(task_manager=TaskManager()))

        return reported, finished


class TestConfigurationErrorPolicy(ConfigurationErrorTestCase):
    async def test_continue_keeps_the_pipeline_running(self):
        reported, finished = await self.run_worker(
            ErroringProcessor(),
            ConfigurationErrorPolicy.CONTINUE,
            [TextFrame("hello"), EndFrame()],
        )

        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0].category, ErrorCategory.AUTHENTICATION)
        # The pipeline ran until the EndFrame we queued, not because of the error.
        self.assertTrue(any(isinstance(frame, EndFrame) for frame in finished))

    async def test_end_stops_the_pipeline(self):
        reported, finished = await self.run_worker(
            ErroringProcessor(),
            ConfigurationErrorPolicy.END,
            [TextFrame("hello")],
        )

        self.assertEqual(len(reported), 1)
        self.assertTrue(any(isinstance(frame, EndFrame) for frame in finished))

    async def test_cancel_stops_the_pipeline(self):
        reported, finished = await self.run_worker(
            ErroringProcessor(),
            ConfigurationErrorPolicy.CANCEL,
            [TextFrame("hello")],
        )

        self.assertEqual(len(reported), 1)
        self.assertTrue(any(isinstance(frame, CancelFrame) for frame in finished))

    async def test_default_policy_is_continue(self):
        worker = PipelineWorker(Pipeline([]))
        self.assertEqual(worker._on_configuration_error, ConfigurationErrorPolicy.CONTINUE)


class TestConfigurationErrorReporting(ConfigurationErrorTestCase):
    async def test_repeated_errors_are_reported_once(self):
        reported, _ = await self.run_worker(
            ErroringProcessor(),
            ConfigurationErrorPolicy.CONTINUE,
            [TextFrame("one"), TextFrame("two"), TextFrame("three"), EndFrame()],
        )

        self.assertEqual(len(reported), 1)

    async def test_each_service_is_reported_separately(self):
        first = ErroringProcessor(name="first")
        second = ErroringProcessor(name="second")
        worker = PipelineWorker(
            Pipeline([first, second]), on_configuration_error=ConfigurationErrorPolicy.CONTINUE
        )

        reported: list[ErrorFrame] = []

        @worker.event_handler("on_pipeline_configuration_error")
        async def on_configuration_error(worker, frame):
            reported.append(frame)

        # The TextFrame reaches both processors, so both report an error.
        await worker.queue_frames([TextFrame("hello"), EndFrame()])

        async with asyncio.timeout(RUN_TIMEOUT_SECS):
            await worker.run(WorkerParams(task_manager=TaskManager()))

        self.assertEqual({frame.processor for frame in reported}, {first, second})

    async def test_transient_errors_are_not_configuration_errors(self):
        reported, finished = await self.run_worker(
            ErroringProcessor(category=ErrorCategory.SERVER),
            ConfigurationErrorPolicy.END,
            [TextFrame("hello"), EndFrame()],
        )

        self.assertEqual(reported, [])
        self.assertTrue(any(isinstance(frame, EndFrame) for frame in finished))

    async def test_unclassified_errors_are_not_configuration_errors(self):
        reported, _ = await self.run_worker(
            ErroringProcessor(category=ErrorCategory.UNKNOWN),
            ConfigurationErrorPolicy.END,
            [TextFrame("hello"), EndFrame()],
        )

        self.assertEqual(reported, [])

    async def test_handled_errors_are_left_alone(self):
        reported, _ = await self.run_worker(
            ErroringProcessor(handled=True),
            ConfigurationErrorPolicy.END,
            [TextFrame("hello"), EndFrame()],
        )

        self.assertEqual(reported, [])
