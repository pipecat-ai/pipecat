#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import EndFrame, HeartbeatFrame, StartFrame, TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import HeartbeatsObserver, run_test


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_single(self):
        pipeline = Pipeline([IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

    async def test_pipeline_multiple(self):
        identity1 = IdentityFilter()
        identity2 = IdentityFilter()
        identity3 = IdentityFilter()

        pipeline = Pipeline([identity1, identity2, identity3])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

    async def test_pipeline_start_metadata(self):
        pipeline = Pipeline([IdentityFilter()])

        frames_to_send = []
        expected_returned_frames = [StartFrame]
        (received_down, _) = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
            ignore_start=False,
            start_metadata={"foo": "bar"},
        )
        assert "foo" in received_down[-1].metadata


class TestParallelPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_single(self):
        pipeline = ParallelPipeline([IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

    async def test_parallel_multiple(self):
        """Should only passthrough one instance of TextFrame."""
        pipeline = ParallelPipeline([IdentityFilter()], [IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_returned_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )


class TestPipelineTask(unittest.IsolatedAsyncioTestCase):
    async def test_task_single(self):
        pipeline = Pipeline([IdentityFilter()])
        task = PipelineTask(pipeline)
        task.set_event_loop(asyncio.get_event_loop())

        await task.queue_frame(TextFrame(text="Hello!"))
        await task.queue_frames([TextFrame(text="Bye!"), EndFrame()])
        await task.run()
        assert task.has_finished()

    async def test_task_heartbeats(self):
        heartbeats_counter = 0

        async def heartbeat_received(processor: FrameProcessor, heartbeat: HeartbeatFrame):
            nonlocal heartbeats_counter
            heartbeats_counter += 1

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        heartbeats_observer = HeartbeatsObserver(
            target=identity, heartbeat_callback=heartbeat_received
        )
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_heartbeats=True,
                heartbeats_period_secs=0.2,
                observers=[heartbeats_observer],
            ),
        )
        task.set_event_loop(asyncio.get_event_loop())

        expected_heartbeats = 1.0 / 0.2

        await task.queue_frame(TextFrame(text="Hello!"))
        try:
            await asyncio.wait_for(task.run(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        assert heartbeats_counter == expected_heartbeats
