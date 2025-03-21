#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
import unittest

from pipecat.frames.frames import EndFrame, HeartbeatFrame, StartFrame, TextFrame
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import HeartbeatsObserver, run_test


class TestPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_single(self):
        pipeline = Pipeline([IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_pipeline_multiple(self):
        identity1 = IdentityFilter()
        identity2 = IdentityFilter()
        identity3 = IdentityFilter()

        pipeline = Pipeline([identity1, identity2, identity3])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_pipeline_start_metadata(self):
        pipeline = Pipeline([IdentityFilter()])

        frames_to_send = []
        expected_down_frames = [StartFrame]
        (received_down, _) = await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            ignore_start=False,
            start_metadata={"foo": "bar"},
        )
        assert "foo" in received_down[-1].metadata


class TestParallelPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_single(self):
        pipeline = ParallelPipeline([IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_parallel_multiple(self):
        """Should only passthrough one instance of TextFrame."""
        pipeline = ParallelPipeline([IdentityFilter()], [IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello from Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
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

    async def test_task_event_handlers(self):
        upstream_received = False
        downstream_received = False

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)
        task.set_event_loop(asyncio.get_event_loop())
        task.set_reached_upstream_filter((TextFrame,))
        task.set_reached_downstream_filter((TextFrame,))

        @task.event_handler("on_frame_reached_upstream")
        async def on_frame_reached_upstream(task, frame):
            nonlocal upstream_received
            if isinstance(frame, TextFrame) and frame.text == "Hello Upstream!":
                upstream_received = True

        @task.event_handler("on_frame_reached_downstream")
        async def on_frame_reached_downstream(task, frame):
            nonlocal downstream_received
            if isinstance(frame, TextFrame) and frame.text == "Hello Downstream!":
                downstream_received = True
                await identity.push_frame(
                    TextFrame(text="Hello Upstream!"), FrameDirection.UPSTREAM
                )

        await task.queue_frame(TextFrame(text="Hello Downstream!"))

        try:
            await asyncio.wait_for(asyncio.shield(task.run()), timeout=1.0)
        except asyncio.TimeoutError:
            pass

        assert upstream_received
        assert downstream_received

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
            ),
            observers=[heartbeats_observer],
            cancel_on_idle_timeout=False,
        )
        task.set_event_loop(asyncio.get_event_loop())

        expected_heartbeats = 1.0 / 0.2

        await task.queue_frame(TextFrame(text="Hello!"))
        try:
            await asyncio.wait_for(asyncio.shield(task.run()), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        assert heartbeats_counter == expected_heartbeats

    async def test_idle_task(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, idle_timeout_secs=0.2)
        task.set_event_loop(asyncio.get_event_loop())
        await task.run()
        assert True

    async def test_no_idle_task(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, idle_timeout_secs=0.2, cancel_on_idle_timeout=False)
        task.set_event_loop(asyncio.get_event_loop())
        try:
            await asyncio.wait_for(asyncio.shield(task.run()), timeout=0.3)
        except asyncio.TimeoutError:
            assert True
        else:
            assert False

    async def test_idle_task_heartbeats(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_heartbeats=True,
                heartbeats_period_secs=0.1,
            ),
            idle_timeout_secs=0.3,
        )
        task.set_event_loop(asyncio.get_event_loop())
        await task.run()
        assert True

    async def test_idle_task_event_handler(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, idle_timeout_secs=0.2, cancel_on_idle_timeout=False)
        task.set_event_loop(asyncio.get_event_loop())

        idle_timeout = False

        @task.event_handler("on_idle_timeout")
        async def on_idle_timeout(task: PipelineTask):
            nonlocal idle_timeout
            idle_timeout = True
            await task.cancel()

        await task.run()
        assert True

    async def test_idle_task_frames(self):
        idle_timeout_secs = 0.2
        sleep_time_secs = idle_timeout_secs / 2

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(
            pipeline,
            idle_timeout_secs=idle_timeout_secs,
            idle_timeout_frames=(TextFrame,),
        )
        task.set_event_loop(asyncio.get_event_loop())

        async def delayed_frames():
            await asyncio.sleep(sleep_time_secs)
            await task.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await task.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await task.queue_frame(TextFrame("Hello Pipecat!"))

        start_time = time.time()

        tasks = {asyncio.create_task(task.run()), asyncio.create_task(delayed_frames())}

        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        diff_time = time.time() - start_time

        self.assertGreater(diff_time, sleep_time_secs * 3)
