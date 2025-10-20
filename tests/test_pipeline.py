#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
import unittest

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    HeartbeatFrame,
    InputAudioRawFrame,
    StartFrame,
    StopFrame,
    TextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.base_task import PipelineTaskParams
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
            pipeline_params=PipelineParams(start_metadata={"foo": "bar"}),
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

        await task.queue_frame(TextFrame(text="Hello!"))
        await task.queue_frames([TextFrame(text="Bye!"), EndFrame()])
        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))
        assert task.has_finished()

    async def test_task_observers(self):
        frame_received = False

        class CustomObserver(BaseObserver):
            async def on_push_frame(self, data: FramePushed):
                nonlocal frame_received

                if isinstance(data.frame, TextFrame):
                    frame_received = True

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, observers=[CustomObserver()])

        await task.queue_frames([TextFrame(text="Hello Downstream!"), EndFrame()])
        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))
        assert frame_received

    async def test_task_add_observer(self):
        frame_received = False
        frame_count_1 = 0
        frame_count_2 = 0

        class CustomObserver(BaseObserver):
            async def on_push_frame(self, data: FramePushed):
                nonlocal frame_received

                if isinstance(data.frame, TextFrame):
                    frame_received = True

        class CustomAddObserver1(BaseObserver):
            async def on_push_frame(self, data: FramePushed):
                nonlocal frame_count_1

                if isinstance(data.source, IdentityFilter) and isinstance(data.frame, TextFrame):
                    frame_count_1 += 1

        class CustomAddObserver2(BaseObserver):
            async def on_push_frame(self, data: FramePushed):
                nonlocal frame_count_2

                if isinstance(data.source, IdentityFilter) and isinstance(data.frame, TextFrame):
                    frame_count_2 += 1

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, observers=[CustomObserver()])

        # Add a new observer right away, before doing anything else with the task.
        observer1 = CustomAddObserver1()
        task.add_observer(observer1)

        async def delayed_add_observer():
            observer2 = CustomAddObserver2()
            # Wait after the pipeline is started and add another observer.
            await asyncio.sleep(0.1)
            task.add_observer(observer2)
            # Push a TextFrame and wait for the observer to pick it up.
            await task.queue_frame(TextFrame(text="Hello Downstream!"))
            await asyncio.sleep(0.1)
            # Remove both observers.
            await task.remove_observer(observer1)
            await task.remove_observer(observer2)
            # Push another TextFrame. This time the counter should not
            # increments since we have removed the observer.
            await task.queue_frame(TextFrame(text="Hello Downstream!"))
            await asyncio.sleep(0.1)
            # Finally end the pipeline.
            await task.queue_frame(EndFrame())

        await asyncio.gather(
            task.run(PipelineTaskParams(loop=asyncio.get_event_loop())), delayed_add_observer()
        )

        assert frame_received
        assert frame_count_1 == 1
        assert frame_count_2 == 1

    async def test_task_started_ended_event_handler(self):
        start_received = False
        end_received = False

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline)

        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame: StartFrame):
            nonlocal start_received
            start_received = True

        @task.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(task, frame: Frame):
            nonlocal end_received
            end_received = isinstance(frame, EndFrame)

        await task.queue_frame(EndFrame())
        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))

        assert start_received
        assert end_received

    async def test_task_stopped_event_handler(self):
        stop_received = False

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline)

        @task.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(task, frame: Frame):
            nonlocal stop_received
            stop_received = isinstance(frame, StopFrame)

        await task.queue_frame(StopFrame())
        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))

        assert stop_received

    async def test_task_frame_reached_event_handlers(self):
        upstream_received = False
        downstream_received = False

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, cancel_on_idle_timeout=False)
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
            await asyncio.wait_for(
                task.run(PipelineTaskParams(loop=asyncio.get_event_loop())),
                timeout=1.0,
            )
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

        expected_heartbeats = 1.0 / 0.2

        await task.queue_frame(TextFrame(text="Hello!"))
        try:
            await asyncio.wait_for(
                task.run(PipelineTaskParams(loop=asyncio.get_event_loop())),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            pass
        assert heartbeats_counter == expected_heartbeats

    async def test_idle_task(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, idle_timeout_secs=0.2)
        # This shouldn't freeze, so nothing to check really.
        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))

    async def test_no_idle_task(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, idle_timeout_secs=0.2, cancel_on_idle_timeout=False)
        try:
            await asyncio.wait_for(
                task.run(PipelineTaskParams(loop=asyncio.get_event_loop())),
                timeout=0.3,
            )
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
        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))

    async def test_idle_task_event_handler_no_frames(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, idle_timeout_secs=0.2, cancel_on_idle_timeout=False)

        idle_timeout = False

        @task.event_handler("on_idle_timeout")
        async def on_idle_timeout(task: PipelineTask):
            nonlocal idle_timeout
            idle_timeout = True
            await task.cancel()

        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))
        assert idle_timeout

    async def test_idle_task_event_handler_quiet_user(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        task = PipelineTask(pipeline, idle_timeout_secs=0.2, cancel_on_idle_timeout=False)

        idle_timeout = 0

        @task.event_handler("on_idle_timeout")
        async def on_idle_timeout(task: PipelineTask):
            nonlocal idle_timeout
            idle_timeout += 1
            # Stay a bit longer here while user audio frames are still being
            # pushed. We do this to make sure this function is only called once.
            await asyncio.sleep(0.1)
            await task.queue_frame(EndFrame())

        async def send_audio():
            # We send audio during and after the 0.2 seconds of idle
            # timeout. Inside `on_idle_timeout` we are waiting a little bit
            # simulating the pipeline finishing (e.g. goodbye message from bot
            # flushing).
            for i in range(30):
                await task.queue_frame(
                    InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
                )
                await asyncio.sleep(0.01)

        await asyncio.gather(
            send_audio(), task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))
        )
        assert idle_timeout == 1

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

        async def delayed_frames():
            await asyncio.sleep(sleep_time_secs)
            await task.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await task.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await task.queue_frame(TextFrame("Hello Pipecat!"))

        start_time = time.time()

        tasks = [
            asyncio.create_task(task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))),
            asyncio.create_task(delayed_frames()),
        ]

        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        diff_time = time.time() - start_time

        self.assertGreater(diff_time, sleep_time_secs * 3)

        # Wait for the pending tasks to complete.
        await asyncio.gather(*pending)

    async def test_task_cancel_timeout(self):
        class CancelFilter(FrameProcessor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if not isinstance(frame, CancelFrame):
                    await self.push_frame(frame, direction)

        pipeline = Pipeline([CancelFilter()])
        task = PipelineTask(pipeline, cancel_timeout_secs=0.2)

        cancelled = False

        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task: PipelineTask, frame: StartFrame):
            await task.cancel()

        @task.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(task: PipelineTask, frame: Frame):
            nonlocal cancelled
            cancelled = isinstance(frame, CancelFrame)

        try:
            await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))
        except asyncio.CancelledError:
            assert cancelled

    async def test_task_error(self):
        class ErrorProcessor(FrameProcessor):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, TextFrame):
                    await self.push_error(ErrorFrame("Boo!"))

                await self.push_frame(frame, direction)

        error_received = False

        pipeline = Pipeline([ErrorProcessor()])
        task = PipelineTask(pipeline)

        @task.event_handler("on_pipeline_error")
        async def on_pipeline_error(task: PipelineTask, frame: ErrorFrame):
            nonlocal error_received
            error_received = True
            await task.cancel()

        await task.queue_frame(TextFrame(text="Hello from Pipecat!"))

        try:
            await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))
        except asyncio.CancelledError:
            assert error_received
