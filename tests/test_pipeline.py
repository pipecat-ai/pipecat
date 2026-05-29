#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import time
import unittest

from loguru import logger

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
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, WorkerParams
from pipecat.processors.filters.frame_filter import FrameFilter
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

    async def test_parallel_internal_frames_buffered_during_start(self):
        """Frames pushed by internal processors during StartFrame processing
        should be buffered and only released after StartFrame synchronization
        completes."""

        class EmitOnStartProcessor(FrameProcessor):
            """Pushes a TextFrame when it receives a StartFrame."""

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                if isinstance(frame, StartFrame):
                    await self.push_frame(TextFrame(text="from start"))

        pipeline = ParallelPipeline([EmitOnStartProcessor()], [IdentityFilter()])

        frames_to_send = [TextFrame(text="Hello!")]

        # StartFrame should come first, then the TextFrame emitted during
        # StartFrame processing, then the regular TextFrame.
        expected_down_frames = [StartFrame, TextFrame, TextFrame]
        await run_test(
            pipeline,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            ignore_start=False,
        )


class TestPipelineTask(unittest.IsolatedAsyncioTestCase):
    async def test_task_single(self):
        pipeline = Pipeline([IdentityFilter()])
        worker = PipelineWorker(pipeline)

        await worker.queue_frame(TextFrame(text="Hello!"))
        await worker.queue_frames([TextFrame(text="Bye!"), EndFrame()])
        await worker.run(WorkerParams(loop=asyncio.get_event_loop()))
        assert worker.has_finished()

    async def test_task_observers(self):
        frame_received = False

        class CustomObserver(BaseObserver):
            async def on_push_frame(self, data: FramePushed):
                nonlocal frame_received

                if isinstance(data.frame, TextFrame):
                    frame_received = True

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(pipeline, observers=[CustomObserver()])

        await worker.queue_frames([TextFrame(text="Hello Downstream!"), EndFrame()])
        await worker.run(WorkerParams(loop=asyncio.get_event_loop()))
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
        worker = PipelineWorker(pipeline, observers=[CustomObserver()])

        # Add a new observer right away, before doing anything else with the worker.
        observer1 = CustomAddObserver1()
        worker.add_observer(observer1)

        async def delayed_add_observer():
            observer2 = CustomAddObserver2()
            # Wait after the pipeline is started and add another observer.
            await asyncio.sleep(0.1)
            worker.add_observer(observer2)
            # Push a TextFrame and wait for the observer to pick it up.
            await worker.queue_frame(TextFrame(text="Hello Downstream!"))
            await asyncio.sleep(0.1)
            # Remove both observers.
            await worker.remove_observer(observer1)
            await worker.remove_observer(observer2)
            # Push another TextFrame. This time the counter should not
            # increments since we have removed the observer.
            await worker.queue_frame(TextFrame(text="Hello Downstream!"))
            await asyncio.sleep(0.1)
            # Finally end the pipeline.
            await worker.queue_frame(EndFrame())

        await asyncio.gather(
            worker.run(WorkerParams(loop=asyncio.get_event_loop())), delayed_add_observer()
        )

        assert frame_received
        assert frame_count_1 == 1
        assert frame_count_2 == 1

    async def test_task_started_ended_event_handler(self):
        start_received = False
        end_received = False

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(pipeline)

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker, frame: StartFrame):
            nonlocal start_received
            start_received = True

        @worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(worker, frame: Frame):
            nonlocal end_received
            end_received = isinstance(frame, EndFrame)

        await worker.queue_frame(EndFrame())
        await worker.run(WorkerParams(loop=asyncio.get_event_loop()))

        assert start_received
        assert end_received

    async def test_task_stopped_event_handler(self):
        stop_received = False

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(pipeline)

        @worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(worker, frame: Frame):
            nonlocal stop_received
            stop_received = isinstance(frame, StopFrame)

        await worker.queue_frame(StopFrame())
        await worker.run(WorkerParams(loop=asyncio.get_event_loop()))

        assert stop_received

    async def test_task_frame_reached_event_handlers(self):
        upstream_received = False
        downstream_received = False

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False)
        worker.set_reached_upstream_filter((TextFrame,))
        worker.set_reached_downstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_upstream")
        async def on_frame_reached_upstream(worker, frame):
            nonlocal upstream_received
            if isinstance(frame, TextFrame) and frame.text == "Hello Upstream!":
                upstream_received = True

        @worker.event_handler("on_frame_reached_downstream")
        async def on_frame_reached_downstream(worker, frame):
            nonlocal downstream_received
            if isinstance(frame, TextFrame) and frame.text == "Hello Downstream!":
                downstream_received = True
                await identity.push_frame(
                    TextFrame(text="Hello Upstream!"), FrameDirection.UPSTREAM
                )

        await worker.queue_frame(TextFrame(text="Hello Downstream!"))

        try:
            await asyncio.wait_for(
                worker.run(WorkerParams(loop=asyncio.get_event_loop())),
                timeout=1.0,
            )
        except TimeoutError:
            pass

        assert upstream_received
        assert downstream_received

    async def test_task_queue_frame_upstream(self):
        upstream_received = False

        pipeline = Pipeline([IdentityFilter()])
        worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False)
        worker.set_reached_upstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_upstream")
        async def on_frame_reached_upstream(worker, frame):
            nonlocal upstream_received
            if isinstance(frame, TextFrame) and frame.text == "Hello Upstream!":
                upstream_received = True

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker, frame):
            await worker.queue_frame(TextFrame(text="Hello Upstream!"), FrameDirection.UPSTREAM)

        try:
            await asyncio.wait_for(
                worker.run(WorkerParams(loop=asyncio.get_event_loop())),
                timeout=1.0,
            )
        except TimeoutError:
            pass

        assert upstream_received

    async def test_task_queue_frames_upstream(self):
        upstream_texts = []

        pipeline = Pipeline([IdentityFilter()])
        worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False)
        worker.set_reached_upstream_filter((TextFrame,))

        @worker.event_handler("on_frame_reached_upstream")
        async def on_frame_reached_upstream(worker, frame):
            if isinstance(frame, TextFrame):
                upstream_texts.append(frame.text)

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker, frame):
            await worker.queue_frames(
                [TextFrame(text="First"), TextFrame(text="Second")],
                FrameDirection.UPSTREAM,
            )

        try:
            await asyncio.wait_for(
                worker.run(WorkerParams(loop=asyncio.get_event_loop())),
                timeout=1.0,
            )
        except TimeoutError:
            pass

        assert "First" in upstream_texts
        assert "Second" in upstream_texts

    async def test_task_heartbeats(self):
        period_secs = 0.2
        expected_heartbeats = 5
        heartbeats_counter = 0
        received_expected = asyncio.Event()

        async def heartbeat_received(processor: FrameProcessor, heartbeat: HeartbeatFrame):
            nonlocal heartbeats_counter
            heartbeats_counter += 1
            if heartbeats_counter >= expected_heartbeats:
                received_expected.set()

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        heartbeats_observer = HeartbeatsObserver(
            target=identity, heartbeat_callback=heartbeat_received
        )
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                enable_heartbeats=True,
                heartbeats_period_secs=period_secs,
            ),
            observers=[heartbeats_observer],
            cancel_on_idle_timeout=False,
        )

        async def wait_for_heartbeats():
            # Wait until we've observed the expected number of heartbeats, then
            # stop the pipeline. We don't assert on the count observed within a
            # fixed wall-clock window: heartbeats are timer-driven, so the count
            # in any given window depends on event-loop scheduling precision and
            # is off-by-one under load (which made this test flaky in CI). The
            # generous timeout only guards against heartbeats never firing.
            try:
                await asyncio.wait_for(received_expected.wait(), timeout=5.0)
            except TimeoutError:
                pass
            await worker.queue_frame(EndFrame())

        await worker.queue_frame(TextFrame(text="Hello!"))

        start_time = time.time()
        await asyncio.gather(
            worker.run(WorkerParams(loop=asyncio.get_event_loop())),
            wait_for_heartbeats(),
        )
        elapsed = time.time() - start_time

        # We observed the expected number of heartbeats...
        assert heartbeats_counter >= expected_heartbeats
        # ...and they were paced by the configured period: each heartbeat waits a
        # full period, so N heartbeats span at least (N - 1) periods. asyncio.sleep
        # is a guaranteed lower bound, so this is robust to scheduling jitter while
        # still catching heartbeats that fire too fast.
        assert elapsed >= (expected_heartbeats - 1) * period_secs

    async def test_heartbeat_monitor_respects_custom_timeout(self):
        """Verify the heartbeat monitor uses heartbeats_monitor_secs from params."""

        class HeartbeatBlocker(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if not isinstance(frame, HeartbeatFrame):
                    await self.push_frame(frame, direction)

        log_output = io.StringIO()
        handler_id = logger.add(log_output, level="WARNING", format="{message}")

        custom_monitor_secs = 0.3

        try:
            pipeline = Pipeline([HeartbeatBlocker()])
            worker = PipelineWorker(
                pipeline,
                params=PipelineParams(
                    enable_heartbeats=True,
                    heartbeats_period_secs=0.1,
                    heartbeats_monitor_secs=custom_monitor_secs,
                ),
                cancel_on_idle_timeout=False,
            )

            try:
                await asyncio.wait_for(
                    worker.run(WorkerParams(loop=asyncio.get_event_loop())),
                    timeout=0.6,
                )
            except TimeoutError:
                pass

            log_text = log_output.getvalue()
            assert f"more than {custom_monitor_secs} seconds" in log_text
        finally:
            logger.remove(handler_id)

    async def test_idle_task(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(pipeline, idle_timeout_secs=0.2)
        # This shouldn't freeze, so nothing to check really.
        await worker.run(WorkerParams(loop=asyncio.get_event_loop()))

    async def test_cancel_runner_on_idle_timeout_cancels_peers(self):
        """``cancel_runner_on_idle_timeout`` brings down the whole runner, not just the worker.

        Build a runner with a forever-running peer ``BaseWorker`` and a
        ``PipelineWorker`` set to time out quickly. Without the new flag the
        runner would hang on the peer; with it, the idle timeout sends a
        ``BusCancelMessage`` and the runner shuts everything down.
        """
        from pipecat.bus import BusCancelWorkerMessage
        from pipecat.workers.base_worker import BaseWorker
        from pipecat.workers.runner import WorkerRunner

        class PeerWorker(BaseWorker):
            """Bus-only worker that exits on cancel so the runner can finish."""

            async def _handle_worker_cancel(self, message: BusCancelWorkerMessage) -> None:
                await super()._handle_worker_cancel(message)
                self._finished_event.set()

        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        main_worker = PipelineWorker(
            pipeline,
            name="main",
            idle_timeout_secs=0.2,
            cancel_runner_on_idle_timeout=True,
        )
        peer = PeerWorker("peer")

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(peer, main_worker)

        await asyncio.wait_for(runner.run(), timeout=5.0)

        # Runner finishes only when both root workers stop. If
        # ``cancel_runner_on_idle_timeout`` worked, the peer received a
        # BusCancelWorkerMessage and exited; otherwise this test times out.
        self.assertTrue(peer._finished_event.is_set())

    async def test_cancel_on_idle_timeout_false_overrides_runner_flag(self):
        """``cancel_on_idle_timeout=False`` keeps the worker alive even with the runner flag on.

        Opting out of local cancellation also opts out of the runner-wide
        cancel — the worker keeps running past the idle timeout and the
        ``on_idle_timeout`` event handler is responsible for the response.
        """
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=0.2,
            cancel_on_idle_timeout=False,
            # Default-True; the gating by cancel_on_idle_timeout=False should win.
        )

        idle_fired = asyncio.Event()

        @worker.event_handler("on_idle_timeout")
        async def on_idle(worker):
            idle_fired.set()
            await worker.queue_frame(EndFrame())

        await asyncio.wait_for(
            worker.run(WorkerParams(loop=asyncio.get_event_loop())),
            timeout=2.0,
        )
        self.assertTrue(idle_fired.is_set())

    async def test_no_idle_task(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=0.2,
            cancel_on_idle_timeout=False,
        )
        try:
            await asyncio.wait_for(
                worker.run(WorkerParams(loop=asyncio.get_event_loop())),
                timeout=0.3,
            )
        except TimeoutError:
            assert True
        else:
            assert False

    async def test_idle_task_heartbeats(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                enable_heartbeats=True,
                heartbeats_period_secs=0.1,
            ),
            idle_timeout_secs=0.3,
        )
        await worker.run(WorkerParams(loop=asyncio.get_event_loop()))

    async def test_idle_task_event_handler_no_frames(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=0.2,
            cancel_on_idle_timeout=False,
        )

        idle_timeout = False

        @worker.event_handler("on_idle_timeout")
        async def on_idle_timeout(worker: PipelineWorker):
            nonlocal idle_timeout
            idle_timeout = True
            await worker.cancel()

        await worker.run(WorkerParams(loop=asyncio.get_event_loop()))
        assert idle_timeout

    async def test_idle_task_event_handler_quiet_user(self):
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=0.2,
            cancel_on_idle_timeout=False,
        )

        idle_timeout = 0

        @worker.event_handler("on_idle_timeout")
        async def on_idle_timeout(worker: PipelineWorker):
            nonlocal idle_timeout
            idle_timeout += 1
            # Stay a bit longer here while user audio frames are still being
            # pushed. We do this to make sure this function is only called once.
            await asyncio.sleep(0.1)
            await worker.queue_frame(EndFrame())

        async def send_audio():
            # We send audio during and after the 0.2 seconds of idle
            # timeout. Inside `on_idle_timeout` we are waiting a little bit
            # simulating the pipeline finishing (e.g. goodbye message from bot
            # flushing).
            for i in range(30):
                await worker.queue_frame(
                    InputAudioRawFrame(audio=b"\x00", sample_rate=16000, num_channels=1)
                )
                await asyncio.sleep(0.01)

        await asyncio.gather(send_audio(), worker.run(WorkerParams(loop=asyncio.get_event_loop())))
        assert idle_timeout == 1

    async def test_idle_task_frames(self):
        idle_timeout_secs = 0.2
        sleep_time_secs = idle_timeout_secs / 2

        # Use the identify filter so the frames just reach the end of the pipeline.
        identity = IdentityFilter()
        pipeline = Pipeline([identity])
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=idle_timeout_secs,
            idle_timeout_frames=(TextFrame,),
        )

        async def delayed_frames():
            """Sending multiple text frames.

            The total amount of elapsed time in this function should be greater
            than the worker idle timeout. If an idle timeout event is triggered it
            means we haven't detected that the TextFrames have been pushed.
            """
            await asyncio.sleep(sleep_time_secs)
            await worker.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await worker.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await worker.queue_frame(TextFrame("Hello Pipecat!"))

        start_time = time.time()

        tasks = [
            asyncio.create_task(worker.run(WorkerParams(loop=asyncio.get_event_loop()))),
            asyncio.create_task(delayed_frames()),
        ]

        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        diff_time = time.time() - start_time

        self.assertGreater(diff_time, sleep_time_secs * 3)

        # Wait for the pending tasks to complete.
        await asyncio.gather(*pending)

    async def test_idle_task_swallowed_frames(self):
        idle_timeout_secs = 0.2
        sleep_time_secs = idle_timeout_secs / 2

        # Block all frames (except system frames). Here, we are testing that
        # generated frames don't trigger an idle timeout (they don't need to
        # reach the end of the pipeline).
        filter = FrameFilter(types=())
        pipeline = Pipeline([filter])
        worker = PipelineWorker(
            pipeline,
            idle_timeout_secs=idle_timeout_secs,
            idle_timeout_frames=(TextFrame,),
        )

        start_time = time.time()

        async def delayed_frames():
            """Sending multiple text frames.

            The total amount of elapsed time in this function should be greater
            than the worker idle timeout. If an idle timeout event is triggered it
            means we haven't detected that the TextFrames have been pushed.
            """
            await asyncio.sleep(sleep_time_secs)
            await worker.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await worker.queue_frame(TextFrame("Hello Pipecat!"))
            await asyncio.sleep(sleep_time_secs)
            await worker.queue_frame(TextFrame("Hello Pipecat!"))

        tasks = [
            asyncio.create_task(worker.run(WorkerParams(loop=asyncio.get_event_loop()))),
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
        worker = PipelineWorker(pipeline, cancel_timeout_secs=0.2)

        cancelled = False

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker: PipelineWorker, frame: StartFrame):
            await worker.cancel()

        @worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(worker: PipelineWorker, frame: Frame):
            nonlocal cancelled
            cancelled = isinstance(frame, CancelFrame)

        try:
            await worker.run(WorkerParams(loop=asyncio.get_event_loop()))
        except asyncio.CancelledError:
            assert cancelled

    async def test_task_cancel_before_start_reaches_sink(self):
        class StartBlocker(FrameProcessor):
            def __init__(self, *, start_received: asyncio.Event, **kwargs):
                super().__init__(**kwargs)
                self._start_received = start_received
                self._block = asyncio.Event()

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)

                if isinstance(frame, StartFrame):
                    self._start_received.set()
                    await self._block.wait()

                await self.push_frame(frame, direction)

        start_received = asyncio.Event()
        pipeline = Pipeline([StartBlocker(start_received=start_received)])
        worker = PipelineWorker(pipeline, cancel_timeout_secs=0.1)

        run_task = asyncio.create_task(worker.run(WorkerParams(loop=asyncio.get_event_loop())))
        await start_received.wait()
        await worker.cancel()
        await asyncio.wait_for(run_task, timeout=1.0)

        assert worker.has_finished()

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
        worker = PipelineWorker(pipeline)

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker: PipelineWorker, frame: ErrorFrame):
            nonlocal error_received
            error_received = True
            await worker.cancel()

        await worker.queue_frame(TextFrame(text="Hello from Pipecat!"))

        try:
            await worker.run(WorkerParams(loop=asyncio.get_event_loop()))
        except asyncio.CancelledError:
            assert error_received


if __name__ == "__main__":
    unittest.main()
