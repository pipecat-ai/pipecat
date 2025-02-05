#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncIterable, Iterable, List

from loguru import logger
from pydantic import BaseModel, ConfigDict

from pipecat.clocks.base_clock import BaseClock
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    CancelFrame,
    CancelTaskFrame,
    EndFrame,
    EndTaskFrame,
    ErrorFrame,
    Frame,
    HeartbeatFrame,
    MetricsFrame,
    StartFrame,
    StopTaskFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.base_task import BaseTask
from pipecat.pipeline.task_observer import TaskObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.asyncio import TaskManager
from pipecat.utils.utils import obj_count, obj_id

HEARTBEAT_SECONDS = 1.0
HEARTBEAT_MONITOR_SECONDS = HEARTBEAT_SECONDS * 5


class PipelineParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    allow_interruptions: bool = False
    enable_heartbeats: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    send_initial_empty_metrics: bool = True
    report_only_initial_ttfb: bool = False
    observers: List[BaseObserver] = []
    heartbeats_period_secs: float = HEARTBEAT_SECONDS


class PipelineTaskSource(FrameProcessor):
    """This is the source processor that is linked at the beginning of the
    pipeline given to the pipeline task. It allows us to easily push frames
    downstream to the pipeline and also receive upstream frames coming from the
    pipeline.

    """

    def __init__(self, up_queue: asyncio.Queue, **kwargs):
        super().__init__(**kwargs)
        self._up_queue = up_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        match direction:
            case FrameDirection.UPSTREAM:
                await self._up_queue.put(frame)
            case FrameDirection.DOWNSTREAM:
                await self.push_frame(frame, direction)


class PipelineTaskSink(FrameProcessor):
    """This is the sink processor that is linked at the end of the pipeline
    given to the pipeline task. It allows us to receive downstream frames and
    act on them, for example, waiting to receive an EndFrame.

    """

    def __init__(self, down_queue: asyncio.Queue, **kwargs):
        super().__init__(**kwargs)
        self._down_queue = down_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self._down_queue.put(frame)


class PipelineTask(BaseTask):
    def __init__(
        self,
        pipeline: BasePipeline,
        params: PipelineParams = PipelineParams(),
        clock: BaseClock = SystemClock(),
    ):
        self._id: int = obj_id()
        self._name: str = f"{self.__class__.__name__}#{obj_count(self)}"

        self._pipeline = pipeline
        self._clock = clock
        self._params = params
        self._finished = False

        # This queue receives frames coming from the pipeline upstream.
        self._up_queue = asyncio.Queue()
        # This queue receives frames coming from the pipeline downstream.
        self._down_queue = asyncio.Queue()
        # This queue is the queue used to push frames to the pipeline.
        self._push_queue = asyncio.Queue()
        # This is the heartbeat queue. When a heartbeat frame is received in the
        # down queue we add it to the heartbeat queue for processing.
        self._heartbeat_queue = asyncio.Queue()
        # This event is used to indicate an EndFrame has been received in the
        # down queue.
        self._endframe_event = asyncio.Event()

        self._source = PipelineTaskSource(self._up_queue)
        self._source.link(pipeline)

        self._sink = PipelineTaskSink(self._down_queue)
        pipeline.link(self._sink)

        self._task_manager = TaskManager()

        self._observer = TaskObserver(observers=params.observers, task_manager=self._task_manager)

    @property
    def id(self) -> int:
        """Returns the unique indetifier for this task."""
        return self._id

    @property
    def name(self) -> str:
        """Returns the name of this task."""
        return self._name

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._task_manager.set_event_loop(loop)

    def has_finished(self) -> bool:
        """Indicates whether the tasks has finished. That is, all processors
        have stopped.

        """
        return self._finished

    async def stop_when_done(self):
        """This is a helper function that sends an EndFrame to the pipeline in
        order to stop the task after everything in it has been processed.

        """
        logger.debug(f"Task {self} scheduled to stop when done")
        await self.queue_frame(EndFrame())

    async def cancel(self):
        """
        Stops the running pipeline immediately.
        """
        logger.debug(f"Canceling pipeline task {self}")
        # Make sure everything is cleaned up downstream. This is sent
        # out-of-band from the main streaming task which is what we want since
        # we want to cancel right away.
        await self._source.push_frame(CancelFrame())
        # Only cancel the push task. Everything else will be cancelled in run().
        await self._task_manager.cancel_task(self._process_push_task)

    async def run(self):
        """
        Starts running the given pipeline.
        """
        if self.has_finished():
            return
        try:
            push_task = await self._create_tasks()
            await self._task_manager.wait_for_task(push_task)
        except asyncio.CancelledError:
            # We are awaiting on the push task and it might be cancelled
            # (e.g. Ctrl-C). This means we will get a CancelledError here as
            # well, because you get a CancelledError in every place you are
            # awaiting a task.
            pass
        await self._cancel_tasks()
        await self._cleanup()
        self._print_dangling_tasks()
        self._finished = True

    async def queue_frame(self, frame: Frame):
        """
        Queue a frame to be pushed down the pipeline.
        """
        await self._push_queue.put(frame)

    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        """
        Queues multiple frames to be pushed down the pipeline.
        """
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame)

    async def _create_tasks(self):
        self._process_up_task = self._task_manager.create_task(
            self._process_up_queue(), f"{self}::_process_up_queue"
        )
        self._process_down_task = self._task_manager.create_task(
            self._process_down_queue(), f"{self}::_process_down_queue"
        )
        self._process_push_task = self._task_manager.create_task(
            self._process_push_queue(), f"{self}::_process_push_queue"
        )

        await self._observer.start()

        return self._process_push_task

    def _maybe_start_heartbeat_tasks(self):
        if self._params.enable_heartbeats:
            self._heartbeat_push_task = self._task_manager.create_task(
                self._heartbeat_push_handler(), f"{self}::_heartbeat_push_handler"
            )
            self._heartbeat_monitor_task = self._task_manager.create_task(
                self._heartbeat_monitor_handler(), f"{self}::_heartbeat_monitor_handler"
            )

    async def _cancel_tasks(self):
        await self._maybe_cancel_heartbeat_tasks()

        await self._task_manager.cancel_task(self._process_up_task)
        await self._task_manager.cancel_task(self._process_down_task)

        await self._observer.stop()

    async def _maybe_cancel_heartbeat_tasks(self):
        if self._params.enable_heartbeats:
            await self._task_manager.cancel_task(self._heartbeat_push_task)
            await self._task_manager.cancel_task(self._heartbeat_monitor_task)

    def _initial_metrics_frame(self) -> MetricsFrame:
        processors = self._pipeline.processors_with_metrics()
        data = []
        for p in processors:
            data.append(TTFBMetricsData(processor=p.name, value=0.0))
            data.append(ProcessingMetricsData(processor=p.name, value=0.0))
        return MetricsFrame(data=data)

    async def _wait_for_endframe(self):
        await self._endframe_event.wait()
        self._endframe_event.clear()

    async def _cleanup(self):
        await self._source.cleanup()
        await self._pipeline.cleanup()
        await self._sink.cleanup()

    async def _process_push_queue(self):
        """This is the task that runs the pipeline for the first time by sending
        a StartFrame and by pushing any other frames queued by the user. It runs
        until the tasks is canceled or stopped (e.g. with an EndFrame).

        """
        self._clock.start()

        self._maybe_start_heartbeat_tasks()

        start_frame = StartFrame(
            clock=self._clock,
            task_manager=self._task_manager,
            allow_interruptions=self._params.allow_interruptions,
            enable_metrics=self._params.enable_metrics,
            enable_usage_metrics=self._params.enable_usage_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb,
            observer=self._observer,
        )
        await self._source.queue_frame(start_frame, FrameDirection.DOWNSTREAM)

        if self._params.enable_metrics and self._params.send_initial_empty_metrics:
            await self._source.queue_frame(self._initial_metrics_frame(), FrameDirection.DOWNSTREAM)

        running = True
        should_cleanup = True
        while running:
            frame = await self._push_queue.get()
            await self._source.queue_frame(frame, FrameDirection.DOWNSTREAM)
            if isinstance(frame, EndFrame):
                await self._wait_for_endframe()
            running = not isinstance(frame, (CancelFrame, EndFrame, StopTaskFrame))
            should_cleanup = not isinstance(frame, StopTaskFrame)
            self._push_queue.task_done()
        # Cleanup only if we need to.
        if should_cleanup:
            await self._cleanup()

    async def _process_up_queue(self):
        """This is the task that processes frames coming upstream from the
        pipeline. These frames might indicate, for example, that we want the
        pipeline to be stopped (e.g. EndTaskFrame) in which case we would send
        an EndFrame down the pipeline.

        """
        while True:
            frame = await self._up_queue.get()
            if isinstance(frame, EndTaskFrame):
                # Tell the task we should end nicely.
                await self.queue_frame(EndFrame())
            elif isinstance(frame, CancelTaskFrame):
                # Tell the task we should end right away.
                await self.queue_frame(CancelFrame())
            elif isinstance(frame, StopTaskFrame):
                await self.queue_frame(StopTaskFrame())
            elif isinstance(frame, ErrorFrame):
                logger.error(f"Error running app: {frame}")
                if frame.fatal:
                    # Cancel all tasks downstream.
                    await self.queue_frame(CancelFrame())
                    # Tell the task we should stop.
                    await self.queue_frame(StopTaskFrame())
            self._up_queue.task_done()

    async def _process_down_queue(self):
        """This tasks process frames coming downstream from the pipeline. For
        example, heartbeat frames or an EndFrame which would indicate all
        processors have handled the EndFrame and therefore we can exit the task
        cleanly.

        """
        while True:
            frame = await self._down_queue.get()
            if isinstance(frame, EndFrame):
                self._endframe_event.set()
            elif isinstance(frame, HeartbeatFrame):
                await self._heartbeat_queue.put(frame)
            self._down_queue.task_done()

    async def _heartbeat_push_handler(self):
        """
        This tasks pushes a heartbeat frame every heartbeat period.
        """
        while True:
            # Don't use `queue_frame()` because if an EndFrame is queued the
            # task will just stop waiting for the pipeline to finish not
            # allowing more frames to be pushed.
            await self._source.queue_frame(HeartbeatFrame(timestamp=self._clock.get_time()))
            await asyncio.sleep(self._params.heartbeats_period_secs)

    async def _heartbeat_monitor_handler(self):
        """This tasks monitors heartbeat frames. If a heartbeat frame has not
        been received for a long period a warning will be logged. It also logs
        the time that a heartbeat frame takes to processes, that is how long it
        takes for the heartbeat frame to traverse all the pipeline.

        """
        wait_time = HEARTBEAT_MONITOR_SECONDS
        while True:
            try:
                frame = await asyncio.wait_for(self._heartbeat_queue.get(), timeout=wait_time)
                process_time = (self._clock.get_time() - frame.timestamp) / 1_000_000_000
                logger.trace(f"{self}: heartbeat frame processed in {process_time} seconds")
                self._heartbeat_queue.task_done()
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self}: heartbeat frame not received for more than {wait_time} seconds"
                )

    def _print_dangling_tasks(self):
        tasks = [t.get_name() for t in self._task_manager.current_tasks()]
        if tasks:
            logger.warning(f"Dangling tasks detected: {tasks}")

    def __str__(self):
        return self.name
