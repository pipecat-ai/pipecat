#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from enum import Enum
from typing import Awaitable, Callable, Coroutine, Optional

from loguru import logger

from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class FrameDirection(Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2


class FrameProcessor(BaseObject):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        metrics: Optional[FrameProcessorMetrics] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self._parent: Optional["FrameProcessor"] = None
        self._prev: Optional["FrameProcessor"] = None
        self._next: Optional["FrameProcessor"] = None

        # Clock
        self._clock: Optional[BaseClock] = None

        # Task Manager
        self._task_manager: Optional[BaseTaskManager] = None

        # Other properties
        self._allow_interruptions = False
        self._enable_metrics = False
        self._enable_usage_metrics = False
        self._report_only_initial_ttfb = False
        self._observer = None

        # Cancellation is done through CancelFrame (a system frame). This could
        # cause other events being triggered (e.g. closing a transport) which
        # could also cause other frames to be pushed from other tasks
        # (e.g. EndFrame). So, when we are cancelling we don't want anything
        # else to be pushed.
        self._cancelling = False

        # Metrics
        self._metrics = metrics or FrameProcessorMetrics()
        self._metrics.set_processor_name(self.name)

        # Processors have an input queue. The input queue will be processed
        # immediately (default) or it will block if `pause_processing_frames()`
        # is called. To resume processing frames we need to call
        # `resume_processing_frames()` which will wake up the event.
        self.__should_block_frames = False
        self.__input_event = asyncio.Event()
        self.__input_frame_task: Optional[asyncio.Task] = None

        # Every processor in Pipecat should only output frames from a single
        # task. This avoid problems like audio overlapping. System frames are the
        # exception to this rule. This create this task.
        self.__push_frame_task: Optional[asyncio.Task] = None

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def interruptions_allowed(self):
        return self._allow_interruptions

    @property
    def metrics_enabled(self):
        return self._enable_metrics

    @property
    def usage_metrics_enabled(self):
        return self._enable_usage_metrics

    @property
    def report_only_initial_ttfb(self):
        return self._report_only_initial_ttfb

    def can_generate_metrics(self) -> bool:
        return False

    def set_core_metrics_data(self, data: MetricsData):
        self._metrics.set_core_metrics_data(data)

    async def start_ttfb_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_ttfb_metrics(self._report_only_initial_ttfb)

    async def stop_ttfb_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfb_metrics()
            if frame:
                await self.push_frame(frame)

    async def start_processing_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_processing_metrics()

    async def stop_processing_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_processing_metrics()
            if frame:
                await self.push_frame(frame)

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_llm_usage_metrics(tokens)
            if frame:
                await self.push_frame(frame)

    async def start_tts_usage_metrics(self, text: str):
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_tts_usage_metrics(text)
            if frame:
                await self.push_frame(frame)

    async def stop_all_metrics(self):
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    def create_task(self, coroutine: Coroutine) -> asyncio.Task:
        if not self._task_manager:
            raise Exception(f"{self} TaskManager is still not initialized.")
        name = f"{self}::{coroutine.cr_code.co_name}"
        return self._task_manager.create_task(coroutine, name)

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        if not self._task_manager:
            raise Exception(f"{self} TaskManager is still not initialized.")
        await self._task_manager.cancel_task(task, timeout)

    async def wait_for_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        if not self._task_manager:
            raise Exception(f"{self} TaskManager is still not initialized.")
        await self._task_manager.wait_for_task(task, timeout)

    async def cleanup(self):
        await super().cleanup()
        await self.__cancel_input_task()
        await self.__cancel_push_task()

    def link(self, processor: "FrameProcessor"):
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        if not self._task_manager:
            raise Exception(f"{self} TaskManager is still not initialized.")
        return self._task_manager.get_event_loop()

    def set_parent(self, parent: "FrameProcessor"):
        self._parent = parent

    def get_parent(self) -> Optional["FrameProcessor"]:
        return self._parent

    def get_clock(self) -> BaseClock:
        if not self._clock:
            raise Exception(f"{self} Clock is still not initialized.")
        return self._clock

    def get_task_manager(self) -> BaseTaskManager:
        if not self._task_manager:
            raise Exception(f"{self} TaskManager is still not initialized.")
        return self._task_manager

    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: Optional[
            Callable[["FrameProcessor", Frame, FrameDirection], Awaitable[None]]
        ] = None,
    ):
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        if isinstance(frame, SystemFrame):
            # We don't want to queue system frames.
            await self.process_frame(frame, direction)
        else:
            # We queue everything else.
            await self.__input_queue.put((frame, direction, callback))

    async def pause_processing_frames(self):
        logger.trace(f"{self}: pausing frame processing")
        self.__should_block_frames = True

    async def resume_processing_frames(self):
        logger.trace(f"{self}: resuming frame processing")
        self.__input_event.set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            self._clock = frame.clock
            self._task_manager = frame.task_manager
            self._allow_interruptions = frame.allow_interruptions
            self._enable_metrics = frame.enable_metrics
            self._enable_usage_metrics = frame.enable_usage_metrics
            self._report_only_initial_ttfb = frame.report_only_initial_ttfb
            self._observer = frame.observer
            await self.__start(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
        elif isinstance(frame, StopInterruptionFrame):
            self._should_report_ttfb = True
        elif isinstance(frame, CancelFrame):
            await self.__cancel(frame)

    async def push_error(self, error: ErrorFrame):
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if not self._check_ready(frame):
            return

        if isinstance(frame, SystemFrame):
            await self.__internal_push_frame(frame, direction)
        else:
            await self.__push_queue.put((frame, direction))

    async def __start(self, frame: StartFrame):
        self.__create_input_task()
        self.__create_push_task()

    async def __cancel(self, frame: CancelFrame):
        self._cancelling = True
        await self.__cancel_input_task()
        await self.__cancel_push_task()

    #
    # Handle interruptions
    #

    async def _start_interruption(self):
        try:
            # Cancel the push frame task. This will stop pushing frames downstream.
            await self.__cancel_push_task()

            # Cancel the input task. This will stop processing queued frames.
            await self.__cancel_input_task()
        except Exception as e:
            logger.exception(f"Uncaught exception in {self}: {e}")
            await self.push_error(ErrorFrame(str(e)))
            raise

        # Create a new input queue and task.
        self.__create_input_task()

        # Create a new output queue and task.
        self.__create_push_task()

    async def _stop_interruption(self):
        # Nothing to do right now.
        pass

    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        try:
            timestamp = self._clock.get_time() if self._clock else 0
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} from {self} to {self._next}")
                if self._observer:
                    await self._observer.on_push_frame(
                        self, self._next, frame, direction, timestamp
                    )
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                if self._observer:
                    await self._observer.on_push_frame(
                        self, self._prev, frame, direction, timestamp
                    )
                await self._prev.queue_frame(frame, direction)
        except Exception as e:
            logger.exception(f"Uncaught exception in {self}: {e}")
            await self.push_error(ErrorFrame(str(e)))
            raise

    def _check_ready(self, frame: Frame):
        # If we are trying to push a frame but we still have no clock, it means
        # we didn't process a StartFrame.
        if not self._clock:
            logger.error(
                f"{self} not properly initialized, missing super().process_frame(frame, direction)?"
            )
            return False
        return True

    def __create_input_task(self):
        if not self.__input_frame_task:
            self.__should_block_frames = False
            self.__input_event.clear()
            self.__input_queue = asyncio.Queue()
            self.__input_frame_task = self.create_task(self.__input_frame_task_handler())

    async def __cancel_input_task(self):
        if self.__input_frame_task:
            await self.cancel_task(self.__input_frame_task)
            self.__input_frame_task = None

    async def __input_frame_task_handler(self):
        while True:
            if self.__should_block_frames:
                logger.trace(f"{self}: frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_frames = False
                logger.trace(f"{self}: frame processing resumed")

            (frame, direction, callback) = await self.__input_queue.get()

            # Process the frame.
            await self.process_frame(frame, direction)

            # If this frame has an associated callback, call it now.
            if callback:
                await callback(self, frame, direction)

            self.__input_queue.task_done()

    def __create_push_task(self):
        if not self.__push_frame_task:
            self.__push_queue = asyncio.Queue()
            self.__push_frame_task = self.create_task(self.__push_frame_task_handler())

    async def __cancel_push_task(self):
        if self.__push_frame_task:
            await self.cancel_task(self.__push_frame_task)
            self.__push_frame_task = None

    async def __push_frame_task_handler(self):
        while True:
            (frame, direction) = await self.__push_queue.get()
            await self.__internal_push_frame(frame, direction)
            self.__push_queue.task_done()
