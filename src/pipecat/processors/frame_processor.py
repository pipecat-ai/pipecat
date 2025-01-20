#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
from enum import Enum
from typing import Awaitable, Callable, Optional

from loguru import logger

from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.utils import obj_count, obj_id


class FrameDirection(Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2


class FrameProcessor:
    def __init__(
        self,
        *,
        name: str | None = None,
        metrics: FrameProcessorMetrics | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs,
    ):
        self.id: int = obj_id()
        self.name = name or f"{self.__class__.__name__}#{obj_count(self)}"
        self._parent: "FrameProcessor" | None = None
        self._prev: "FrameProcessor" | None = None
        self._next: "FrameProcessor" | None = None
        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_running_loop()

        self._event_handlers: dict = {}

        # Clock
        self._clock: BaseClock | None = None

        # Properties
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
        # `resume_processing_frames()`.
        self.__should_block_frames = False
        self.__create_input_task()

        # Every processor in Pipecat should only output frames from a single
        # task. This avoid problems like audio overlapping. System frames are
        # the exception to this rule. This create this task.
        self.__create_push_task()

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

    async def cleanup(self):
        await self.__cancel_input_task()
        await self.__cancel_push_task()

    def link(self, processor: "FrameProcessor"):
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def set_parent(self, parent: "FrameProcessor"):
        self._parent = parent

    def get_parent(self) -> "FrameProcessor":
        return self._parent

    def get_clock(self) -> BaseClock:
        return self._clock

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
        logger.trace("f{self}: resuming frame processing")
        self.__input_event.set()
        self.__should_block_frames = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            self._clock = frame.clock
            self._allow_interruptions = frame.allow_interruptions
            self._enable_metrics = frame.enable_metrics
            self._enable_usage_metrics = frame.enable_usage_metrics
            self._report_only_initial_ttfb = frame.report_only_initial_ttfb
            self._observer = frame.observer
        elif isinstance(frame, StartInterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
        elif isinstance(frame, StopInterruptionFrame):
            self._should_report_ttfb = True
        elif isinstance(frame, CancelFrame):
            self._cancelling = True

    async def push_error(self, error: ErrorFrame):
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, SystemFrame):
            await self.__internal_push_frame(frame, direction)
        else:
            await self.__push_queue.put((frame, direction))

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def add_event_handler(self, event_name: str, handler):
        if event_name not in self._event_handlers:
            raise Exception(f"Event handler {event_name} not registered")
        self._event_handlers[event_name].append(handler)

    def _register_event_handler(self, event_name: str):
        if event_name in self._event_handlers:
            raise Exception(f"Event handler {event_name} already registered")
        self._event_handlers[event_name] = []

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

    def __create_input_task(self):
        self.__should_block_frames = False
        self.__input_queue = asyncio.Queue()
        self.__input_frame_task = self.get_event_loop().create_task(
            self.__input_frame_task_handler()
        )
        self.__input_event = asyncio.Event()

    async def __cancel_input_task(self):
        self.__input_frame_task.cancel()
        await self.__input_frame_task

    async def __input_frame_task_handler(self):
        running = True
        while running:
            try:
                if self.__should_block_frames:
                    logger.trace(f"{self}: frame processing paused")
                    await self.__input_event.wait()
                    self.__input_event.clear()
                    logger.trace(f"{self}: frame processing resumed")

                (frame, direction, callback) = await self.__input_queue.get()

                # Process the frame.
                await self.process_frame(frame, direction)

                # If this frame has an associated callback, call it now.
                if callback:
                    await callback(self, frame, direction)

                running = not isinstance(frame, EndFrame)

                self.__input_queue.task_done()
            except asyncio.CancelledError:
                logger.trace(f"{self}: cancelled input task")
                break
            except Exception as e:
                logger.exception(f"{self}: Uncaught exception {e}")
                await self.push_error(ErrorFrame(str(e)))

    def __create_push_task(self):
        self.__push_queue = asyncio.Queue()
        self.__push_frame_task = self.get_event_loop().create_task(self.__push_frame_task_handler())

    async def __cancel_push_task(self):
        self.__push_frame_task.cancel()
        await self.__push_frame_task

    async def __push_frame_task_handler(self):
        running = True
        while running:
            try:
                (frame, direction) = await self.__push_queue.get()
                await self.__internal_push_frame(frame, direction)
                running = not isinstance(frame, EndFrame)
                self.__push_queue.task_done()
            except asyncio.CancelledError:
                logger.trace(f"{self}: cancelled push task")
                break
            except Exception as e:
                logger.exception(f"{self}: Uncaught exception {e}")
                await self.push_error(ErrorFrame(str(e)))

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    await handler(self, *args, **kwargs)
                else:
                    handler(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in event handler {event_name}: {e}")

    def __str__(self):
        return self.name
