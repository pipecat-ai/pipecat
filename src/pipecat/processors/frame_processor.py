#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processing pipeline infrastructure for Pipecat.

This module provides the core frame processing system that enables building
audio/video processing pipelines. It includes frame processors, pipeline
management, and frame flow control mechanisms.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Coroutine, List, Optional, Sequence

from loguru import logger

from pipecat.audio.interruptions.base_interruption_strategy import BaseInterruptionStrategy
from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    FrameProcessorPauseFrame,
    FrameProcessorPauseUrgentFrame,
    FrameProcessorResumeFrame,
    FrameProcessorResumeUrgentFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.asyncio.watchdog_event import WatchdogEvent
from pipecat.utils.asyncio.watchdog_queue import WatchdogQueue
from pipecat.utils.base_object import BaseObject


class FrameDirection(Enum):
    """Direction of frame flow in the processing pipeline.

    Parameters:
        DOWNSTREAM: Frames flowing from input to output.
        UPSTREAM: Frames flowing back from output to input.
    """

    DOWNSTREAM = 1
    UPSTREAM = 2


@dataclass
class FrameProcessorSetup:
    """Configuration parameters for frame processor initialization.

    Parameters:
        clock: The clock instance for timing operations.
        task_manager: The task manager for handling async operations.
        observer: Optional observer for monitoring frame processing events.
        watchdog_timers_enabled: Whether to enable watchdog timers by default.
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    observer: Optional[BaseObserver] = None
    watchdog_timers_enabled: bool = False


class FrameProcessorQueue(WatchdogQueue):
    """A priority queue for systems frames and other frames.

    This is a specialized queue for frame processors that separates and
    prioritizes system frames over other frames.

    This queue uses two internal `WatchdogQueue` instances:
    - One for system-level frames (`SystemFrame`)
    - One for regular frames

    It ensures that `SystemFrame` objects are processed before any other
    frames. Additionally, it uses an `asyncio.Event` to signal when new items
    have been added to either queue, allowing consumers to wait efficiently when
    the queue is empty.

    """

    def __init__(self, manager: BaseTaskManager):
        """Initialize the FrameProcessorQueue.

        Args:
            manager (BaseTaskManager): The task manager used by the internal watchdog queues.

        """
        super().__init__(manager)
        self.__event = WatchdogEvent(manager)
        self.__main_queue = WatchdogQueue(manager)
        self.__system_queue = WatchdogQueue(manager)

    async def put(self, item: Any):
        """Put an item into the appropriate queue.

        System frames (`SystemFrame`) are placed into the system queue and all others
        into the regular queue. Signals the event to wake up any waiting consumers.

        Args:
            item (Any): The item to enqueue.

        """
        if isinstance(item, SystemFrame):
            await self.__system_queue.put(item)
        else:
            await self.__main_queue.put(item)
        self.__event.set()

    async def get(self) -> Any:
        """Retrieve the next item from the queue.

        System frames are prioritized. If both queues are empty, this method
        waits until an item is available.

        Returns:
            Any: The next item from the system or main queue.

        """
        # Wait for an item in any of the queues if they are empty.
        if self.__main_queue.empty() and self.__system_queue.empty():
            await self.__event.wait()

        # Prioritize system frames.
        if self.__system_queue.qsize() > 0:
            item = await self.__system_queue.get()
            self.__system_queue.task_done()
        else:
            item = await self.__main_queue.get()
            self.__main_queue.task_done()

        # Clear the event only if all queues are empty.
        if self.__main_queue.empty() and self.__system_queue.empty():
            self.__event.clear()

        return item

    def cancel(self):
        """Cancel both internal queues.

        This method is used to stop processing and release any pending tasks
        in both the system and main queues. Typically used during shutdown
        or cleanup to prevent further processing of frames.

        """
        self.__main_queue.cancel()
        self.__system_queue.cancel()


FrameCallback = Callable[["FrameProcessor", Frame, FrameDirection], Awaitable[None]]


class FrameProcessor(BaseObject):
    """Base class for all frame processors in the pipeline.

    Frame processors are the building blocks of Pipecat pipelines, they can be
    linked to form complex processing pipelines. They receive frames, process
    them, and pass them to the next or previous processor in the chain.  Each
    frame processor guarantees frame ordering and processes frames in its own
    task. System frames are also processed in a separate task which guarantees
    frame priority.

    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        enable_watchdog_logging: Optional[bool] = None,
        enable_watchdog_timers: Optional[bool] = None,
        metrics: Optional[FrameProcessorMetrics] = None,
        watchdog_timeout_secs: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the frame processor.

        Args:
            name: Optional name for this processor instance.
            enable_watchdog_logging: Whether to enable watchdog logging for tasks.
            enable_watchdog_timers: Whether to enable watchdog timers for tasks.
            metrics: Optional metrics collector for this processor.
            watchdog_timeout_secs: Timeout in seconds for watchdog operations.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(name=name)
        self._parent: Optional["FrameProcessor"] = None
        self._prev: Optional["FrameProcessor"] = None
        self._next: Optional["FrameProcessor"] = None

        # Enable watchdog timers for all tasks created by this frame processor.
        self._enable_watchdog_timers = enable_watchdog_timers

        # Enable watchdog logging for all tasks created by this frame processor.
        self._enable_watchdog_logging = enable_watchdog_logging

        # Allow this frame processor to control their tasks timeout.
        self._watchdog_timeout_secs = watchdog_timeout_secs

        # Clock
        self._clock: Optional[BaseClock] = None

        # Task Manager
        self._task_manager: Optional[BaseTaskManager] = None

        # Observer
        self._observer: Optional[BaseObserver] = None

        # Other properties
        self._allow_interruptions = False
        self._enable_metrics = False
        self._enable_usage_metrics = False
        self._report_only_initial_ttfb = False
        self._interruption_strategies: List[BaseInterruptionStrategy] = []

        # Indicates whether we have received the StartFrame.
        self.__started = False

        # Cancellation is done through CancelFrame (a system frame). This could
        # cause other events being triggered (e.g. closing a transport) which
        # could also cause other frames to be pushed from other tasks
        # (e.g. EndFrame). So, when we are cancelling we don't want anything
        # else to be pushed.
        self._cancelling = False

        # Metrics
        self._metrics = metrics or FrameProcessorMetrics()
        self._metrics.set_processor_name(self.name)

        # Processors have an input priority queue which stores any type of
        # frames in order. System frames have higher priority than any other
        # frames, so they will be returned first from the queue.
        #
        # If a system frame is obtained it will be processed immediately any
        # other type of frame (data and control) will be put in a separate queue
        # for later processing. This guarantees that each frame processor will
        # always process system frames before any other frame in the queue.

        # The input task that handles all types of frames. It processes system
        # frames right away and queues non-system frames for later processing.
        self.__input_frame_task: Optional[asyncio.Task] = None

        # The process task processes non-system frames.  Non-system frames will
        # be processed as soon as they are received by the processing task
        # (default) or they will block if `pause_processing_frames()` is
        # called. To resume processing frames we need to call
        # `resume_processing_frames()` which will wake up the event.
        self.__should_block_frames = False
        self.__process_event = None
        self.__process_frame_task: Optional[asyncio.Task] = None
        self.__process_queue = None

    @property
    def id(self) -> int:
        """Get the unique identifier for this processor.

        Returns:
            The unique integer ID of this processor.
        """
        return self._id

    @property
    def name(self) -> str:
        """Get the name of this processor.

        Returns:
            The name of this processor instance.
        """
        return self._name

    @property
    def interruptions_allowed(self):
        """Check if interruptions are allowed for this processor.

        Returns:
            True if interruptions are allowed.
        """
        return self._allow_interruptions

    @property
    def metrics_enabled(self):
        """Check if metrics collection is enabled.

        Returns:
            True if metrics collection is enabled.
        """
        return self._enable_metrics

    @property
    def usage_metrics_enabled(self):
        """Check if usage metrics collection is enabled.

        Returns:
            True if usage metrics collection is enabled.
        """
        return self._enable_usage_metrics

    @property
    def report_only_initial_ttfb(self):
        """Check if only initial TTFB should be reported.

        Returns:
            True if only initial time-to-first-byte should be reported.
        """
        return self._report_only_initial_ttfb

    @property
    def interruption_strategies(self) -> Sequence[BaseInterruptionStrategy]:
        """Get the interruption strategies for this processor.

        Returns:
            Sequence of interruption strategies.
        """
        return self._interruption_strategies

    @property
    def task_manager(self) -> BaseTaskManager:
        """Get the task manager for this processor.

        Returns:
            The task manager instance.

        Raises:
            Exception: If the task manager is not initialized.
        """
        if not self._task_manager:
            raise Exception(f"{self} TaskManager is still not initialized.")
        return self._task_manager

    def can_generate_metrics(self) -> bool:
        """Check if this processor can generate metrics.

        Returns:
            True if this processor can generate metrics.
        """
        return False

    def set_core_metrics_data(self, data: MetricsData):
        """Set core metrics data for this processor.

        Args:
            data: The metrics data to set.
        """
        self._metrics.set_core_metrics_data(data)

    async def start_ttfb_metrics(self):
        """Start time-to-first-byte metrics collection."""
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_ttfb_metrics(self._report_only_initial_ttfb)

    async def stop_ttfb_metrics(self):
        """Stop time-to-first-byte metrics collection and push results."""
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfb_metrics()
            if frame:
                await self.push_frame(frame)

    async def start_processing_metrics(self):
        """Start processing metrics collection."""
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_processing_metrics()

    async def stop_processing_metrics(self):
        """Stop processing metrics collection and push results."""
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_processing_metrics()
            if frame:
                await self.push_frame(frame)

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Start LLM usage metrics collection.

        Args:
            tokens: Token usage information for the LLM.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_llm_usage_metrics(tokens)
            if frame:
                await self.push_frame(frame)

    async def start_tts_usage_metrics(self, text: str):
        """Start TTS usage metrics collection.

        Args:
            text: The text being processed by TTS.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_tts_usage_metrics(text)
            if frame:
                await self.push_frame(frame)

    async def stop_all_metrics(self):
        """Stop all active metrics collection."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    def create_task(
        self,
        coroutine: Coroutine,
        name: Optional[str] = None,
        *,
        enable_watchdog_logging: Optional[bool] = None,
        enable_watchdog_timers: Optional[bool] = None,
        watchdog_timeout_secs: Optional[float] = None,
    ) -> asyncio.Task:
        """Create a new task managed by this processor.

        Args:
            coroutine: The coroutine to run in the task.
            name: Optional name for the task.
            enable_watchdog_logging: Whether to enable watchdog logging.
            enable_watchdog_timers: Whether to enable watchdog timers.
            watchdog_timeout_secs: Timeout in seconds for watchdog operations.

        Returns:
            The created asyncio task.
        """
        if name:
            name = f"{self}::{name}"
        else:
            name = f"{self}::{coroutine.cr_code.co_name}"
        return self.task_manager.create_task(
            coroutine,
            name,
            enable_watchdog_logging=(
                enable_watchdog_logging
                if enable_watchdog_logging
                else self._enable_watchdog_logging
            ),
            enable_watchdog_timers=(
                enable_watchdog_timers if enable_watchdog_timers else self._enable_watchdog_timers
            ),
            watchdog_timeout=(
                watchdog_timeout_secs if watchdog_timeout_secs else self._watchdog_timeout_secs
            ),
        )

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Cancel a task managed by this processor.

        Args:
            task: The task to cancel.
            timeout: Optional timeout for task cancellation.
        """
        await self.task_manager.cancel_task(task, timeout)

    async def wait_for_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Wait for a task to complete.

        Args:
            task: The task to wait for.
            timeout: Optional timeout for waiting.
        """
        await self.task_manager.wait_for_task(task, timeout)

    def reset_watchdog(self):
        """Reset the watchdog timer for the current task."""
        self.task_manager.task_reset_watchdog()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._clock = setup.clock
        self._task_manager = setup.task_manager
        self._observer = setup.observer
        self._watchdog_timers_enabled = (
            self._enable_watchdog_timers
            if self._enable_watchdog_timers
            else setup.watchdog_timers_enabled
        )

        # Create processing tasks.
        self.__create_input_task()

        if self._metrics is not None:
            await self._metrics.setup(self._task_manager)

    async def cleanup(self):
        """Clean up processor resources."""
        await super().cleanup()
        await self.__cancel_input_task()
        await self.__cancel_process_task()
        if self._metrics is not None:
            await self._metrics.cleanup()

    def link(self, processor: "FrameProcessor"):
        """Link this processor to the next processor in the pipeline.

        Args:
            processor: The processor to link to.
        """
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this processor.

        Returns:
            The asyncio event loop.
        """
        return self.task_manager.get_event_loop()

    def set_parent(self, parent: "FrameProcessor"):
        """Set the parent processor for this processor.

        Args:
            parent: The parent processor.
        """
        self._parent = parent

    def get_parent(self) -> Optional["FrameProcessor"]:
        """Get the parent processor.

        Returns:
            The parent processor, or None if no parent is set.
        """
        return self._parent

    def get_clock(self) -> BaseClock:
        """Get the clock used by this processor.

        Returns:
            The clock instance.

        Raises:
            Exception: If the clock is not initialized.
        """
        if not self._clock:
            raise Exception(f"{self} Clock is still not initialized.")
        return self._clock

    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: Optional[FrameCallback] = None,
    ):
        """Queue a frame for processing.

        Args:
            frame: The frame to queue.
            direction: The direction of frame flow.
            callback: Optional callback to call after processing.
        """
        # If we are cancelling we don't want to process any other frame.
        if self._cancelling:
            return

        await self.__input_queue.put((frame, direction, callback))

    async def pause_processing_frames(self):
        """Pause processing of queued frames."""
        logger.trace(f"{self}: pausing frame processing")
        self.__should_block_frames = True

    async def resume_processing_frames(self):
        """Resume processing of queued frames."""
        logger.trace(f"{self}: resuming frame processing")
        if self.__process_event:
            self.__process_event.set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        if isinstance(frame, StartFrame):
            await self.__start(frame)
        elif isinstance(frame, StartInterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
        elif isinstance(frame, StopInterruptionFrame):
            self._should_report_ttfb = True
        elif isinstance(frame, CancelFrame):
            await self.__cancel(frame)
        elif isinstance(frame, (FrameProcessorPauseFrame, FrameProcessorPauseUrgentFrame)):
            await self.__pause(frame)
        elif isinstance(frame, (FrameProcessorResumeFrame, FrameProcessorResumeUrgentFrame)):
            await self.__resume(frame)

    async def push_error(self, error: ErrorFrame):
        """Push an error frame upstream.

        Args:
            error: The error frame to push.
        """
        if not error.processor:
            error.processor = self
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame to the next processor in the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        if not self._check_started(frame):
            return

        await self.__internal_push_frame(frame, direction)

    async def __start(self, frame: StartFrame):
        """Handle the start frame to initialize processor state.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self.__started = True
        self._allow_interruptions = frame.allow_interruptions
        self._enable_metrics = frame.enable_metrics
        self._enable_usage_metrics = frame.enable_usage_metrics
        self._interruption_strategies = frame.interruption_strategies
        self._report_only_initial_ttfb = frame.report_only_initial_ttfb

        self.__create_process_task()

    async def __cancel(self, frame: CancelFrame):
        """Handle the cancel frame to stop processor operation.

        Args:
            frame: The cancel frame.
        """
        self._cancelling = True
        await self.__cancel_process_task()

    async def __pause(self, frame: FrameProcessorPauseFrame | FrameProcessorPauseUrgentFrame):
        """Handle pause frame to pause processor operation.

        Args:
            frame: The pause frame.
        """
        if frame.processor.name == self.name:
            await self.pause_processing_frames()

    async def __resume(self, frame: FrameProcessorResumeFrame | FrameProcessorResumeUrgentFrame):
        """Handle resume frame to resume processor operation.

        Args:
            frame: The resume frame.
        """
        if frame.processor.name == self.name:
            await self.resume_processing_frames()

    #
    # Handle interruptions
    #

    async def _start_interruption(self):
        """Start handling an interruption by cancelling current tasks."""
        try:
            # Cancel the process task. This will stop processing queued frames.
            await self.__cancel_process_task()
        except Exception as e:
            logger.exception(f"Uncaught exception in {self} when handling _start_interruption: {e}")
            await self.push_error(ErrorFrame(str(e)))

        # Create a new process queue and task.
        self.__create_process_task()

    async def _stop_interruption(self):
        """Stop handling an interruption."""
        # Nothing to do right now.
        pass

    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        """Internal method to push frames to adjacent processors.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        try:
            timestamp = self._clock.get_time() if self._clock else 0
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} from {self} to {self._next}")

                if self._observer:
                    data = FramePushed(
                        source=self,
                        destination=self._next,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await self._observer.on_push_frame(data)
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                if self._observer:
                    data = FramePushed(
                        source=self,
                        destination=self._prev,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await self._observer.on_push_frame(data)
                await self._prev.queue_frame(frame, direction)
        except Exception as e:
            logger.exception(f"Uncaught exception in {self}: {e}")
            await self.push_error(ErrorFrame(str(e)))

    def _check_started(self, frame: Frame):
        """Check if the processor has been started.

        Args:
            frame: The frame being processed.

        Returns:
            True if the processor has been started.
        """
        if not self.__started:
            logger.error(f"{self} Trying to process {frame} but StartFrame not received yet")
        return self.__started

    def __create_input_task(self):
        """Create the frame input processing task."""
        if not self.__input_frame_task:
            self.__input_queue = FrameProcessorQueue(self.task_manager)
            self.__input_frame_task = self.create_task(self.__input_frame_task_handler())

    async def __cancel_input_task(self):
        """Cancel the frame input processing task."""
        if self.__input_frame_task:
            self.__input_queue.cancel()
            await self.cancel_task(self.__input_frame_task)
            self.__input_frame_task = None

    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if not self.__process_frame_task:
            self.__should_block_frames = False
            if not self.__process_event:
                self.__process_event = WatchdogEvent(self.task_manager)
            self.__process_event.clear()
            self.__process_queue = WatchdogQueue(self.task_manager)
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())

    async def __cancel_process_task(self):
        """Cancel the non-system frame processing task."""
        if self.__process_frame_task:
            self.__process_queue.cancel()
            await self.cancel_task(self.__process_frame_task)
            self.__process_frame_task = None

    async def __process_frame(
        self, frame: Frame, direction: FrameDirection, callback: FrameCallback
    ):
        try:
            # Process the frame.
            await self.process_frame(frame, direction)
            # If this frame has an associated callback, call it now.
            if callback:
                await callback(self, frame, direction)
        except Exception as e:
            logger.exception(f"{self}: error processing frame: {e}")
            await self.push_error(ErrorFrame(str(e)))

    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queue for another task
        to execute.

        """
        while True:
            (frame, direction, callback) = await self.__input_queue.get()

            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
            else:
                raise RuntimeError(
                    f"{self}: __process_queue is None when processing frame {frame.name}"
                )

    async def __process_frame_task_handler(self):
        """Handle non-system frames from the process queue."""
        while True:
            if self.__should_block_frames and self.__process_event:
                logger.trace(f"{self}: frame processing paused")
                await self.__process_event.wait()
                self.__process_event.clear()
                self.__should_block_frames = False
                logger.trace(f"{self}: frame processing resumed")

            (frame, direction, callback) = await self.__process_queue.get()

            await self.__process_frame(frame, direction, callback)
