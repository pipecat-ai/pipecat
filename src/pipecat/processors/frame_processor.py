#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processing pipeline infrastructure for Pipecat.

This module provides the core frame processing system that enables building
audio/video processing pipelines. It includes frame processors, pipeline
management, and frame flow control mechanisms.
"""

from __future__ import annotations

import asyncio
import dataclasses
import traceback
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Optional,
)

from loguru import logger

from pipecat.clocks.base_clock import BaseClock
from pipecat.frames.frames import (
    CancelFrame,
    ErrorFrame,
    Frame,
    FrameProcessorPauseFrame,
    FrameProcessorPauseUrgentFrame,
    FrameProcessorResumeFrame,
    FrameProcessorResumeUrgentFrame,
    InterruptionFrame,
    StartFrame,
    SystemFrame,
    UninterruptibleFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject
from pipecat.utils.frame_queue import FrameQueue


class FrameDirection(Enum):
    """Direction of frame flow in the processing pipeline.

    Parameters:
        DOWNSTREAM: Frames flowing from input to output.
        UPSTREAM: Frames flowing back from output to input.
    """

    DOWNSTREAM = 1
    UPSTREAM = 2


FrameCallback = Callable[["FrameProcessor", Frame, FrameDirection], Awaitable[None]]


@dataclass
class FrameProcessorSetup:
    """Configuration parameters for frame processor initialization.

    Parameters:
        clock: The clock instance for timing operations.
        task_manager: The task manager for handling async operations.
        observer: Optional observer for monitoring frame processing events.
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    observer: BaseObserver | None = None


class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for systems frames and other frames.

    This is a specialized queue for frame processors that separates and
    prioritizes system frames over other frames. It ensures that `SystemFrame`
    objects are processed before any other frames by using a priority queue.

    """

    HIGH_PRIORITY = 1
    LOW_PRIORITY = 2

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()
        self.__high_counter = 0
        self.__low_counter = 0

    async def put(self, item: tuple[Frame, FrameDirection, FrameCallback]):
        """Put an item into the priority queue.

        System frames (`SystemFrame`) have higher priority than any other
        frames. If a non-frame item (e.g. a watchdog cancellation sentinel) is
        provided it will have the highest priority.

        Args:
            item (Any): The item to enqueue.

        """
        frame, _, _ = item
        if isinstance(frame, SystemFrame):
            self.__high_counter += 1
            await super().put((self.HIGH_PRIORITY, self.__high_counter, item))
        else:
            self.__low_counter += 1
            await super().put((self.LOW_PRIORITY, self.__low_counter, item))

    async def get(self) -> Any:
        """Retrieve the next item from the queue.

        System frames are prioritized. If both queues are empty, this method
        waits until an item is available.

        Returns:
            Any: The next item from the system or main queue.

        """
        _, _, item = await super().get()
        return item


# Timeout in seconds for cancelling the input frame processing task.
# This prevents hanging if a library swallows asyncio.CancelledError.
INPUT_TASK_CANCEL_TIMEOUT_SECS = 3


class FrameProcessor(BaseObject):
    """Base class for all frame processors in the pipeline.

    Frame processors are the building blocks of Pipecat pipelines, they can be
    linked to form complex processing pipelines. They receive frames, process
    them, and pass them to the next or previous processor in the chain.  Each
    frame processor guarantees frame ordering and processes frames in its own
    task. System frames are also processed in a separate task which guarantees
    frame priority.

    Event handlers available:

    - on_before_process_frame: Called before a frame is processed
    - on_after_process_frame: Called after a frame is processed
    - on_before_push_frame: Called before a frame is pushed
    - on_after_push_frame: Called after a frame is pushed
    - on_error: Called when an error is raised in the frame processing.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        enable_direct_mode: bool = False,
        metrics: FrameProcessorMetrics | None = None,
        **kwargs,
    ):
        """Initialize the frame processor.

        Args:
            name: Optional name for this processor instance.
            enable_direct_mode: Whether to process frames immediately or use internal queues.
            metrics: Optional metrics collector for this processor.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(name=name, **kwargs)
        self._prev: FrameProcessor | None = None
        self._next: FrameProcessor | None = None

        # Enable direct mode to skip queues and process frames right away.
        self._enable_direct_mode = enable_direct_mode

        # Clock
        self._clock: BaseClock | None = None

        # Task Manager
        self._task_manager: BaseTaskManager | None = None

        # Observer
        self._observer: BaseObserver | None = None

        # Other properties
        self._enable_metrics = False
        self._enable_usage_metrics = False
        self._report_only_initial_ttfb = False

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
        self.__should_block_system_frames = False
        self.__input_queue = FrameProcessorQueue()
        self.__input_event: asyncio.Event | None = None
        self.__input_frame_task: asyncio.Task | None = None

        # The process task processes non-system frames.  Non-system frames will
        # be processed as soon as they are received by the processing task
        # (default) or they will block if `pause_processing_frames()` is
        # called. To resume processing frames we need to call
        # `resume_processing_frames()` which will wake up the event.
        self.__should_block_frames = False
        self.__process_queue = FrameQueue(frame_getter=lambda item: item[0])
        self.__process_event: asyncio.Event | None = None
        self.__process_frame_task: asyncio.Task | None = None
        self.__process_current_frame: Frame | None = None

        # Frame processor events.
        self._register_event_handler("on_before_process_frame", sync=True)
        self._register_event_handler("on_after_process_frame", sync=True)
        self._register_event_handler("on_before_push_frame", sync=True)
        self._register_event_handler("on_after_push_frame", sync=True)
        self._register_event_handler("on_error", sync=True)

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
    def processors(self) -> list[FrameProcessor]:
        """Return the list of sub-processors contained within this processor.

        Only compound processors (e.g. pipelines and parallel pipelines) have
        sub-processors. Non-compound processors will return an empty list.

        Returns:
            The list of sub-processors if this is a compound processor.
        """
        return []

    @property
    def entry_processors(self) -> list[FrameProcessor]:
        """Return the list of entry processors for this processor.

        Entry processors are the first processors in a compound processor
        (e.g. pipelines, parallel pipelines). Note that pipelines can also be an
        entry processor as pipelines are processors themselves. Non-compound
        processors will simply return an empty list.

        Returns:
            The list of entry processors.
        """
        return []

    @property
    def next(self) -> FrameProcessor | None:
        """Get the next processor.

        Returns:
            The next processor, or None if there's no next processor.
        """
        return self._next

    @property
    def previous(self) -> FrameProcessor | None:
        """Get the previous processor.

        Returns:
            The previous processor, or None if there's no previous processor.
        """
        return self._prev

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

    def processors_with_metrics(self):
        """Return processors that can generate metrics.

        Recursively collects all processors that support metrics generation,
        including those from nested processors.

        Returns:
            List of frame processors that can generate metrics.
        """
        return []

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

    async def start_ttfb_metrics(self, *, start_time: float | None = None):
        """Start time-to-first-byte metrics collection.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_ttfb_metrics(
                start_time=start_time, report_only_initial_ttfb=self._report_only_initial_ttfb
            )

    async def stop_ttfb_metrics(self, *, end_time: float | None = None):
        """Stop time-to-first-byte metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfb_metrics(end_time=end_time)
            if frame:
                await self.push_frame(frame)

    async def start_processing_metrics(self, *, start_time: float | None = None):
        """Start processing metrics collection.

        Args:
            start_time: Optional timestamp to use as the start time. If None,
                uses the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_processing_metrics(start_time=start_time)

    async def stop_processing_metrics(self, *, end_time: float | None = None):
        """Stop processing metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_processing_metrics(end_time=end_time)
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

    async def start_text_aggregation_metrics(self):
        """Start text aggregation time metrics collection."""
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_text_aggregation_metrics()

    async def stop_text_aggregation_metrics(self):
        """Stop text aggregation time metrics collection and push results."""
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_text_aggregation_metrics()
            if frame:
                await self.push_frame(frame)

    async def stop_all_metrics(self):
        """Stop all active metrics collection."""
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()
        await self.stop_text_aggregation_metrics()

    def create_task(self, coroutine: Coroutine, name: str | None = None) -> asyncio.Task:
        """Create a new task managed by this processor.

        Args:
            coroutine: The coroutine to run in the task.
            name: Optional name for the task.

        Returns:
            The created asyncio task.
        """
        if name:
            name = f"{self}::{name}"
        else:
            name = f"{self}::{coroutine.cr_code.co_name}"
        return self.task_manager.create_task(coroutine, name)

    async def cancel_task(self, task: asyncio.Task, timeout: float | None = 1.0):
        """Cancel a task managed by this processor.

        A default timeout if 1 second is used in order to avoid potential
        freezes caused by certain libraries that swallow
        `asyncio.CancelledError`.

        Args:
            task: The task to cancel.
            timeout: Optional timeout for task cancellation.
        """
        await self.task_manager.cancel_task(task, timeout)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._clock = setup.clock
        self._task_manager = setup.task_manager
        self._observer = setup.observer

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

    def link(self, processor: FrameProcessor):
        """Link this processor to the next processor in the pipeline.

        Args:
            processor: The processor to link to.
        """
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

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

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop used by this processor.

        Returns:
            The asyncio event loop.
        """
        return self.task_manager.get_event_loop()

    async def queue_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
        callback: FrameCallback | None = None,
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

        if self._enable_direct_mode:
            await self.__process_frame(frame, direction, callback)
        else:
            await self.__input_queue.put((frame, direction, callback))

    async def pause_processing_frames(self):
        """Pause processing of queued frames."""
        logger.trace(f"{self}: pausing frame processing")
        self.__should_block_frames = True
        if self.__process_event:
            self.__process_event.clear()

    async def pause_processing_system_frames(self):
        """Pause processing of queued system frames."""
        logger.trace(f"{self}: pausing system frame processing")
        self.__should_block_system_frames = True
        if self.__input_event:
            self.__input_event.clear()

    async def resume_processing_frames(self):
        """Resume processing of queued frames."""
        logger.trace(f"{self}: resuming frame processing")
        if self.__process_event:
            self.__process_event.set()

    async def resume_processing_system_frames(self):
        """Resume processing of queued system frames."""
        logger.trace(f"{self}: resuming system frame processing")
        if self.__input_event:
            self.__input_event.set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        if self._observer:
            timestamp = self._clock.get_time() if self._clock else 0
            data = FrameProcessed(
                processor=self,
                frame=frame,
                direction=direction,
                timestamp=timestamp,
            )
            await self._observer.on_process_frame(data)

        if isinstance(frame, StartFrame):
            await self.__start(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._start_interruption()
            await self.stop_all_metrics()
        elif isinstance(frame, CancelFrame):
            await self.__cancel(frame)
        elif isinstance(frame, (FrameProcessorPauseFrame, FrameProcessorPauseUrgentFrame)):
            await self.__pause(frame)
        elif isinstance(frame, (FrameProcessorResumeFrame, FrameProcessorResumeUrgentFrame)):
            await self.__resume(frame)

    async def push_error(
        self,
        error_msg: str,
        exception: Exception | None = None,
        fatal: bool = False,
    ):
        """Creates and pushes an ErrorFrame upstream.

        Creates and pushes an ErrorFrame upstream to notify other processors in the
        pipeline about an error condition. The error frame will include context about
        which processor generated the error.

        Args:
            error_msg: Descriptive message explaining the error condition.
            exception: Optional exception object that caused the error, if available.
                This provides additional context for debugging and error handling.
            fatal: Whether this error should be considered fatal to the pipeline.
                Fatal errors typically cause the entire pipeline to stop processing.
                Defaults to False for non-fatal errors.

        Example::

            ```python
            # Non-fatal error
            await self.push_error("Failed to process audio chunk, skipping")

            # Fatal error with exception context
            try:
                result = some_critical_operation()
            except Exception as e:
                await self.push_error("Critical operation failed", exception=e, fatal=True)
            ```
        """
        error_frame = ErrorFrame(error=error_msg, fatal=fatal, exception=exception, processor=self)
        await self.push_error_frame(error=error_frame)

    async def push_error_frame(self, error: ErrorFrame):
        """Push an error frame upstream.

        Args:
            error: The error frame to push.
        """
        if not error.processor:
            error.processor = self
        await self._call_event_handler("on_error", error)

        if error.exception:
            tb = traceback.extract_tb(error.exception.__traceback__)
            last = tb[-1]
            error_message = (
                f"{error.processor} exception ({last.filename}:{last.lineno}): {error.error}"
            )
        else:
            error_message = f"{error.processor} error: {error.error}"

        logger.error(error_message)
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame to the next processor in the pipeline.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        if not self._check_started(frame):
            return

        await self._call_event_handler("on_before_push_frame", frame)

        await self.__internal_push_frame(frame, direction)

        await self._call_event_handler("on_after_push_frame", frame)

    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        self.__reset_process_task()
        await self.stop_all_metrics()
        await self.broadcast_frame(InterruptionFrame)

    async def push_interruption_task_frame_and_wait(self, *, timeout: float = 5.0):
        """Push an interruption task frame upstream and wait for the interruption.

        .. deprecated:: 0.0.104
            Use :meth:`broadcast_interruption` instead. This method now
            delegates to ``broadcast_interruption()`` and ignores *timeout*.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`FrameProcessor.push_interruption_task_frame_and_wait()` is deprecated. "
                "Use `FrameProcessor.broadcast_interruption()` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        await self.broadcast_interruption()

    async def broadcast_frame(self, frame_cls: type[Frame], **kwargs):
        """Broadcasts a frame of the specified class upstream and downstream.

        This method creates two instances of the given frame class using the
        provided keyword arguments (without deep-copying them) and pushes them
        upstream and downstream.

        Args:
            frame_cls: The class of the frame to be broadcasted.
            **kwargs: Keyword arguments to be passed to the frame's constructor.
        """
        downstream_frame = frame_cls(**kwargs)
        upstream_frame = frame_cls(**kwargs)
        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

    async def broadcast_frame_instance(self, frame: Frame):
        """Broadcasts a frame instance upstream and downstream.

        This method creates two new frame instances shallow-copying all fields
        from the original frame except `id` and `name`, which get fresh values.

        Args:
            frame: The frame instance to broadcast.

        Note:
            Prefer using `broadcast_frame()` when possible, as it is more
            efficient. This method should only be used when you are not the
            creator of the frame and need to broadcast an existing instance.
        """
        frame_cls = type(frame)
        init_fields = {f.name: getattr(frame, f.name) for f in dataclasses.fields(frame) if f.init}
        extra_fields = {
            f.name: getattr(frame, f.name)
            for f in dataclasses.fields(frame)
            if not f.init and f.name not in ("id", "name")
        }

        downstream_frame = frame_cls(**init_fields)
        for k, v in extra_fields.items():
            setattr(downstream_frame, k, v)

        upstream_frame = frame_cls(**init_fields)
        for k, v in extra_fields.items():
            setattr(upstream_frame, k, v)

        downstream_frame.broadcast_sibling_id = upstream_frame.id
        upstream_frame.broadcast_sibling_id = downstream_frame.id
        await self.push_frame(downstream_frame)
        await self.push_frame(upstream_frame, FrameDirection.UPSTREAM)

    async def __start(self, frame: StartFrame):
        """Handle the start frame to initialize processor state.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self.__started = True
        self._enable_metrics = frame.enable_metrics
        self._enable_usage_metrics = frame.enable_usage_metrics
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
            current_is_uninterruptible = isinstance(
                self.__process_current_frame, UninterruptibleFrame
            )
            if current_is_uninterruptible or self.__process_queue.has_uninterruptible:
                # We don't want to cancel an UninterruptibleFrame (either the
                # one currently being processed or one waiting in the queue),
                # so we simply cleanup the queue keeping only
                # UninterruptibleFrames.
                self.__reset_process_queue()
            else:
                # Cancel and re-create the process task.
                await self.__cancel_process_task()
                self.__create_process_task()
        except Exception as e:
            await self.push_error(
                error_msg=f"Uncaught exception handling _start_interruption: {e}",
                exception=e,
            )

    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        """Internal method to push frames to adjacent processors.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        try:
            timestamp = self._clock.get_time() if self._clock else 0
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")

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
            await self.push_error(error_msg=f"Uncaught exception: {e}", exception=e)

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
        if self._enable_direct_mode:
            return

        if not self.__input_frame_task:
            self.__input_event = asyncio.Event()
            self.__input_frame_task = self.create_task(self.__input_frame_task_handler())

    async def __cancel_input_task(self):
        """Cancel the frame input processing task."""
        if self.__input_frame_task:
            # Apply a timeout as a safeguard: if a library swallows asyncio.CancelledError,
            # the task would otherwise never be cancelled. With a timeout, we can detect this
            # situation and surface it in the logs instead of hanging indefinitely.
            await self.cancel_task(self.__input_frame_task, INPUT_TASK_CANCEL_TIMEOUT_SECS)
            self.__input_frame_task = None

    def __create_process_task(self):
        """Create the non-system frame processing task."""
        if self._enable_direct_mode:
            return

        if not self.__process_frame_task:
            self.__reset_process_task()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())

    def __reset_process_task(self):
        """Reset non-system frame processing task."""
        if self._enable_direct_mode:
            return

        self.__should_block_frames = False
        self.__process_event = asyncio.Event()
        self.__reset_process_queue()

    def __reset_process_queue(self):
        """Reset non-system frame processing queue."""
        self.__process_queue.reset()

    def has_queued_frame(self, frame_type: type[Frame] | type[UninterruptibleFrame]) -> bool:
        """Return True if a frame of the given type is waiting in the processing queue.

        Delegates to :meth:`FrameQueue.has_frame` so the check is O(distinct
        enqueued types) with no queue scanning.  ``frame_type`` may be any
        ``Frame`` subclass or ``UninterruptibleFrame`` (a mixin).

        Args:
            frame_type: The frame class (or mixin) to look for.

        Returns:
            True if at least one matching frame is queued.
        """
        return self.__process_queue.has_frame(frame_type)

    async def __cancel_process_task(self):
        """Cancel the non-system frame processing task."""
        if self.__process_frame_task:
            await self.cancel_task(self.__process_frame_task)
            self.__process_frame_task = None

    async def __process_frame(
        self, frame: Frame, direction: FrameDirection, callback: FrameCallback | None
    ):
        try:
            await self._call_event_handler("on_before_process_frame", frame)

            # Process the frame.
            await self.process_frame(frame, direction)
            # If this frame has an associated callback, call it now.
            if callback:
                await callback(self, frame, direction)

            await self._call_event_handler("on_after_process_frame", frame)
        except Exception as e:
            await self.push_error(error_msg=f"Error processing frame: {e}", exception=e)

    async def __input_frame_task_handler(self):
        """Handle frames from the input queue.

        It only processes system frames. Other frames are queue for another task
        to execute.

        """
        while True:
            (frame, direction, callback) = await self.__input_queue.get()

            if self.__should_block_system_frames and self.__input_event:
                logger.trace(f"{self}: system frame processing paused")
                await self.__input_event.wait()
                self.__input_event.clear()
                self.__should_block_system_frames = False
                logger.trace(f"{self}: system frame processing resumed")

            if isinstance(frame, SystemFrame):
                await self.__process_frame(frame, direction, callback)
            elif self.__process_queue:
                await self.__process_queue.put((frame, direction, callback))
            else:
                raise RuntimeError(
                    f"{self}: __process_queue is None when processing frame {frame.name}"
                )

            self.__input_queue.task_done()

    async def __process_frame_task_handler(self):
        """Handle non-system frames from the process queue."""
        while True:
            self.__process_current_frame = None

            (frame, direction, callback) = await self.__process_queue.get()

            self.__process_current_frame = frame

            if self.__should_block_frames and self.__process_event:
                logger.trace(f"{self}: frame processing paused")
                await self.__process_event.wait()
                self.__process_event.clear()
                self.__should_block_frames = False
                logger.trace(f"{self}: frame processing resumed")

            await self.__process_frame(frame, direction, callback)

            self.__process_queue.task_done()
