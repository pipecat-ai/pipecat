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
from typing import Any, Awaitable, Callable, Coroutine, List, Optional, Sequence, Tuple

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
    InterruptionFrame,
    InterruptionTaskFrame,
    StartFrame,
    SystemFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


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
    observer: Optional[BaseObserver] = None


class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for systems frames and other frames.

    This is a specialized queue for frame processors that separates and
    prioritizes system frames over other frames. It ensures that `SystemFrame`
    objects are processed before any other frames by using a priority queue.

    """

    HIGH_PRIORITY = 1
    LOW_PRIORITY = 2

    def __init__(self):
        """Initialize the FrameProcessorQueue.

        Args:
            manager (BaseTaskManager): The task manager used by the internal watchdog queues.

        """
        super().__init__()
        self.__high_counter = 0
        self.__low_counter = 0

    async def put(self, item: Tuple[Frame, FrameDirection, FrameCallback]):
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
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        enable_direct_mode: bool = False,
        metrics: Optional[FrameProcessorMetrics] = None,
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
        self._prev: Optional["FrameProcessor"] = None
        self._next: Optional["FrameProcessor"] = None

        # Enable direct mode to skip queues and process frames right away.
        self._enable_direct_mode = enable_direct_mode

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
        self.__should_block_system_frames = False
        self.__input_event: Optional[asyncio.Event] = None
        self.__input_frame_task: Optional[asyncio.Task] = None

        # The process task processes non-system frames.  Non-system frames will
        # be processed as soon as they are received by the processing task
        # (default) or they will block if `pause_processing_frames()` is
        # called. To resume processing frames we need to call
        # `resume_processing_frames()` which will wake up the event.
        self.__should_block_frames = False
        self.__process_event: Optional[asyncio.Event] = None
        self.__process_frame_task: Optional[asyncio.Task] = None

        # To interrupt a pipeline, we push an `InterruptionTaskFrame` upstream.
        # Then we wait for the corresponding `InterruptionFrame` to travel from
        # the start of the pipeline back to the processor that sent the
        # `InterruptionTaskFrame`. This wait is handled using the following
        # event.
        self._wait_for_interruption = False
        self._wait_interruption_event = asyncio.Event()

        # Frame processor events.
        self._register_event_handler("on_before_process_frame", sync=True)
        self._register_event_handler("on_after_process_frame", sync=True)
        self._register_event_handler("on_before_push_frame", sync=True)
        self._register_event_handler("on_after_push_frame", sync=True)

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
    def processors(self) -> List["FrameProcessor"]:
        """Return the list of sub-processors contained within this processor.

        Only compound processors (e.g. pipelines and parallel pipelines) have
        sub-processors. Non-compound processors will return an empty list.

        Returns:
            The list of sub-processors if this is a compound processor.
        """
        return []

    @property
    def entry_processors(self) -> List["FrameProcessor"]:
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
    def next(self) -> Optional["FrameProcessor"]:
        """Get the next processor.

        Returns:
            The next processor, or None if there's no next processor.
        """
        return self._next

    @property
    def previous(self) -> Optional["FrameProcessor"]:
        """Get the previous processor.

        Returns:
            The previous processor, or None if there's no previous processor.
        """
        return self._prev

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

    def create_task(self, coroutine: Coroutine, name: Optional[str] = None) -> asyncio.Task:
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

    async def cancel_task(self, task: asyncio.Task, timeout: Optional[float] = 1.0):
        """Cancel a task managed by this processor.

        A default timeout if 1 second is used in order to avoid potential
        freezes caused by certain libraries that swallow
        `asyncio.CancelledError`.

        Args:
            task: The task to cancel.
            timeout: Optional timeout for task cancellation.
        """
        await self.task_manager.cancel_task(task, timeout)

    async def wait_for_task(self, task: asyncio.Task, timeout: Optional[float] = None):
        """Wait for a task to complete.

        .. deprecated:: 0.0.81
            This function is deprecated, use `await task` or
            `await asyncio.wait_for(task, timeout)` instead.

        Args:
            task: The task to wait for.
            timeout: Optional timeout for waiting.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`FrameProcessor.wait_for_task()` is deprecated. "
                "Use `await task` or `await asyncio.wait_for(task, timeout)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if timeout:
            await asyncio.wait_for(task, timeout)
        else:
            await task

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

    def link(self, processor: "FrameProcessor"):
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

        # If we are waiting for an interruption we will bypass all queued system
        # frames and we will process the frame right away. This is because a
        # previous system frame might be waiting for the interruption frame and
        # it's blocking the input task.
        if self._wait_for_interruption and isinstance(frame, InterruptionFrame):
            await self.__process_frame(frame, direction, callback)
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

        await self._call_event_handler("on_before_push_frame", frame)

        await self.__internal_push_frame(frame, direction)

        await self._call_event_handler("on_after_push_frame", frame)

        # If we are waiting for an interruption and we get an interruption, then
        # we can unblock `push_interruption_task_frame_and_wait()`.
        if self._wait_for_interruption and isinstance(frame, InterruptionFrame):
            self._wait_interruption_event.set()

    async def push_interruption_task_frame_and_wait(self):
        """Push an interruption task frame upstream and wait for the interruption.

        This function sends an `InterruptionTaskFrame` upstream to the pipeline
        task and waits to receive the corresponding `InterruptionFrame`. When
        the function finishes it is guaranteed that the `InterruptionFrame` has
        been pushed downstream.
        """
        self._wait_for_interruption = True

        await self.push_frame(InterruptionTaskFrame(), FrameDirection.UPSTREAM)

        # Wait for an `InterruptionFrame` to come to this processor and be
        # pushed. Take a look at `push_frame()` to see how we first push the
        # `InterruptionFrame` and then we set the event in order to maintain
        # frame ordering.
        await self._wait_interruption_event.wait()

        # Clean the event.
        self._wait_interruption_event.clear()

        self._wait_for_interruption = False

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
            if self._wait_for_interruption:
                # If we get here we know the process task was just waiting for
                # an interruption (push_interruption_task_frame_and_wait()), so
                # we can't cancel the task because it might still need to do
                # more things (e.g. pushing a frame after the
                # interruption). Instead we just drain the queue because this is
                # an interruption.
                self.__reset_process_task()
            else:
                # Cancel and re-create the process task including the queue.
                await self.__cancel_process_task()
                self.__create_process_task()
        except Exception as e:
            logger.exception(f"Uncaught exception in {self} when handling _start_interruption: {e}")
            await self.push_error(ErrorFrame(str(e)))

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
        if self._enable_direct_mode:
            return

        if not self.__input_frame_task:
            self.__input_event = asyncio.Event()
            self.__input_queue = FrameProcessorQueue()
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
            self.__should_block_frames = False
            self.__process_event = asyncio.Event()
            self.__process_queue = asyncio.Queue()
            self.__process_frame_task = self.create_task(self.__process_frame_task_handler())

    def __reset_process_task(self):
        """Reset non-system frame processing task."""
        if self._enable_direct_mode:
            return

        self.__should_block_frames = False
        self.__process_event = asyncio.Event()
        while not self.__process_queue.empty():
            self.__process_queue.get_nowait()
            self.__process_queue.task_done()

    async def __cancel_process_task(self):
        """Cancel the non-system frame processing task."""
        if self.__process_frame_task:
            await self.cancel_task(self.__process_frame_task)
            self.__process_frame_task = None

    async def __process_frame(
        self, frame: Frame, direction: FrameDirection, callback: Optional[FrameCallback]
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
            logger.exception(f"{self}: error processing frame: {e}")
            await self.push_error(ErrorFrame(str(e)))

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
            (frame, direction, callback) = await self.__process_queue.get()

            if self.__should_block_frames and self.__process_event:
                logger.trace(f"{self}: frame processing paused")
                await self.__process_event.wait()
                self.__process_event.clear()
                self.__should_block_frames = False
                logger.trace(f"{self}: frame processing resumed")

            await self.__process_frame(frame, direction, callback)

            self.__process_queue.task_done()
