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
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
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
    TTSAudioRawFrame,
    UninterruptibleFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, MetricsData, STTUsage
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject
from pipecat.utils.deprecation import deprecated, warn_deprecated_read
from pipecat.utils.errors import ErrorCategory, classify_http_exception
from pipecat.utils.frame_queue import FrameQueue

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.utils.tracing.tracing_context import TracingContext
    from pipecat.workers.runner import WorkerRunner


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
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_out_sample_rate: Output audio sample rate in Hz.
        clock: The clock instance for timing operations.
        enable_metrics: Whether to enable performance metrics collection.
        enable_tracing: Whether to enable OpenTelemetry tracing.
        enable_usage_metrics: Whether to enable usage metrics collection.
        pipeline_worker: The :class:`PipelineWorker` running this pipeline. Stored
            on each processor as ``self.pipeline_worker`` so processors can
            reach task-scoped state (e.g. ``self.pipeline_worker.app_resources``).
        observer: Optional observer for monitoring frame processing events.
        task_manager: The task manager for handling async operations.
        tool_resources: Deprecated. :class:`PipelineWorker` continues to populate
            this with ``app_resources`` so that custom :class:`FrameProcessor`
            subclasses whose ``setup()`` overrides read ``setup.tool_resources``
            keep working. New code should read
            ``setup.pipeline_worker.app_resources`` instead.

            .. deprecated:: 1.2.0
                Read ``setup.pipeline_worker.app_resources`` instead. Will be
                removed in 2.0.0.
        tracing_context: Pipeline-scoped tracing context for span hierarchy.
    """

    clock: BaseClock
    task_manager: BaseTaskManager
    pipeline_worker: PipelineWorker
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_metrics: bool = False
    enable_tracing: bool = False
    enable_usage_metrics: bool = False
    observer: BaseObserver | None = None
    report_only_initial_ttfb: bool = False
    tracing_context: TracingContext | None = None
    # Deprecated fields
    tool_resources: Any = None

    def __getattribute__(self, name: str) -> Any:
        # Warn when user code reads the deprecated ``tool_resources`` field.
        # Set is unaffected (goes through ``__setattr__``), so PipelineWorker can
        # populate it for backwards compat without tripping the warning.
        if name == "tool_resources":
            value = object.__getattribute__(self, "tool_resources")
            if value is not None:
                warn_deprecated_read(
                    "`FrameProcessorSetup.tool_resources` is deprecated since 1.2.0; "
                    "read `setup.pipeline_worker.app_resources` instead."
                )
            return value
        return object.__getattribute__(self, name)


class FrameProcessorQueue(asyncio.PriorityQueue):
    """A priority queue for the frames arriving at a frame processor.

    Frames are dequeued in three tiers: the `StartFrame` first, then
    `SystemFrame`, then data and control frames. Frames of the same tier keep
    their arrival order.

    """

    START_PRIORITY = 1
    SYSTEM_PRIORITY = 10
    DEFAULT_PRIORITY = 20

    def __init__(self):
        """Initialize the FrameProcessorQueue."""
        super().__init__()
        # Counts every frame enqueued, which keeps frames of the same tier in
        # arrival order and stops the queue from ever having to compare frames.
        self.__counter = 0

    async def put(self, item: tuple[Frame, FrameDirection, FrameCallback | None]):
        """Put an item into the priority queue.

        The `StartFrame` outranks every other frame and `SystemFrame` frames
        outrank data and control frames.

        Args:
            item: The frame to enqueue, with its direction and callback.

        """
        frame, _, _ = item
        if isinstance(frame, StartFrame):
            priority = self.START_PRIORITY
        elif isinstance(frame, SystemFrame):
            priority = self.SYSTEM_PRIORITY
        else:
            priority = self.DEFAULT_PRIORITY

        self.__counter += 1
        await super().put((priority, self.__counter, item))

    async def get(self) -> Any:
        """Retrieve the next item from the queue.

        Waits until an item is available.

        Returns:
            Any: The highest priority item in the queue.

        """
        _, _, item = await super().get()
        return item


# How long a processor holds frames waiting for a readiness condition before
# giving up and resuming. See pause_processing_all_frames_until().
PAUSE_UNTIL_READY_TIMEOUT_SECS = 5.0

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
    - on_usable_changed: Called with the new value of `is_usable` when the
      processor stops or starts being able to do its job.

    Example::

        @processor.event_handler("on_usable_changed")
        async def on_usable_changed(processor, is_usable):
            ...
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

        # Processor setup
        self._setup: FrameProcessorSetup | None = None

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
        # Watches the readiness condition passed to
        # pause_processing_all_frames_until() and lifts the pause it took.
        self.__pause_watcher_task: asyncio.Task | None = None

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

        # Whether this processor can still do its job. Flipped by the errors it
        # reports, so it is already up to date by the time an error travels.
        self._is_usable = True

        # Frame processor events.
        self._register_event_handler("on_before_process_frame", sync=True)
        self._register_event_handler("on_after_process_frame", sync=True)
        self._register_event_handler("on_before_push_frame", sync=True)
        self._register_event_handler("on_after_push_frame", sync=True)
        self._register_event_handler("on_error", sync=True)
        self._register_event_handler("on_usable_changed")

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
    def is_usable(self) -> bool:
        """Whether this processor can still do its job.

        A processor stays usable through failures it might recover from, and
        becomes unusable once its work can no longer succeed: a provider has
        rejected its API key, model or voice, or it has failed enough times to
        stop trying. Sending it more work would only produce more of the same
        error, so services stop accepting work and stop reconnecting once this
        is False.

        Errors set this as they are reported, so an error handler reading
        ``frame.processor.is_usable`` sees the verdict that came with the error
        it is handling.

        Returns:
            True while the processor can still be given work.
        """
        return self._is_usable

    async def set_usable(self, is_usable: bool):
        """Set whether this processor can be given work.

        Call this to bring back a processor that became unusable, once
        whatever stopped it working has been dealt with — new credentials, or
        a provider that has come back up. Services also do this for themselves
        when their settings change, since new settings may be the fix.

        Args:
            is_usable: Whether the processor can be given work.
        """
        if is_usable == self._is_usable:
            return

        self._is_usable = is_usable
        logger.debug(f"{self}: {'usable' if is_usable else 'no longer usable'}")
        await self._call_event_handler("on_usable_changed", is_usable)

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
    def processor_setup(self) -> FrameProcessorSetup:
        """Get the configuration this processor was set up with.

        Returns:
            The :class:`FrameProcessorSetup` given to :meth:`setup`.
        """
        if not self._setup:
            raise Exception(f"{self} is still not set up.")
        return self._setup

    @property
    def metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled.

        Returns:
            True if metrics collection is enabled.
        """
        return bool(self._setup and self._setup.enable_metrics)

    @property
    def usage_metrics_enabled(self) -> bool:
        """Check if usage metrics collection is enabled.

        Returns:
            True if usage metrics collection is enabled.
        """
        return bool(self._setup and self._setup.enable_usage_metrics)

    @property
    def report_only_initial_ttfb(self) -> bool:
        """Check if only initial TTFB should be reported.

        Returns:
            True if only initial time-to-first-byte should be reported.
        """
        return bool(self._setup and self._setup.report_only_initial_ttfb)

    @property
    def pipeline_worker(self) -> PipelineWorker:
        """Get the :class:`PipelineWorker` this processor is running in.

        Provides access to worker-scoped state from inside a processor — most
        notably ``self.pipeline_worker.app_resources`` for the application's
        shared bag of resources (DB handles, clients, feature flags, etc.).

        Returns:
            The :class:`PipelineWorker` instance that set up this processor.
        """
        return self.processor_setup.pipeline_worker

    @property
    def worker_runner(self) -> WorkerRunner:
        """Get the :class:`WorkerRunner` hosting this processor's worker.

        Use it to reach another worker on the runner by name, e.g.
        ``self.worker_runner.get_worker("ui-jobs")``.

        Returns:
            The runner this processor's :class:`PipelineWorker` is attached to.
        """
        return self.pipeline_worker.worker_runner

    @property
    @deprecated(
        "`FrameProcessor.pipeline_task` is deprecated since 1.3.0 and will be removed in 2.0.0. "
        "Use `pipeline_worker` instead."
    )
    def pipeline_task(self) -> PipelineWorker:
        """Deprecated alias for :attr:`pipeline_worker`.

        .. deprecated:: 1.3.0
            Use :attr:`pipeline_worker` instead. Will be removed in 2.0.0.
        """
        return self.processor_setup.pipeline_worker

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
                start_time=start_time, report_only_initial_ttfb=self.report_only_initial_ttfb
            )

    async def cancel_ttfb_metrics(self):
        """Abandon the current time-to-first-byte measurement without reporting it."""
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.cancel_ttfb_metrics()

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

    async def process_ttfa_metrics(self, frame: TTSAudioRawFrame):
        """Scan a TTS audio frame for the first audible sample and push TTFA.

        Should be called for every audio frame until a measurement is produced;
        the metrics collector tracks leading silence across chunks internally.

        Args:
            frame: The TTS audio frame to inspect.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            metrics_frame = await self._metrics.process_ttfa_metrics(
                audio=frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            if metrics_frame:
                await self.push_frame(metrics_frame)

    async def stop_ttfat_metrics(self, *, end_time: float | None = None):
        """Stop time-to-first-answer-token metrics collection and push results.

        Args:
            end_time: Optional timestamp to use as the end time. If None, uses
                the current time.
        """
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfat_metrics(end_time=end_time)
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

    async def start_stt_usage_metrics(self, usage: STTUsage):
        """Start STT usage metrics collection.

        Args:
            usage: Usage information for the STT operation.
        """
        if self.can_generate_metrics() and self.usage_metrics_enabled:
            frame = await self._metrics.start_stt_usage_metrics(usage)
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

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor with required components.

        This is where a processor connects and does its other slow start-up
        work, so that the pipeline pays for the slowest processor rather than
        all of them: a pipeline sets its processors up concurrently, so this
        runs alongside every other processor's. A resource shared with another
        processor therefore needs guarding, which
        :func:`~pipecat.utils.shared.acquires` does.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup.task_manager)
        self._setup = setup

        if self._metrics is not None:
            await self._metrics.setup(self.task_manager)

    async def cleanup(self):
        """Release this processor's resources at teardown.

        This base implementation cancels only the processor's internal
        input/process tasks; tasks created via :meth:`create_task` are released
        by an override. Like :meth:`setup`, this runs concurrently with every
        other processor's, so a resource shared with another processor is
        released with :func:`~pipecat.utils.shared.releases`.
        """
        await super().cleanup()
        await self.__cancel_pause_watcher()
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
        return self.processor_setup.clock

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
            return

        await self.__input_queue.put((frame, direction, callback))

        # Nothing drains the queue until the StartFrame arrives, so a processor
        # never acts on a frame before it has been started. Frames pushed
        # between setup and the StartFrame simply wait, and the StartFrame is
        # dequeued ahead of them.
        if isinstance(frame, StartFrame):
            self.__create_input_task()

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

    async def pause_processing_all_frames_until(
        self,
        ready: Callable[[], Awaitable[Any]],
        *,
        timeout: float = PAUSE_UNTIL_READY_TIMEOUT_SECS,
    ):
        """Hold frames arriving at this processor until ``ready`` resolves.

        Useful for a processor that cannot act on frames until some condition
        holds, such as one that establishes a connection in the background.
        Frames wait in the processor's queues and are delivered in order once
        the condition resolves, so nothing is lost.

        The frame being processed when this is called is unaffected: the pause
        takes hold from the next frame onwards. A processor pausing while it
        handles its ``StartFrame``, for instance, still passes that frame on
        downstream, so pipeline startup is not delayed.

        Both queues are held, so a processor left paused could not process the
        frames that shut it down. The pause is therefore always lifted: when
        ``ready`` resolves, when ``timeout`` elapses, or at teardown, whichever
        comes first.

        Args:
            ready: Awaited to learn when frames can be acted on, e.g.
                ``some_event.wait``.
            timeout: Seconds to hold frames before giving up and resuming.
        """
        if self._enable_direct_mode:
            logger.warning(f"{self}: cannot hold frames, this processor runs in direct mode")
            return

        await self.__cancel_pause_watcher()
        await self.pause_processing_system_frames()
        await self.pause_processing_frames()
        self.__pause_watcher_task = self.create_task(
            self.__pause_watcher_handler(ready, timeout), name=f"{self}::pause_watcher"
        )

    async def __pause_watcher_handler(self, ready: Callable[[], Awaitable[Any]], timeout: float):
        """Lift the pause taken by pause_processing_all_frames_until()."""
        try:
            await asyncio.wait_for(ready(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                f"{self}: still not ready after {timeout}s, resuming frame processing anyway"
            )
        await self.__resume_processing_all_frames()

    async def __cancel_pause_watcher(self):
        """Stop watching, and lift the pause the watcher was going to lift."""
        if not self.__pause_watcher_task:
            return

        task = self.__pause_watcher_task
        self.__pause_watcher_task = None
        await self.cancel_task(task)
        await self.__resume_processing_all_frames()

    async def __resume_processing_all_frames(self):
        """Resume both queues held by pause_processing_all_frames_until()."""
        await self.resume_processing_system_frames()
        await self.resume_processing_frames()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow.
        """
        observer = self._setup.observer if self._setup else None
        if observer:
            data = FrameProcessed(
                processor=self,
                frame=frame,
                direction=direction,
                timestamp=self.get_clock().get_time(),
            )
            await observer.on_process_frame(data)

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

    def _classify_error(self, exception: Exception) -> ErrorCategory | None:
        """Classify an exception this processor knows the shape of.

        Override for providers that signal failures through SDK-specific
        exceptions rather than an HTTP status code, or whose credentials can be
        rejected for reasons a reconnection would clear.

        Args:
            exception: The exception to classify.

        Returns:
            The category, or None to fall back to :func:`classify_http_exception`.
        """
        return None

    async def push_error(
        self,
        error_msg: str,
        exception: Exception | None = None,
        fatal: bool = False,
        category: ErrorCategory | None = None,
        force_treat_as_permanent: bool = False,
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

                .. deprecated:: 1.8.0
                    Use ``force_treat_as_permanent=True`` instead, when the
                    error leaves its originating processor unable to do its
                    job: the pipeline worker then applies its
                    :class:`ProcessorUnusablePolicy`, of which ``CANCEL``
                    matches what ``fatal=True`` did. For an error that isn't
                    about that processor's state, push an
                    :class:`EndWorkerFrame` after the error to end the pipeline.
                    Will be removed in 2.0.0.

            category: Why the error occurred, when the caller knows. Leave it
                unset to let the category be worked out from the exception, or
                pass `ErrorCategory.UNKNOWN` to report an error whose cause
                can't be attributed — an unexpected one caught by a broad
                ``except``, say, which may not have come from this processor at
                all.
            force_treat_as_permanent: Whether to treat this error as one that will
                keep recurring, leaving the processor unable to do any more
                work — having failed too many times to keep trying, say. Only
                needed for a failure the category doesn't already convey:
                leaving it False doesn't keep the processor usable, since a
                permanent category costs it its usability on its own. Either way
                the processor stops being given work, and the pipeline worker
                decides what to do about it through its
                :class:`ProcessorUnusablePolicy`: report the error and keep
                running (the default), end the pipeline, or cancel it.

        Example::

            ```python
            # An error this processor can recover from
            await self.push_error("Failed to process audio chunk, skipping")

            # An error that leaves this processor unable to do any more work
            try:
                result = some_critical_operation()
            except Exception as e:
                await self.push_error(
                    "Critical operation failed", exception=e, force_treat_as_permanent=True
                )
            ```
        """
        if fatal:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "`push_error(fatal=True)` is deprecated since 1.8.0 and will be removed "
                    "in 2.0.0. If the error leaves its originating processor unable to do "
                    "its job, pass `force_treat_as_permanent=True` instead: that marks the "
                    "processor unusable, and the PipelineWorker acts on it according to its "
                    "`processor_unusable_policy` (`ProcessorUnusablePolicy.CANCEL` does what "
                    "`fatal=True` did). Otherwise, drop `fatal` and push an "
                    "`EndWorkerFrame` after the error to end the pipeline.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        error_frame = ErrorFrame(
            error=error_msg,
            exception=exception,
            processor=self,
            category=category,
        )
        # Set after construction so the frame doesn't warn about the deprecated
        # flag on top of the warning already reported above.
        error_frame.fatal = fatal

        # Subclasses may override `push_error_frame` with its original
        # one-argument signature, so only pass the flag when it is set.
        if force_treat_as_permanent:
            await self.push_error_frame(error=error_frame, force_treat_as_permanent=True)
        else:
            await self.push_error_frame(error=error_frame)

    async def push_error_frame(self, error: ErrorFrame, force_treat_as_permanent: bool = False):
        """Push an error frame upstream.

        Args:
            error: The error frame to push. Its deprecated ``fatal`` flag still
                cancels the pipeline; ``force_treat_as_permanent`` is the replacement.
            force_treat_as_permanent: Whether to treat this error as one that will
                keep recurring, leaving the processor unable to do any more
                work. Leaving it False doesn't keep the processor usable — a
                permanent category costs it its usability either way. See
                :meth:`push_error`.
        """
        if not error.processor:
            error.processor = self
        # Anything still unset by now is going to stay that way, so settle it
        # here and let handlers read a category off every error they receive.
        if error.category is None and error.exception:
            error.category = self._classify_error(error.exception) or classify_http_exception(
                error.exception
            )
        if error.category is None:
            error.category = ErrorCategory.UNKNOWN

        # Before anything sees the error, so that handlers reading
        # `frame.processor.is_usable` get the verdict that came with it.
        if force_treat_as_permanent or error.category.is_permanent:
            await self.set_usable(False)

        await self._call_event_handler("on_error", error)

        # An exception carries a traceback only once it has been raised, so fall
        # back to the plain message rather than losing the error entirely.
        tb = traceback.extract_tb(error.exception.__traceback__) if error.exception else []
        if tb:
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
        await self._call_event_handler("on_before_push_frame", frame)

        await self.__internal_push_frame(frame, direction)

        await self._call_event_handler("on_after_push_frame", frame)

    async def broadcast_interruption(self):
        """Broadcast an `InterruptionFrame` both upstream and downstream."""
        logger.debug(f"{self}: broadcasting interruption")
        self.__reset_process_task()
        await self.stop_all_metrics()
        await self.broadcast_frame(InterruptionFrame)

    @deprecated(
        "`FrameProcessor.push_interruption_task_frame_and_wait` is deprecated since 0.0.104 "
        "and will be removed in 2.0.0. Use `broadcast_interruption` instead."
    )
    async def push_interruption_task_frame_and_wait(self, *, timeout: float = 5.0):
        """Push an interruption task frame upstream and wait for the interruption.

        .. deprecated:: 0.0.104
            Use :meth:`broadcast_interruption` instead. This method now
            delegates to ``broadcast_interruption()`` and ignores *timeout*.
            Will be removed in 2.0.0.
        """
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
            if current_is_uninterruptible:
                # The frame currently being processed is uninterruptible, so we
                # must not cancel it. Just flush non-uninterruptible frames from
                # the queue; any uninterruptible ones will be kept and processed
                # after the current frame finishes.
                self.__reset_process_queue()
            else:
                # Cancel and re-create the process task. Previously this branch
                # was skipped when the queue contained an uninterruptible frame,
                # which caused slow non-uninterruptible frames to block
                # interruptions. Uninterruptible queued frames are safe here
                # because __create_process_task calls __reset_process_queue
                # internally, which always preserves them.
                await self.__cancel_process_task()
                self.__create_process_task()
        except Exception as e:
            await self.push_error(
                error_msg=f"Uncaught exception handling _start_interruption: {e}",
                exception=e,
                # A broad catch: this may not have come from this processor at
                # all, so its cause can't be attributed.
                category=ErrorCategory.UNKNOWN,
            )

    async def __internal_push_frame(self, frame: Frame, direction: FrameDirection):
        """Internal method to push frames to adjacent processors.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        observer = self._setup.observer if self._setup else None
        try:
            timestamp = self.get_clock().get_time() if self._setup else 0
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} downstream from {self} to {self._next}")

                if observer:
                    data = FramePushed(
                        source=self,
                        destination=self._next,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await observer.on_push_frame(data)
                await self._next.queue_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                if observer:
                    data = FramePushed(
                        source=self,
                        destination=self._prev,
                        frame=frame,
                        direction=direction,
                        timestamp=timestamp,
                    )
                    await observer.on_push_frame(data)
                await self._prev.queue_frame(frame, direction)
        except Exception as e:
            # Observers and the downstream processor run inside this
            # block, so the cause of an unexpected failure can't be attributed.
            await self.push_error(
                error_msg=f"Uncaught exception: {e}",
                exception=e,
                category=ErrorCategory.UNKNOWN,
            )

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
            # The frame callback runs inside this block, so the cause of an
            # unexpected failure can't be attributed.
            await self.push_error(
                error_msg=f"Error processing frame: {e}",
                exception=e,
                category=ErrorCategory.UNKNOWN,
            )

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
