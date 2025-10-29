#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline task implementation for managing frame processing pipelines.

This module provides the main PipelineTask class that orchestrates pipeline
execution, frame routing, lifecycle management, and monitoring capabilities
including heartbeats, idle detection, and observer integration.
"""

import asyncio
import time
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Tuple, Type

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from pipecat.audio.interruptions.base_interruption_strategy import BaseInterruptionStrategy
from pipecat.clocks.base_clock import BaseClock
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotSpeakingFrame,
    CancelFrame,
    CancelTaskFrame,
    EndFrame,
    EndTaskFrame,
    ErrorFrame,
    Frame,
    HeartbeatFrame,
    InterruptionFrame,
    InterruptionTaskFrame,
    MetricsFrame,
    StartFrame,
    StopFrame,
    StopTaskFrame,
    UserSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.pipeline.base_task import BasePipelineTask, PipelineTaskParams
from pipecat.pipeline.pipeline import Pipeline, PipelineSink, PipelineSource
from pipecat.pipeline.task_observer import TaskObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import BaseTaskManager, TaskManager, TaskManagerParams
from pipecat.utils.tracing.setup import is_tracing_available
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver

HEARTBEAT_SECS = 1.0
HEARTBEAT_MONITOR_SECS = HEARTBEAT_SECS * 10

IDLE_TIMEOUT_SECS = 300

CANCEL_TIMEOUT_SECS = 20.0


class PipelineParams(BaseModel):
    """Configuration parameters for pipeline execution.

    These parameters are usually passed to all frame processors through
    StartFrame. For other generic pipeline task parameters use PipelineTask
    constructor arguments instead.

    Parameters:
        allow_interruptions: Whether to allow pipeline interruptions.
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_out_sample_rate: Output audio sample rate in Hz.
        enable_heartbeats: Whether to enable heartbeat monitoring.
        enable_metrics: Whether to enable metrics collection.
        enable_usage_metrics: Whether to enable usage metrics.
        heartbeats_period_secs: Period between heartbeats in seconds.
        interruption_strategies: Strategies for bot interruption behavior.
        observers: [deprecated] Use `observers` arg in `PipelineTask` class.

            .. deprecated:: 0.0.58
                Use the `observers` argument in the `PipelineTask` class instead.

        report_only_initial_ttfb: Whether to report only initial time to first byte.
        send_initial_empty_metrics: Whether to send initial empty metrics.
        start_metadata: Additional metadata for pipeline start.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    allow_interruptions: bool = True
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_heartbeats: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    heartbeats_period_secs: float = HEARTBEAT_SECS
    interruption_strategies: List[BaseInterruptionStrategy] = Field(default_factory=list)
    observers: List[BaseObserver] = Field(default_factory=list)
    report_only_initial_ttfb: bool = False
    send_initial_empty_metrics: bool = True
    start_metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineTask(BasePipelineTask):
    """Manages the execution of a pipeline, handling frame processing and task lifecycle.

    This class orchestrates pipeline execution with comprehensive monitoring,
    event handling, and lifecycle management. It provides event handlers for
    various pipeline states and frame types, idle detection, heartbeat monitoring,
    and observer integration.

    Event handlers available:

    - on_frame_reached_upstream: Called when upstream frames reach the source
    - on_frame_reached_downstream: Called when downstream frames reach the sink
    - on_idle_timeout: Called when pipeline is idle beyond timeout threshold
    - on_pipeline_started: Called when pipeline starts with StartFrame
    - on_pipeline_stopped: [deprecated] Called when pipeline stops with StopFrame

            .. deprecated:: 0.0.86
                Use `on_pipeline_finished` instead.

    - on_pipeline_ended: [deprecated] Called when pipeline ends with EndFrame

            .. deprecated:: 0.0.86
                Use `on_pipeline_finished` instead.

    - on_pipeline_cancelled: [deprecated] Called when pipeline is cancelled with CancelFrame

            .. deprecated:: 0.0.86
                Use `on_pipeline_finished` instead.

    - on_pipeline_finished: Called after the pipeline has reached any terminal state.
          This includes:

              - StopFrame: pipeline was stopped (processors keep connections open)
              - EndFrame: pipeline ended normally
              - CancelFrame: pipeline was cancelled

          Use this event for cleanup, logging, or post-processing tasks. Users can inspect
          the frame if they need to handle specific cases.

    - on_pipeline_error: Called when an error occurs with ErrorFrame

    Example::

        @task.event_handler("on_frame_reached_upstream")
        async def on_frame_reached_upstream(task, frame):
            ...

        @task.event_handler("on_idle_timeout")
        async def on_pipeline_idle_timeout(task):
            ...

        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame):
            ...

        @task.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(task, frame):
            ...

        @task.event_handler("on_pipeline_error")
        async def on_pipeline_error(task, frame):
            ...
    """

    def __init__(
        self,
        pipeline: FrameProcessor,
        *,
        params: Optional[PipelineParams] = None,
        additional_span_attributes: Optional[dict] = None,
        cancel_on_idle_timeout: bool = True,
        cancel_timeout_secs: float = CANCEL_TIMEOUT_SECS,
        check_dangling_tasks: bool = True,
        clock: Optional[BaseClock] = None,
        conversation_id: Optional[str] = None,
        enable_tracing: bool = False,
        enable_turn_tracking: bool = True,
        idle_timeout_frames: Tuple[Type[Frame], ...] = (BotSpeakingFrame, UserSpeakingFrame),
        idle_timeout_secs: Optional[float] = IDLE_TIMEOUT_SECS,
        observers: Optional[List[BaseObserver]] = None,
        task_manager: Optional[BaseTaskManager] = None,
    ):
        """Initialize the PipelineTask.

        Args:
            pipeline: The pipeline to execute.
            params: Configuration parameters for the pipeline.
            additional_span_attributes: Optional dictionary of attributes to propagate as
                OpenTelemetry conversation span attributes.
            cancel_on_idle_timeout: Whether the pipeline task should be cancelled if
                the idle timeout is reached.
            cancel_timeout_secs: Timeout (in seconds) to wait for cancellation to happen
                cleanly.
            check_dangling_tasks: Whether to check for processors' tasks finishing properly.
            clock: Clock implementation for timing operations.
            conversation_id: Optional custom ID for the conversation.
            enable_tracing: Whether to enable tracing.
            enable_turn_tracking: Whether to enable turn tracking.
            idle_timeout_frames: A tuple with the frames that should trigger an idle
                timeout if not received within `idle_timeout_seconds`.
            idle_timeout_secs: Timeout (in seconds) to consider pipeline idle or
                None. If a pipeline is idle the pipeline task will be cancelled
                automatically.
            observers: List of observers for monitoring pipeline execution.
            task_manager: Optional task manager for handling asyncio tasks.
        """
        super().__init__()
        self._params = params or PipelineParams()
        self._additional_span_attributes = additional_span_attributes or {}
        self._cancel_on_idle_timeout = cancel_on_idle_timeout
        self._cancel_timeout_secs = cancel_timeout_secs
        self._check_dangling_tasks = check_dangling_tasks
        self._clock = clock or SystemClock()
        self._conversation_id = conversation_id
        self._enable_tracing = enable_tracing and is_tracing_available()
        self._enable_turn_tracking = enable_turn_tracking
        self._idle_timeout_frames = idle_timeout_frames
        self._idle_timeout_secs = idle_timeout_secs
        if self._params.observers:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Field 'observers' is deprecated, use the 'observers' parameter instead.",
                    DeprecationWarning,
                )
            observers = self._params.observers
        observers = observers or []
        self._turn_tracking_observer: Optional[TurnTrackingObserver] = None
        self._turn_trace_observer: Optional[TurnTraceObserver] = None
        if self._enable_turn_tracking:
            self._turn_tracking_observer = TurnTrackingObserver()
            observers.append(self._turn_tracking_observer)
        if self._enable_tracing and self._turn_tracking_observer:
            self._turn_trace_observer = TurnTraceObserver(
                self._turn_tracking_observer,
                conversation_id=self._conversation_id,
                additional_span_attributes=self._additional_span_attributes,
            )
            observers.append(self._turn_trace_observer)
        self._finished = False
        self._cancelled = False

        # This task maneger will handle all the asyncio tasks created by this
        # PipelineTask and its frame processors.
        self._task_manager = task_manager or TaskManager()

        # This queue is the queue used to push frames to the pipeline.
        self._push_queue = asyncio.Queue()
        self._process_push_task: Optional[asyncio.Task] = None
        # This is the heartbeat queue. When a heartbeat frame is received in the
        # down queue we add it to the heartbeat queue for processing.
        self._heartbeat_queue = asyncio.Queue()
        self._heartbeat_push_task: Optional[asyncio.Task] = None
        self._heartbeat_monitor_task: Optional[asyncio.Task] = None
        # This is the idle queue. When frames are received downstream they are
        # put in the queue. If no frame is received the pipeline is considered
        # idle.
        self._idle_queue = asyncio.Queue()
        self._idle_monitor_task: Optional[asyncio.Task] = None

        # This event is used to indicate the StartFrame has been received at the
        # end of the pipeline.
        self._pipeline_start_event = asyncio.Event()

        # This event is used to indicate a finalize frame (e.g. EndFrame,
        # StopFrame) has been received at the end of the pipeline.
        self._pipeline_end_event = asyncio.Event()

        # This event is set when the pipeline truly finishes.
        self._pipeline_finished_event = asyncio.Event()

        # This is the final pipeline. It is composed of a source processor,
        # followed by the user pipeline, and ending with a sink processor. The
        # source allows us to receive and react to upstream frames, and the sink
        # allows us to receive and react to downstream frames.
        source = PipelineSource(self._source_push_frame, name=f"{self}::Source")
        sink = PipelineSink(self._sink_push_frame, name=f"{self}::Sink")
        self._pipeline = Pipeline([pipeline], source=source, sink=sink)

        # The task observer acts as a proxy to the provided observers. This way,
        # we only need to pass a single observer (using the StartFrame) which
        # then just acts as a proxy.
        self._observer = TaskObserver(observers=observers, task_manager=self._task_manager)

        # These events can be used to check which frames make it to the source
        # or sink processors. Instead of calling the event handlers for every
        # frame the user needs to specify which events they are interested
        # in. This is mainly for efficiency reason because each event handler
        # creates a task and most likely you only care about one or two frame
        # types.
        self._reached_upstream_types: Tuple[Type[Frame], ...] = ()
        self._reached_downstream_types: Tuple[Type[Frame], ...] = ()
        self._register_event_handler("on_frame_reached_upstream")
        self._register_event_handler("on_frame_reached_downstream")
        self._register_event_handler("on_idle_timeout")
        self._register_event_handler("on_pipeline_started")
        self._register_event_handler("on_pipeline_stopped")
        self._register_event_handler("on_pipeline_ended")
        self._register_event_handler("on_pipeline_cancelled")
        self._register_event_handler("on_pipeline_finished")
        self._register_event_handler("on_pipeline_error")

    @property
    def params(self) -> PipelineParams:
        """Get the pipeline parameters for this task.

        Returns:
            The pipeline parameters configuration.
        """
        return self._params

    @property
    def turn_tracking_observer(self) -> Optional[TurnTrackingObserver]:
        """Get the turn tracking observer if enabled.

        Returns:
            The turn tracking observer instance or None if not enabled.
        """
        return self._turn_tracking_observer

    @property
    def turn_trace_observer(self) -> Optional[TurnTraceObserver]:
        """Get the turn trace observer if enabled.

        Returns:
            The turn trace observer instance or None if not enabled.
        """
        return self._turn_trace_observer

    def event_handler(self, event_name: str):
        """Decorator for registering event handlers.

        Args:
            event_name: The name of the event to handle.

        Returns:
            The decorator function that registers the handler.
        """
        if event_name in ["on_pipeline_stopped", "on_pipeline_ended", "on_pipeline_cancelled"]:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    f"Event '{event_name}' is deprecated, use 'on_pipeline_finished' instead.",
                    DeprecationWarning,
                )

        return super().event_handler(event_name)

    def add_observer(self, observer: BaseObserver):
        """Add an observer to monitor pipeline execution.

        Args:
            observer: The observer to add to the pipeline monitoring.
        """
        self._observer.add_observer(observer)

    async def remove_observer(self, observer: BaseObserver):
        """Remove an observer from pipeline monitoring.

        Args:
            observer: The observer to remove from pipeline monitoring.
        """
        await self._observer.remove_observer(observer)

    def set_reached_upstream_filter(self, types: Tuple[Type[Frame], ...]):
        """Set which frame types trigger the on_frame_reached_upstream event.

        Args:
            types: Tuple of frame types to monitor for upstream events.
        """
        self._reached_upstream_types = types

    def set_reached_downstream_filter(self, types: Tuple[Type[Frame], ...]):
        """Set which frame types trigger the on_frame_reached_downstream event.

        Args:
            types: Tuple of frame types to monitor for downstream events.
        """
        self._reached_downstream_types = types

    def has_finished(self) -> bool:
        """Check if the pipeline task has finished execution.

        This indicates whether the tasks has finished, meaninig all processors
        have stopped.

        Returns:
            True if all processors have stopped and the task is complete.
        """
        return self._finished

    async def stop_when_done(self):
        """Schedule the pipeline to stop after processing all queued frames.

        Sends an EndFrame to gracefully terminate the pipeline once all
        current processing is complete.
        """
        logger.debug(f"Task {self} scheduled to stop when done")
        await self.queue_frame(EndFrame())

    async def cancel(self):
        """Request the running pipeline to cancel."""
        if not self._finished:
            await self._cancel()

    async def run(self, params: PipelineTaskParams):
        """Start and manage the pipeline execution until completion or cancellation.

        Args:
            params: Configuration parameters for pipeline execution.
        """
        if self.has_finished():
            return

        # Setup processors.
        await self._setup(params)

        # Create all main tasks and wait for the main push task. This is the
        # task that pushes frames to the very beginning of our pipeline (i.e. to
        # our controlled source processor).
        await self._create_tasks()

        try:
            # Wait for pipeline to finish.
            await self._wait_for_pipeline_finished()
        except asyncio.CancelledError:
            logger.debug(f"Pipeline task {self} got cancelled from outside...")
            # We have been cancelled from outside, let's just cancel everything.
            await self._cancel()
            # Wait again for pipeline to finish. This time we have really
            # cancelled, so it should really finish.
            await self._wait_for_pipeline_finished()
            # Re-raise in case there's more cleanup to do.
            raise
        finally:
            # We can reach this point for different reasons:
            #
            # 1. The pipeline task has finished (try case).
            # 2. By an asyncio task cancellation (except case).
            logger.debug(f"Pipeline task {self} is finishing...")
            await self._cancel_tasks()
            if self._check_dangling_tasks:
                self._print_dangling_tasks()
            self._finished = True
            logger.debug(f"Pipeline task {self} has finished")

    async def queue_frame(self, frame: Frame):
        """Queue a single frame to be pushed down the pipeline.

        Args:
            frame: The frame to be processed.
        """
        await self._push_queue.put(frame)

    async def queue_frames(self, frames: Iterable[Frame] | AsyncIterable[Frame]):
        """Queues multiple frames to be pushed down the pipeline.

        Args:
            frames: An iterable or async iterable of frames to be processed.
        """
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame)

    async def _cancel(self):
        """Internal cancellation logic for the pipeline task."""
        if not self._cancelled:
            logger.debug(f"Cancelling pipeline task {self}")
            self._cancelled = True
            await self.queue_frame(CancelFrame())

    async def _create_tasks(self):
        """Create and start all pipeline processing tasks."""
        self._process_push_task = self._task_manager.create_task(
            self._process_push_queue(), f"{self}::_process_push_queue"
        )

        await self._observer.start()

        return self._process_push_task

    def _maybe_start_heartbeat_tasks(self):
        """Start heartbeat tasks if heartbeats are enabled and not already running."""
        if self._params.enable_heartbeats and self._heartbeat_push_task is None:
            self._heartbeat_push_task = self._task_manager.create_task(
                self._heartbeat_push_handler(), f"{self}::_heartbeat_push_handler"
            )
            self._heartbeat_monitor_task = self._task_manager.create_task(
                self._heartbeat_monitor_handler(), f"{self}::_heartbeat_monitor_handler"
            )

    def _maybe_start_idle_task(self):
        """Start idle monitoring task if idle timeout is configured."""
        if self._idle_timeout_secs:
            self._idle_monitor_task = self._task_manager.create_task(
                self._idle_monitor_handler(), f"{self}::_idle_monitor_handler"
            )

    async def _cancel_tasks(self):
        """Cancel all running pipeline tasks."""
        await self._observer.stop()

        if self._process_push_task:
            await self._task_manager.cancel_task(self._process_push_task)
            self._process_push_task = None

        await self._maybe_cancel_heartbeat_tasks()
        await self._maybe_cancel_idle_task()

    async def _maybe_cancel_heartbeat_tasks(self):
        """Cancel heartbeat tasks if they are running."""
        if not self._params.enable_heartbeats:
            return

        if self._heartbeat_push_task:
            await self._task_manager.cancel_task(self._heartbeat_push_task)
            self._heartbeat_push_task = None

        if self._heartbeat_monitor_task:
            await self._task_manager.cancel_task(self._heartbeat_monitor_task)
            self._heartbeat_monitor_task = None

    async def _maybe_cancel_idle_task(self):
        """Cancel idle monitoring task if it is running."""
        if self._idle_timeout_secs and self._idle_monitor_task:
            await self._task_manager.cancel_task(self._idle_monitor_task)
            self._idle_monitor_task = None

    def _initial_metrics_frame(self) -> MetricsFrame:
        """Create an initial metrics frame with zero values for all processors."""
        processors = self._pipeline.processors_with_metrics()
        data = []
        for p in processors:
            data.append(TTFBMetricsData(processor=p.name, value=0.0))
            data.append(ProcessingMetricsData(processor=p.name, value=0.0))
        return MetricsFrame(data=data)

    async def _wait_for_pipeline_start(self, frame: Frame):
        """Wait for the specified start frame to reach the end of the pipeline."""
        logger.debug(f"{self}: Starting. Waiting for {frame} to reach the end of the pipeline...")
        await self._pipeline_start_event.wait()
        self._pipeline_start_event.clear()
        logger.debug(f"{self}: {frame} reached the end of the pipeline, pipeline is now ready.")

    async def _wait_for_pipeline_end(self, frame: Frame):
        """Wait for the specified frame to reach the end of the pipeline."""

        async def wait_for_cancel():
            try:
                await asyncio.wait_for(
                    self._pipeline_end_event.wait(), timeout=self._cancel_timeout_secs
                )
                logger.debug(f"{self}: {frame} reached the end of the pipeline.")
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self}: timeout waiting for {frame} to reach the end of the pipeline (being blocked somewhere?)."
                )
            finally:
                await self._call_event_handler("on_pipeline_cancelled", frame)
                await self._call_event_handler("on_pipeline_finished", frame)

        logger.debug(f"{self}: Closing. Waiting for {frame} to reach the end of the pipeline...")

        if isinstance(frame, CancelFrame):
            await wait_for_cancel()
        else:
            await self._pipeline_end_event.wait()
            logger.debug(f"{self}: {frame} reached the end of the pipeline, pipeline is closing.")

        self._pipeline_end_event.clear()

        # We are really done.
        self._pipeline_finished_event.set()

    async def _wait_for_pipeline_finished(self):
        await self._pipeline_finished_event.wait()
        self._pipeline_finished_event.clear()
        # Make sure we wait for the main task to complete.
        if self._process_push_task:
            await self._process_push_task
            self._process_push_task = None

    async def _setup(self, params: PipelineTaskParams):
        """Set up the pipeline task and all processors."""
        mgr_params = TaskManagerParams(loop=params.loop)
        self._task_manager.setup(mgr_params)

        setup = FrameProcessorSetup(
            clock=self._clock,
            task_manager=self._task_manager,
            observer=self._observer,
        )
        await self._pipeline.setup(setup)

    async def _cleanup(self, cleanup_pipeline: bool):
        """Clean up the pipeline task and processors."""
        # Cleanup base object.
        await self.cleanup()

        # Cleanup observers.
        if self._observer:
            await self._observer.cleanup()

        # End conversation tracing if it's active - this will also close any active turn span
        if self._enable_tracing and hasattr(self, "_turn_trace_observer"):
            self._turn_trace_observer.end_conversation_tracing()

        # Cleanup pipeline processors.
        if cleanup_pipeline:
            await self._pipeline.cleanup()

    async def _process_push_queue(self):
        """Process frames from the push queue and send them through the pipeline.

        This is the task that runs the pipeline for the first time by sending
        a StartFrame and by pushing any other frames queued by the user. It runs
        until the tasks is cancelled or stopped (e.g. with an EndFrame).
        """
        self._clock.start()

        self._maybe_start_idle_task()

        start_frame = StartFrame(
            allow_interruptions=self._params.allow_interruptions,
            audio_in_sample_rate=self._params.audio_in_sample_rate,
            audio_out_sample_rate=self._params.audio_out_sample_rate,
            enable_metrics=self._params.enable_metrics,
            enable_tracing=self._enable_tracing,
            enable_usage_metrics=self._params.enable_usage_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb,
            interruption_strategies=self._params.interruption_strategies,
        )
        start_frame.metadata = self._params.start_metadata
        await self._pipeline.queue_frame(start_frame)

        # Wait for the pipeline to be started before pushing any other frame.
        await self._wait_for_pipeline_start(start_frame)

        if self._params.enable_metrics and self._params.send_initial_empty_metrics:
            await self._pipeline.queue_frame(self._initial_metrics_frame())

        running = True
        cleanup_pipeline = True
        while running:
            frame = await self._push_queue.get()
            await self._pipeline.queue_frame(frame)
            if isinstance(frame, (CancelFrame, EndFrame, StopFrame)):
                await self._wait_for_pipeline_end(frame)
            running = not isinstance(frame, (CancelFrame, EndFrame, StopFrame))
            cleanup_pipeline = not isinstance(frame, StopFrame)
            self._push_queue.task_done()
        await self._cleanup(cleanup_pipeline)

    async def _source_push_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames coming upstream from the pipeline.

        This is the task that processes frames coming upstream from the
        pipeline. These frames might indicate, for example, that we want the
        pipeline to be stopped (e.g. EndTaskFrame) in which case we would send
        an EndFrame down the pipeline.
        """
        if isinstance(frame, self._reached_upstream_types):
            await self._call_event_handler("on_frame_reached_upstream", frame)

        if isinstance(frame, EndTaskFrame):
            # Tell the task we should end nicely.
            logger.debug(f"{self}: received end task frame {frame}")
            await self.queue_frame(EndFrame())
        elif isinstance(frame, CancelTaskFrame):
            # Tell the task we should end right away.
            logger.debug(f"{self}: received cancel task frame {frame}")
            await self.queue_frame(CancelFrame())
        elif isinstance(frame, StopTaskFrame):
            # Tell the task we should stop nicely.
            logger.debug(f"{self}: received stop task frame {frame}")
            await self.queue_frame(StopFrame())
        elif isinstance(frame, InterruptionTaskFrame):
            # Tell the task we should interrupt the pipeline. Note that we are
            # bypassing the push queue and directly queue into the
            # pipeline. This is in case the push task is blocked waiting for a
            # pipeline-ending frame to finish traversing the pipeline.
            logger.debug(f"{self}: received interruption task frame {frame}")
            await self._pipeline.queue_frame(InterruptionFrame())
        elif isinstance(frame, ErrorFrame):
            await self._call_event_handler("on_pipeline_error", frame)
            if frame.fatal:
                logger.error(f"A fatal error occurred: {frame}")
                # Cancel all tasks downstream.
                await self.queue_frame(CancelFrame())
            else:
                logger.warning(f"{self}: Something went wrong: {frame}")

    async def _sink_push_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames coming downstream from the pipeline.

        This tasks process frames coming downstream from the pipeline. For
        example, heartbeat frames or an EndFrame which would indicate all
        processors have handled the EndFrame and therefore we can exit the task
        cleanly.
        """
        # Queue received frame to the idle queue so we can monitor idle
        # pipelines.
        await self._idle_queue.put(frame)

        if isinstance(frame, self._reached_downstream_types):
            await self._call_event_handler("on_frame_reached_downstream", frame)

        if isinstance(frame, StartFrame):
            await self._call_event_handler("on_pipeline_started", frame)

            # Start heartbeat tasks now that StartFrame has been processed
            # by all processors in the pipeline
            self._maybe_start_heartbeat_tasks()

            self._pipeline_start_event.set()
        elif isinstance(frame, EndFrame):
            await self._call_event_handler("on_pipeline_ended", frame)
            await self._call_event_handler("on_pipeline_finished", frame)
            self._pipeline_end_event.set()
        elif isinstance(frame, StopFrame):
            await self._call_event_handler("on_pipeline_stopped", frame)
            await self._call_event_handler("on_pipeline_finished", frame)
            self._pipeline_end_event.set()
        elif isinstance(frame, CancelFrame):
            self._pipeline_end_event.set()
        elif isinstance(frame, HeartbeatFrame):
            await self._heartbeat_queue.put(frame)

    async def _heartbeat_push_handler(self):
        """Push heartbeat frames at regular intervals."""
        while True:
            # Don't use `queue_frame()` because if an EndFrame is queued the
            # task will just stop waiting for the pipeline to finish not
            # allowing more frames to be pushed.
            await self._pipeline.queue_frame(HeartbeatFrame(timestamp=self._clock.get_time()))
            await asyncio.sleep(self._params.heartbeats_period_secs)

    async def _heartbeat_monitor_handler(self):
        """Monitor heartbeat frames for processing time and timeout detection.

        This task monitors heartbeat frames. If a heartbeat frame has not
        been received for a long period a warning will be logged. It also logs
        the time that a heartbeat frame takes to processes, that is how long it
        takes for the heartbeat frame to traverse all the pipeline.
        """
        wait_time = HEARTBEAT_MONITOR_SECS
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

    async def _idle_monitor_handler(self):
        """Monitor pipeline activity and detect idle conditions.

        Tracks frame activity and triggers idle timeout events when the
        pipeline hasn't received relevant frames within the timeout period.

        Note: Heartbeats are excluded from idle detection.
        """
        running = True
        last_frame_time = 0

        while running:
            try:
                frame = await asyncio.wait_for(
                    self._idle_queue.get(), timeout=self._idle_timeout_secs
                )

                if isinstance(frame, StartFrame) or isinstance(frame, self._idle_timeout_frames):
                    # If we find a StartFrame or one of the frames that prevents a
                    # time out we update the time.
                    last_frame_time = time.time()
                else:
                    # If we find any other frame we check if the pipeline is
                    # idle by checking the last time we received one of the
                    # valid frames.
                    diff_time = time.time() - last_frame_time
                    if diff_time >= self._idle_timeout_secs:
                        running = await self._idle_timeout_detected()
                        # Reset `last_frame_time` so we don't trigger another
                        # immediate idle timeout if we are not cancelling. For
                        # example, we might want to force the bot to say goodbye
                        # and then clean nicely with an `EndFrame`.
                        last_frame_time = time.time()

                self._idle_queue.task_done()

            except asyncio.TimeoutError:
                running = await self._idle_timeout_detected()

    async def _idle_timeout_detected(self) -> bool:
        """Handle idle timeout detection and optional cancellation.

        Returns:
            Whether the pipeline task should continue running.
        """
        # If we are cancelling, just exit the task.
        if self._cancelled:
            return True

        logger.warning("Idle timeout detected.")
        await self._call_event_handler("on_idle_timeout")
        if self._cancel_on_idle_timeout:
            logger.warning(f"Idle pipeline detected, cancelling pipeline task...")
            await self.cancel()
            return False
        return True

    def _print_dangling_tasks(self):
        """Log any dangling tasks that haven't been properly cleaned up."""
        tasks = [t.get_name() for t in self._task_manager.current_tasks()]
        if tasks:
            logger.warning(f"Dangling tasks detected: {tasks}")
