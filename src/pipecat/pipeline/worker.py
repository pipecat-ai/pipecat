#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipeline worker implementation for managing frame processing pipelines.

This module provides the main PipelineWorker class that orchestrates pipeline
execution, frame routing, lifecycle management, and monitoring capabilities
including heartbeats, idle detection, and observer integration.
"""

import asyncio
import warnings
from collections.abc import AsyncIterable, Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from pipecat.bus import (
    BusCancelWorkerMessage,
    BusEndWorkerMessage,
    BusMessage,
    BusTTSSpeakMessage,
)
from pipecat.bus.bridge_processor import _BusEdgeProcessor
from pipecat.bus.ui.messages import (
    _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME,
    _UI_SNAPSHOT_BUS_EVENT_NAME,
    BusUICommandMessage,
    BusUIDataMessage,
    BusUIEventMessage,
    BusUIJobCompletedMessage,
    BusUIJobGroupCompletedMessage,
    BusUIJobGroupStartedMessage,
    BusUIJobUpdateMessage,
)
from pipecat.clocks.base_clock import BaseClock
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import (
    BotSpeakingFrame,
    CancelFrame,
    CancelWorkerFrame,
    EndFrame,
    EndWorkerFrame,
    ErrorFrame,
    Frame,
    HeartbeatFrame,
    InterruptionFrame,
    InterruptionWorkerFrame,
    MetricsFrame,
    PipelineFlushFrame,
    StartFrame,
    StopFrame,
    StopWorkerFrame,
    TTSSpeakFrame,
    UserSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import Pipeline, PipelineSink, PipelineSource
from pipecat.pipeline.worker_observer import WorkerObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIObserverParams, RTVIProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIUICommandFrame, RTVIUIJobGroupFrame
from pipecat.processors.frameworks.rtvi.models import (
    UICancelJobGroupMessage,
    UIEventMessage,
    UIJobCompletedData,
    UIJobGroupCompletedData,
    UIJobGroupStartedData,
    UIJobUpdateData,
    UISnapshotMessage,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.deprecation import deprecated
from pipecat.utils.startup import run_setup_hook
from pipecat.utils.tracing.setup import is_tracing_available
from pipecat.utils.tracing.tracing_context import TracingContext
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver
from pipecat.workers.base_worker import BaseWorker, WorkerParams

HEARTBEAT_SECS = 1.0
HEARTBEAT_MONITOR_SECS = 10.0

IDLE_TIMEOUT_SECS = 300

CANCEL_TIMEOUT_SECS = 20.0


T = TypeVar("T")


class IdleFrameObserver(BaseObserver):
    """Idle timeout observer.

    This observer waits for specific frames being generated in the pipeline. If
    the frames are generated the given asyncio event is set. If the event is not
    set it means the pipeline is probably idle.

    """

    def __init__(self, *, idle_event: asyncio.Event, idle_timeout_frames: tuple[type[Frame], ...]):
        """Initialize the observer.

        Args:
            idle_event: The event to set if the idle timeout frames are being pushed.
            idle_timeout_frames: A tuple with the frames that should set the event when received
        """
        super().__init__()
        self._idle_event = idle_event
        self._idle_timeout_frames = idle_timeout_frames
        self._processed_frames = set()

    async def on_push_frame(self, data: FramePushed):
        """Callback executed when a frame is pushed in the pipeline.

        Args:
            data: The frame push event data.
        """
        # Skip already processed frames
        if data.frame.id in self._processed_frames:
            return

        self._processed_frames.add(data.frame.id)

        if isinstance(data.frame, StartFrame) or isinstance(data.frame, self._idle_timeout_frames):
            self._idle_event.set()


class PipelineParams(BaseModel):
    """Configuration parameters for pipeline execution.

    These parameters are usually passed to all frame processors through
    StartFrame. For other generic pipeline worker parameters use PipelineWorker
    constructor arguments instead.

    Parameters:
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_out_sample_rate: Output audio sample rate in Hz.
        enable_heartbeats: Whether to enable heartbeat monitoring.
        enable_metrics: Whether to enable metrics collection.
        enable_usage_metrics: Whether to enable usage metrics.
        heartbeats_period_secs: Period between heartbeats in seconds.
        heartbeats_monitor_secs: Timeout (in seconds) before warning about
            missed heartbeats. Defaults to 10 seconds.
        report_only_initial_ttfb: Whether to report only initial time to first byte.
        send_initial_empty_metrics: Whether to send initial empty metrics.
        start_metadata: Additional metadata for pipeline start.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_heartbeats: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    heartbeats_period_secs: float = HEARTBEAT_SECS
    heartbeats_monitor_secs: float = HEARTBEAT_MONITOR_SECS
    report_only_initial_ttfb: bool = False
    send_initial_empty_metrics: bool = True
    start_metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineWorker(BaseWorker):
    """Manages the execution of a pipeline, handling frame processing and worker lifecycle.

    This class orchestrates pipeline execution with comprehensive monitoring,
    event handling, and lifecycle management. It provides event handlers for
    various pipeline states and frame types, idle detection, heartbeat monitoring,
    and observer integration.

    Event handlers available:

    - on_frame_reached_upstream: Called when upstream frames reach the source
    - on_frame_reached_downstream: Called when downstream frames reach the sink
    - on_heartbeat_timeout: Called when a heartbeat frame is not received within the monitor timeout.
          Fires repeatedly every ``heartbeats_monitor_secs`` for as long as the stall persists.
    - on_idle_timeout: Called when pipeline is idle beyond timeout threshold
    - on_pipeline_started: Called when pipeline starts with StartFrame
    - on_pipeline_finished: Called after the pipeline has reached any terminal state.
          This includes:

              - StopFrame: pipeline was stopped (processors keep connections open)
              - EndFrame: pipeline ended normally
              - CancelFrame: pipeline was cancelled

          Use this event for cleanup, logging, or post-processing tasks. Users can inspect
          the frame if they need to handle specific cases.

    - on_pipeline_error: Called when an error occurs with ErrorFrame

    Example::

        @worker.event_handler("on_frame_reached_upstream")
        async def on_frame_reached_upstream(worker, frame):
            ...

        @worker.event_handler("on_heartbeat_timeout")
        async def on_heartbeat_timeout(worker):
            ...

        @worker.event_handler("on_idle_timeout")
        async def on_pipeline_idle_timeout(worker):
            ...

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(worker, frame):
            ...

        @worker.event_handler("on_pipeline_finished")
        async def on_pipeline_finished(worker, frame):
            ...

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame):
            ...
    """

    def __init__(
        self,
        pipeline: BasePipeline,
        *,
        active: bool = True,
        additional_span_attributes: dict | None = None,
        app_resources: Any = None,
        bridged: tuple[str, ...] | None = None,
        cancel_on_idle_timeout: bool = True,
        cancel_runner_on_idle_timeout: bool = True,
        cancel_timeout_secs: float = CANCEL_TIMEOUT_SECS,
        check_dangling_tasks: bool = True,
        clock: BaseClock | None = None,
        conversation_id: str | None = None,
        enable_tracing: bool = False,
        enable_turn_tracking: bool = True,
        enable_rtvi: bool = True,
        exclude_frames: tuple[type[Frame], ...] | None = None,
        idle_timeout_frames: tuple[type[Frame], ...] = (BotSpeakingFrame, UserSpeakingFrame),
        idle_timeout_secs: float | None = IDLE_TIMEOUT_SECS,
        name: str | None = None,
        observers: list[BaseObserver] | None = None,
        params: PipelineParams | None = None,
        rtvi_processor: RTVIProcessor | None = None,
        rtvi_observer_params: RTVIObserverParams | None = None,
        task_manager: BaseTaskManager | None = None,
        tool_resources: Any = None,
    ):
        """Initialize the PipelineWorker.

        Args:
            pipeline: The pipeline to execute.
            active: Whether the worker starts active. Forwarded to
                :class:`BaseWorker`.
            additional_span_attributes: Optional dictionary of attributes to propagate as
                OpenTelemetry conversation span attributes.
            app_resources: Optional application-defined bag of anything your
                application code may want to share across this session (DB
                handles, HTTP clients, etc.), passed by reference. Pipecat
                passes it through untouched and exposes it on the worker itself
                as ``worker.app_resources`` and passes it to tool handlers as
                ``FunctionCallParams.app_resources``. The framework never
                copies or clears this object; the caller retains their handle
                and can read any mutations after the worker finishes.
            bridged: Bridge configuration. ``None`` means the pipeline
                is not bridged. An empty tuple ``()`` wraps the pipeline
                with bus edge processors that accept frames from all
                bridges. A tuple of names like ``("voice",)`` accepts
                only frames from those bridges. The bus comes from
                :meth:`attach` (called by the runner).
            cancel_on_idle_timeout: Whether reaching the idle timeout should
                cancel the pipeline worker. When ``False``, the idle event
                still fires ``on_idle_timeout`` but the worker is left alone
                (and ``cancel_runner_on_idle_timeout`` is ignored too: opting
                out of local cancellation also opts out of the runner-wide
                cancel).
            cancel_runner_on_idle_timeout: When ``cancel_on_idle_timeout`` is
                also ``True``, whether reaching the idle timeout should also
                cancel the entire :class:`WorkerRunner`. The worker is
                always cancelled first; when this is ``True`` the worker also
                emits a ``BusCancelMessage`` so the runner broadcasts
                cancellation to every other root worker. Defaults to ``True``
                so a multi-worker bot's helpers shut down with the main
                pipeline; set to ``False`` for a sidecar ``PipelineWorker``
                that should self-cancel on idle without bringing down its
                peers.
            cancel_timeout_secs: Timeout (in seconds) to wait for cancellation to happen
                cleanly.
            check_dangling_tasks: Whether to warn about tasks left running when
                the worker finishes. Only applies when the worker owns its task
                manager; otherwise the runner reports dangling tasks.
            clock: Clock implementation for timing operations.
            conversation_id: Optional custom ID for the conversation.
            enable_rtvi: Whether to automatically add RTVI support to the pipeline.
            enable_tracing: Whether to enable tracing.
            enable_turn_tracking: Whether to enable turn tracking.
            exclude_frames: When ``bridged`` is set, extra frame types
                that should not cross the bus (lifecycle frames are
                always excluded).
            idle_timeout_frames: A tuple with the frames that should trigger an idle
                timeout if not received within `idle_timeout_seconds`.
            idle_timeout_secs: Timeout (in seconds) to consider pipeline idle or
                None. If a pipeline is idle the pipeline worker will be cancelled
                automatically.
            name: Optional worker name (used for worker-style addressing on the bus).
            observers: List of observers for monitoring pipeline execution.
            params: Configuration parameters for the pipeline.
            rtvi_observer_params: The RTVI observer parameter to use if RTVI is enabled.
            rtvi_processor: The RTVI processor to add if RTVI is enabled.
            task_manager: Optional task manager for handling asyncio tasks.
            tool_resources: Deprecated alias for ``app_resources``.

                .. deprecated:: 1.2.0
                    Use ``app_resources`` instead. ``tool_resources`` will be
                    removed in 2.0.0.
        """
        super().__init__(
            name=name,
            active=active,
            task_manager=task_manager,
            check_dangling_tasks=check_dangling_tasks,
        )
        self._bridged = bridged
        if tool_resources is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "`PipelineWorker(tool_resources=...)` is deprecated since 1.2.0, "
                    "use `app_resources` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if app_resources is None:
                app_resources = tool_resources
        self._params = params or PipelineParams()
        self._additional_span_attributes = additional_span_attributes or {}
        self._cancel_on_idle_timeout = cancel_on_idle_timeout
        self._cancel_runner_on_idle_timeout = cancel_runner_on_idle_timeout
        self._cancel_timeout_secs = cancel_timeout_secs
        self._clock = clock or SystemClock()
        self._conversation_id = conversation_id
        self._enable_tracing = enable_tracing and is_tracing_available()
        self._enable_turn_tracking = enable_turn_tracking
        self._idle_timeout_secs = idle_timeout_secs
        self._app_resources = app_resources
        observers = observers or []
        self._turn_tracking_observer: TurnTrackingObserver | None = None
        self._user_bot_latency_observer: UserBotLatencyObserver | None = None
        self._turn_trace_observer: TurnTraceObserver | None = None
        self._tracing_context: TracingContext | None = None
        if self._enable_turn_tracking:
            self._turn_tracking_observer = TurnTrackingObserver()
            observers.append(self._turn_tracking_observer)
        if self._enable_tracing and self._turn_tracking_observer:
            # Create pipeline-scoped tracing context
            self._tracing_context = TracingContext()
            # Create latency observer for tracing
            self._user_bot_latency_observer = UserBotLatencyObserver()
            observers.append(self._user_bot_latency_observer)
            # Create turn trace observer with latency tracking
            self._turn_trace_observer = TurnTraceObserver(
                self._turn_tracking_observer,
                latency_tracker=self._user_bot_latency_observer,
                conversation_id=self._conversation_id,
                additional_span_attributes=self._additional_span_attributes,
                tracing_context=self._tracing_context,
            )
            observers.append(self._turn_trace_observer)

        self._finished = False
        self._cancelled = False

        # This queue is the queue used to push frames to the pipeline.
        self._push_queue = asyncio.Queue()
        self._process_push_task: asyncio.Task | None = None

        # This is the heartbeat queue. When a heartbeat frame is received in the
        # down queue we add it to the heartbeat queue for processing.
        self._heartbeat_queue = asyncio.Queue()
        self._heartbeat_push_task: asyncio.Task | None = None
        self._heartbeat_monitor_task: asyncio.Task | None = None

        # RTVI support
        self._rtvi = None
        prepend_rtvi = False
        external_rtvi = self._find_processor(pipeline, RTVIProcessor)
        external_observer_found = any(isinstance(o, RTVIObserver) for o in observers)

        if external_rtvi and not external_observer_found:
            logger.error(
                f"{self}: RTVIProcessor found in pipeline but no RTVIObserver in observers. "
                "Make sure to add both."
            )
        elif not external_rtvi and external_observer_found:
            logger.error(
                f"{self}: RTVIObserver found in observers but no RTVIProcessor in pipeline. "
                "Make sure to add both."
            )
        elif external_rtvi and external_observer_found:
            logger.warning(
                f"{self}: RTVIProcessor and RTVIObserver found, skipping default ones. "
                "They are both added by default, no need to add them yourself."
            )
            self._rtvi = external_rtvi
        elif enable_rtvi:
            self._rtvi = rtvi_processor or RTVIProcessor()
            observers.append(self._rtvi.create_rtvi_observer(params=rtvi_observer_params))
            prepend_rtvi = True

        if self._rtvi:
            # Automatically call RTVIProcessor.set_bot_ready()
            @self.rtvi.event_handler("on_client_ready")
            async def on_client_ready(rtvi: RTVIProcessor):
                await rtvi.set_bot_ready()

            # Republish inbound client UI messages onto the bus so
            # UIWorker subscribers can dispatch them.
            @self.rtvi.event_handler("on_ui_message")
            async def on_ui_message(rtvi: RTVIProcessor, message):
                await self._republish_ui_message_on_bus(message)

        # This is the idle event. When selected frames are pushed from any
        # processor we consider the pipeline is not idle. We use an observer
        # which will be listening any part of the pipeline.
        self._idle_event = asyncio.Event()
        self._idle_monitor_task: asyncio.Task | None = None
        if self._idle_timeout_secs:
            idle_frame_observer = IdleFrameObserver(
                idle_event=self._idle_event,
                idle_timeout_frames=idle_timeout_frames,
            )
            observers.append(idle_frame_observer)

        # This event is used to indicate the StartFrame has been received at the
        # end of the pipeline.
        self._pipeline_start_event = asyncio.Event()

        # This event is used to indicate a finalize frame (e.g. EndFrame,
        # StopFrame) has been received at the end of the pipeline.
        self._pipeline_end_event = asyncio.Event()

        # When bridged, wrap the user pipeline with bus edge processors
        # so frames tee onto the bus at the source/sink and incoming bus
        # frames are injected back into the local pipeline. The edges
        # read the worker's bus lazily, so the bus only needs to be set
        # (via ``attach()``) before ``run()`` is called.
        if bridged is not None:
            edge_source = _BusEdgeProcessor(
                worker=self,
                direction=FrameDirection.UPSTREAM,
                bridges=bridged,
                exclude_frames=exclude_frames,
                name=f"{self}::EdgeSource",
            )
            edge_sink = _BusEdgeProcessor(
                worker=self,
                direction=FrameDirection.DOWNSTREAM,
                bridges=bridged,
                exclude_frames=exclude_frames,
                name=f"{self}::EdgeSink",
            )
            pipeline = Pipeline([edge_source, pipeline, edge_sink])

        # This is the final pipeline. It is composed of a source processor,
        # followed by the user pipeline, and ending with a sink processor. The
        # source allows us to receive and react to upstream frames, and the sink
        # allows us to receive and react to downstream frames.
        source = PipelineSource(self._source_push_frame, name=f"{self}::Source")
        self._sink = PipelineSink(self._sink_push_frame, name=f"{self}::Sink")
        # Only prepend the RTVIProcessor if we created it ourselves. When the
        # user already placed it inside their pipeline we must not insert it
        # again or it will appear twice in the frame chain.
        processors = [self._rtvi, pipeline] if prepend_rtvi else [pipeline]
        self._pipeline = Pipeline(processors, source=source, sink=self._sink)

        # The worker observer acts as a proxy to the provided observers. This way,
        # we only need to pass a single observer (using the StartFrame) which
        # then just acts as a proxy.
        self._observer = WorkerObserver(observers=observers)

        # These events can be used to check which frames make it to the source
        # or sink processors. Instead of calling the event handlers for every
        # frame the user needs to specify which events they are interested
        # in. This is mainly for efficiency reason because each event handler
        # creates a worker and most likely you only care about one or two frame
        # types.
        self._reached_upstream_types: set[type[Frame]] = set()
        self._reached_downstream_types: set[type[Frame]] = set()
        self._register_event_handler("on_frame_reached_upstream")
        self._register_event_handler("on_frame_reached_downstream")
        self._register_event_handler("on_heartbeat_timeout")
        self._register_event_handler("on_idle_timeout")
        self._register_event_handler("on_pipeline_started")
        self._register_event_handler("on_pipeline_finished")
        self._register_event_handler("on_pipeline_error")

        # Bridge pipeline lifecycle to the BaseWorker lifecycle so the bus
        # registry sees this worker as ready/finished.
        @self.event_handler("on_pipeline_started")
        async def on_started(_task, _frame):
            await self.start()

        @self.event_handler("on_pipeline_finished")
        async def on_finished(_task, _frame):
            await self.stop()

    @property
    def params(self) -> PipelineParams:
        """Get the pipeline parameters for this worker.

        Returns:
            The pipeline parameters configuration.
        """
        return self._params

    @property
    def bridged(self) -> bool:
        """Whether this pipeline is bridged onto the bus."""
        return self._bridged is not None

    @property
    def app_resources(self) -> Any:
        """Get the application-defined resources passed to this worker.

        This is the same object passed to the constructor as
        ``app_resources``. Tool handlers can also access it via
        ``FunctionCallParams.app_resources``. The framework returns the
        original reference; mutations are visible to all callers.

        Returns:
            The application-defined resources, or ``None`` if none were
            passed.
        """
        return self._app_resources

    @property
    def pipeline(self) -> BasePipeline:
        """Get the full pipeline managed by this pipeline worker.

        This will also include any internal processors added by the pipeline worker.

        Returns:
            The pipeline managed by the pipeline worker.
        """
        return self._pipeline

    @property
    def turn_tracking_observer(self) -> TurnTrackingObserver | None:
        """Get the turn tracking observer if enabled.

        Returns:
            The turn tracking observer instance or None if not enabled.
        """
        return self._turn_tracking_observer

    @property
    def turn_trace_observer(self) -> TurnTraceObserver | None:
        """Get the turn trace observer if enabled.

        Returns:
            The turn trace observer instance or None if not enabled.
        """
        return self._turn_trace_observer

    @property
    def rtvi(self) -> RTVIProcessor:
        """Get the RTVI processor if RTVI is enabled.

        Returns:
            The RTVI processor added to the pipeline when RTVI is enabled.
        """
        if not self._rtvi:
            raise Exception(f"{self} RTVI is not enabled.")
        return self._rtvi

    @property
    def reached_upstream_types(self) -> tuple[type[Frame], ...]:
        """Get the currently configured upstream frame type filters.

        Returns:
            Tuple of frame types that trigger the on_frame_reached_upstream event.
        """
        return tuple(self._reached_upstream_types)

    @property
    def reached_downstream_types(self) -> tuple[type[Frame], ...]:
        """Get the currently configured downstream frame type filters.

        Returns:
            Tuple of frame types that trigger the on_frame_reached_downstream event.
        """
        return tuple(self._reached_downstream_types)

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

    def set_reached_upstream_filter(self, types: tuple[type[Frame], ...]):
        """Set which frame types trigger the on_frame_reached_upstream event.

        Args:
            types: Tuple of frame types to monitor for upstream events.
        """
        self._reached_upstream_types = set(types)

    def set_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
        """Set which frame types trigger the on_frame_reached_downstream event.

        Args:
            types: Tuple of frame types to monitor for downstream events.
        """
        self._reached_downstream_types = set(types)

    def add_reached_upstream_filter(self, types: tuple[type[Frame], ...]):
        """Add frame types to trigger the on_frame_reached_upstream event.

        Args:
            types: Tuple of frame types to add to upstream monitoring.
        """
        self._reached_upstream_types.update(types)

    def add_reached_downstream_filter(self, types: tuple[type[Frame], ...]):
        """Add frame types to trigger the on_frame_reached_downstream event.

        Args:
            types: Tuple of frame types to add to downstream monitoring.
        """
        self._reached_downstream_types.update(types)

    def has_finished(self) -> bool:
        """Check if the pipeline worker has finished execution.

        This indicates whether the worker has finished, meaning all processors
        have stopped.

        Returns:
            True if all processors have stopped and the worker is complete.
        """
        return self._finished

    async def stop_when_done(self):
        """Schedule the pipeline to stop after processing all queued frames.

        Sends an EndFrame to gracefully terminate the pipeline once all
        current processing is complete.
        """
        logger.debug(f"Task {self} scheduled to stop when done")
        await self.queue_frame(EndFrame())

    async def cancel(self, *, reason: str | None = None):
        """Request the running pipeline to cancel.

        Args:
            reason: Optional reason to indicate why the pipeline is being cancelled.
        """
        if not self._finished:
            await self._cancel(reason=reason)

    async def run(self, params: WorkerParams):
        """Start and manage the pipeline execution until completion or cancellation.

        Args:
            params: Configuration parameters for pipeline execution.
        """
        if self.has_finished():
            return

        # Setup processors.
        await self._setup(params)

        # Create all main tasks and wait for the main push worker. This is the
        # worker that pushes frames to the very beginning of our pipeline (i.e. to
        # our controlled source processor).
        await self._create_tasks()

        try:
            # Wait for pipeline to finish.
            await self._wait_for_pipeline_finished()
        except asyncio.CancelledError:
            logger.debug(f"Pipeline worker {self} got cancelled from outside...")
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
            # 1. The pipeline worker has finished (try case).
            # 2. By an asyncio worker cancellation (except case).
            logger.debug(f"Pipeline worker {self} is finishing...")
            await self._cancel_tasks()
            self._print_dangling_tasks()
            self._finished = True
            logger.debug(f"Pipeline worker {self} has finished")

    async def queue_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Queue a single frame to be pushed through the pipeline.

        Downstream frames are pushed from the beginning of the pipeline.
        Upstream frames are pushed from the end of the pipeline.

        Args:
            frame: The frame to be processed.
            direction: The direction to push the frame. Defaults to downstream.
        """
        if direction == FrameDirection.DOWNSTREAM:
            await self._push_queue.put(frame)
        else:
            await self._sink.queue_frame(frame, direction)

    async def queue_frames(
        self,
        frames: Iterable[Frame] | AsyncIterable[Frame],
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ):
        """Queue multiple frames to be pushed through the pipeline.

        Downstream frames are pushed from the beginning of the pipeline.
        Upstream frames are pushed from the end of the pipeline.

        Args:
            frames: An iterable or async iterable of frames to be processed.
            direction: The direction to push the frames. Defaults to downstream.
        """
        if isinstance(frames, AsyncIterable):
            async for frame in frames:
                await self.queue_frame(frame, direction)
        elif isinstance(frames, Iterable):
            for frame in frames:
                await self.queue_frame(frame, direction)

    async def flush_pipeline(self, timeout: float = 5.0) -> bool:
        """Flush all in-flight frames from the pipeline and wait for it to drain.

        Pushes a :class:`~pipecat.frames.frames.PipelineFlushFrame` downstream;
        the sink bounces it back upstream and the source sets its event once it
        completes the round-trip, signalling that every frame queued ahead of it
        has been processed. The probe is injected straight into the pipeline so
        it bypasses any ``queue_frame`` override (e.g. tool-call deferral).

        Args:
            timeout: Seconds to wait before giving up. On timeout a warning is
                logged and ``False`` is returned rather than blocking forever
                (e.g. if a processor swallows the probe).

        Returns:
            True if the pipeline drained, False if the wait timed out.
        """
        event = asyncio.Event()
        await self._pipeline.queue_frame(PipelineFlushFrame(event=event))
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except TimeoutError:
            logger.warning(f"{self}: pipeline flush timed out after {timeout}s")
            return False

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle outbound bus messages: TTS playback and RTVI UI translation.

        Runs the base lifecycle/job dispatch first. A ``BusTTSSpeakMessage``
        targeted at this worker is queued as a ``TTSSpeakFrame`` (pipelines
        without a TTS service let it flow through). When this worker owns the
        RTVI processor, UI carriers produced by a ``UIWorker``
        (``BusUIDataMessage`` subclasses) are translated into RTVI frames by
        ``_handle_ui_bus_message``; other workers skip the translation.
        """
        await super().on_bus_message(message)

        # ``BaseWorker.on_bus_message`` already drops targeted messages for
        # other workers, but it returns early before reaching here -- re-apply
        # the filter before queueing pipeline frames.
        if message.target and message.target != self.name:
            return

        if isinstance(message, BusTTSSpeakMessage):
            await self.queue_frame(
                TTSSpeakFrame(text=message.text, append_to_context=message.append_to_context)
            )
            return

        if self._rtvi and isinstance(message, BusUIDataMessage):
            await self._handle_ui_bus_message(message)

    async def _handle_ui_bus_message(self, message: BusUIDataMessage) -> None:
        """Translate a UI carrier into the matching RTVI frame and queue it.

        Called only when this worker owns the RTVI processor. The
        ``RTVIObserver`` later wraps the queued frame into a typed
        ``ui-command`` / ``ui-job-group`` envelope for the client. Inbound
        carriers (e.g. ``BusUIEventMessage``) match no branch and are ignored.
        """
        frame: Frame | None = None
        if isinstance(message, BusUICommandMessage):
            frame = RTVIUICommandFrame(
                command=message.command_name,
                payload=message.payload,
            )
        elif isinstance(message, BusUIJobGroupStartedMessage):
            frame = RTVIUIJobGroupFrame(
                data=UIJobGroupStartedData(
                    job_id=message.job_id,
                    workers=list(message.workers or []),
                    label=message.label,
                    cancellable=message.cancellable,
                    at=message.at,
                )
            )
        elif isinstance(message, BusUIJobUpdateMessage):
            frame = RTVIUIJobGroupFrame(
                data=UIJobUpdateData(
                    job_id=message.job_id,
                    worker_name=message.worker_name,
                    data=message.data,
                    at=message.at,
                )
            )
        elif isinstance(message, BusUIJobCompletedMessage):
            frame = RTVIUIJobGroupFrame(
                data=UIJobCompletedData(
                    job_id=message.job_id,
                    worker_name=message.worker_name,
                    status=message.status,
                    response=message.response,
                    at=message.at,
                )
            )
        elif isinstance(message, BusUIJobGroupCompletedMessage):
            frame = RTVIUIJobGroupFrame(
                data=UIJobGroupCompletedData(
                    job_id=message.job_id,
                    at=message.at,
                )
            )

        if frame is not None:
            await self.queue_frame(frame)

    async def _republish_ui_message_on_bus(
        self, message: UIEventMessage | UISnapshotMessage | UICancelJobGroupMessage
    ) -> None:
        """Republish an inbound client UI message onto the bus.

        Translates a typed RTVI UI message from the client
        (``UIEventMessage``, ``UISnapshotMessage``, or
        ``UICancelJobGroupMessage``) into a single ``BusUIEventMessage``
        carrier so ``UIWorker`` subscribers can dispatch it. This is the
        inbound counterpart of :meth:`on_bus_message`. Unrecognized
        message types are ignored.
        """
        if isinstance(message, UIEventMessage):
            event_name = message.data.event
            payload = message.data.payload
        elif isinstance(message, UISnapshotMessage):
            event_name = _UI_SNAPSHOT_BUS_EVENT_NAME
            payload = message.data.tree.model_dump(exclude_none=True)
        elif isinstance(message, UICancelJobGroupMessage):
            event_name = _UI_CANCEL_JOB_GROUP_BUS_EVENT_NAME
            payload = {
                "job_id": message.data.job_id,
                "reason": message.data.reason,
            }
        else:
            return
        await self.send_bus_message(
            BusUIEventMessage(
                source=self.name,
                target=None,
                event_name=event_name,
                payload=payload,
            )
        )

    async def _cancel(self, *, reason: str | None = None):
        """Internal cancellation logic for the pipeline worker.

        Args:
            reason: Optional reason to indicate why the pipeline is being cancelled.
        """
        if not self._cancelled:
            logger.debug(f"Cancelling pipeline worker {self}")
            self._cancelled = True
            if not self._pipeline_start_event.is_set():
                self._pipeline_start_event.set()
            await self.queue_frame(CancelFrame(reason=reason))

    async def _create_tasks(self):
        """Create and start all pipeline processing tasks."""
        self._process_push_task = self.create_task(self._process_push_queue())
        return self._process_push_task

    def _maybe_start_heartbeat_tasks(self):
        """Start heartbeat tasks if heartbeats are enabled and not already running."""
        if self._params.enable_heartbeats and self._heartbeat_push_task is None:
            self._heartbeat_push_task = self.create_task(self._heartbeat_push_handler())
            self._heartbeat_monitor_task = self.create_task(self._heartbeat_monitor_handler())

    def _maybe_start_idle_task(self):
        """Start idle monitoring worker if idle timeout is configured."""
        if self._idle_timeout_secs:
            self._idle_monitor_task = self.create_task(self._idle_monitor_handler())

    async def _cancel_tasks(self):
        """Cancel all running pipeline tasks."""
        if self._process_push_task:
            await self.cancel_task(self._process_push_task)
            self._process_push_task = None

        await self._maybe_cancel_heartbeat_tasks()
        await self._maybe_cancel_idle_task()

    async def _maybe_cancel_heartbeat_tasks(self):
        """Cancel heartbeat tasks if they are running."""
        if not self._params.enable_heartbeats:
            return

        if self._heartbeat_push_task:
            await self.cancel_task(self._heartbeat_push_task)
            self._heartbeat_push_task = None

        if self._heartbeat_monitor_task:
            await self.cancel_task(self._heartbeat_monitor_task)
            self._heartbeat_monitor_task = None

    async def _maybe_cancel_idle_task(self):
        """Cancel idle monitoring worker if it is running."""
        if self._idle_monitor_task:
            await self.cancel_task(self._idle_monitor_task)
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
            except TimeoutError:
                logger.warning(
                    f"{self}: timeout waiting for {frame} to reach the end of the pipeline (being blocked somewhere?)."
                )
            finally:
                await self._call_event_handler("on_pipeline_finished", frame)

        logger.debug(f"{self}: Closing. Waiting for {frame} to reach the end of the pipeline...")

        if isinstance(frame, CancelFrame):
            await wait_for_cancel()
        else:
            await self._pipeline_end_event.wait()
            logger.debug(f"{self}: {frame} reached the end of the pipeline, pipeline is closing.")

        self._pipeline_end_event.clear()

        # We are really done. Setting ``_finished_event`` makes
        # ``BaseWorker.wait()`` resolve for callers awaiting this worker.
        self._finished_event.set()

    async def _wait_for_pipeline_finished(self):
        await self._finished_event.wait()
        # Make sure we wait for the main worker to complete.
        if self._process_push_task:
            await self._process_push_task
            self._process_push_task = None

    async def _setup(self, params: WorkerParams):
        """Set up the pipeline worker and all processors."""
        await super().setup(self._task_manager or params.task_manager)

        setup = FrameProcessorSetup(
            clock=self._clock,
            task_manager=self.task_manager,
            observer=self._observer,
            pipeline_worker=self,
            # Populate the deprecated `tool_resources` field for backwards
            # compatibility with custom FrameProcessor subclasses whose
            # ``setup()`` overrides still read it. Reading the field emits a
            # DeprecationWarning; new code should read
            # ``setup.pipeline_worker.app_resources`` instead.
            tool_resources=self._app_resources,
        )
        await self._pipeline.setup(setup)

        # Do any additional pipeline worker setup externally.
        await self._load_setup_files()

        # Start worker observer.
        await self._observer.setup(self.task_manager)
        await self._observer.start()

    async def _cleanup(self, cleanup_pipeline: bool):
        """Clean up the pipeline worker and processors."""
        # Cleanup base object.
        await self.cleanup()

        # Cleanup observers.
        await self._observer.stop()
        await self._observer.cleanup()

        # End conversation tracing if it's active - this will also close any active turn span
        if self._enable_tracing and self._turn_trace_observer:
            self._turn_trace_observer.end_conversation_tracing()

        # Cleanup pipeline processors.
        if cleanup_pipeline:
            await self._pipeline.cleanup()

    async def _handle_worker_end(self, message: BusEndWorkerMessage) -> None:
        """End the pipeline after propagating end to children.

        Drives shutdown through the pipeline (``EndFrame``) so
        ``_finished_event`` fires once the frame drains through the
        sink, rather than calling ``stop()`` directly.
        """
        logger.debug(f"Worker '{self}': received end")
        await self._propagate_end_to_children(message)
        await self.queue_frame(EndFrame(reason=message.reason))

    async def _handle_worker_cancel(self, message: BusCancelWorkerMessage) -> None:
        """Cancel the pipeline after propagating cancel to children.

        Drives shutdown through the pipeline (``CancelFrame``) so
        ``_finished_event`` fires once the frame drains, rather than
        calling ``stop()`` directly.
        """
        logger.debug(f"Worker '{self}': received cancel")
        await self._propagate_cancel_to_children(message)
        await self.cancel(reason=message.reason)

    async def _process_push_queue(self):
        """Process frames from the push queue and send them through the pipeline.

        This is the worker that runs the pipeline for the first time by sending
        a StartFrame and by pushing any other frames queued by the user. It runs
        until the worker is cancelled or stopped (e.g. with an EndFrame).
        """
        self._clock.start()

        self._maybe_start_idle_task()

        start_frame = StartFrame(
            audio_in_sample_rate=self._params.audio_in_sample_rate,
            audio_out_sample_rate=self._params.audio_out_sample_rate,
            enable_metrics=self._params.enable_metrics,
            enable_tracing=self._enable_tracing,
            enable_usage_metrics=self._params.enable_usage_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb,
            tracing_context=self._tracing_context,
        )
        start_frame.metadata = self._create_start_metadata()
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

        This is the worker that processes frames coming upstream from the
        pipeline. These frames might indicate, for example, that we want the
        pipeline to be stopped (e.g. EndWorkerFrame) in which case we would send
        an EndFrame down the pipeline.
        """
        if isinstance(frame, tuple(self._reached_upstream_types)):
            await self._call_event_handler("on_frame_reached_upstream", frame)

        if isinstance(frame, PipelineFlushFrame):
            # The flush probe completed its round-trip (down to the sink, back up
            # to the source). Everything queued ahead of it has been processed;
            # release whoever is awaiting it.
            logger.debug(f"{self}: flush probe reached source — pipeline drained")
            if frame.event:
                frame.event.set()
            return

        if isinstance(frame, EndWorkerFrame):
            # Tell the worker we should end nicely.
            logger.debug(f"{self}: received end worker frame upstream {frame}")
            await self.queue_frame(EndFrame(reason=frame.reason))
        elif isinstance(frame, CancelWorkerFrame):
            # Tell the worker we should end right away.
            logger.debug(f"{self}: received cancel worker frame upstream {frame}")
            await self.queue_frame(CancelFrame(reason=frame.reason))
        elif isinstance(frame, StopWorkerFrame):
            # Tell the worker we should stop nicely.
            logger.debug(f"{self}: received stop worker frame upstream {frame}")
            await self.queue_frame(StopFrame())
        elif isinstance(frame, InterruptionWorkerFrame):
            # Tell the worker we should interrupt the pipeline. Note that we are
            # bypassing the push queue and directly queue into the
            # pipeline. This is in case the push worker is blocked waiting for a
            # pipeline-ending frame to finish traversing the pipeline.
            logger.debug(f"{self}: received interruption worker frame upstream {frame}")
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
        processors have handled the EndFrame and therefore we can exit the worker
        cleanly.
        """
        if isinstance(frame, tuple(self._reached_downstream_types)):
            await self._call_event_handler("on_frame_reached_downstream", frame)

        if isinstance(frame, PipelineFlushFrame):
            # The flush probe reached the sink. Bounce the same instance back
            # upstream so it returns to the source (carrying its event) and the
            # round-trip drains both directions.
            logger.debug(f"{self}: flush probe reached sink — bouncing upstream")
            await self._sink.push_frame(frame, FrameDirection.UPSTREAM)
            return

        if isinstance(frame, StartFrame):
            await self._call_event_handler("on_pipeline_started", frame)
            await self._observer.on_pipeline_started()

            # Start heartbeat tasks now that StartFrame has been processed
            # by all processors in the pipeline
            self._maybe_start_heartbeat_tasks()

            self._pipeline_start_event.set()
        elif isinstance(frame, EndFrame):
            await self._call_event_handler("on_pipeline_finished", frame)
            self._pipeline_end_event.set()
        elif isinstance(frame, StopFrame):
            await self._call_event_handler("on_pipeline_finished", frame)
            self._pipeline_end_event.set()
        elif isinstance(frame, CancelFrame):
            self._pipeline_end_event.set()
        elif isinstance(frame, HeartbeatFrame):
            await self._heartbeat_queue.put(frame)
        elif isinstance(frame, EndWorkerFrame):
            logger.debug(f"{self}: received end worker frame downstream {frame}")
            await self.queue_frame(EndWorkerFrame(reason=frame.reason), FrameDirection.UPSTREAM)
        elif isinstance(frame, StopWorkerFrame):
            logger.debug(f"{self}: received stop worker frame downstream {frame}")
            await self.queue_frame(StopWorkerFrame(), FrameDirection.UPSTREAM)
        elif isinstance(frame, CancelWorkerFrame):
            logger.debug(f"{self}: received cancel worker frame downstream {frame}")
            await self.queue_frame(CancelWorkerFrame(reason=frame.reason), FrameDirection.UPSTREAM)
        elif isinstance(frame, InterruptionWorkerFrame):
            logger.debug(f"{self}: received interruption worker frame downstream {frame}")
            await self.queue_frame(InterruptionWorkerFrame(), FrameDirection.UPSTREAM)

    async def _heartbeat_push_handler(self):
        """Push heartbeat frames at regular intervals."""
        while True:
            # Don't use `queue_frame()` because if an EndFrame is queued the
            # worker will just stop waiting for the pipeline to finish not
            # allowing more frames to be pushed.
            await self._pipeline.queue_frame(HeartbeatFrame(timestamp=self._clock.get_time()))
            await asyncio.sleep(self._params.heartbeats_period_secs)

    async def _heartbeat_monitor_handler(self):
        """Monitor heartbeat frames for processing time and timeout detection.

        Logs the time each heartbeat takes to traverse the pipeline. If no
        heartbeat arrives within ``heartbeats_monitor_secs``, logs a warning
        and fires ``on_heartbeat_timeout``. The event fires repeatedly every
        ``heartbeats_monitor_secs`` for as long as the stall persists.
        """
        wait_time = self._params.heartbeats_monitor_secs
        while True:
            try:
                frame = await asyncio.wait_for(self._heartbeat_queue.get(), timeout=wait_time)
                process_time = (self._clock.get_time() - frame.timestamp) / 1_000_000_000
                logger.trace(f"{self}: heartbeat frame processed in {process_time} seconds")
                self._heartbeat_queue.task_done()
            except TimeoutError:
                logger.warning(
                    f"{self}: heartbeat frame not received for more than {wait_time} seconds"
                )
                await self._call_event_handler("on_heartbeat_timeout")

    async def _idle_monitor_handler(self):
        """Monitor pipeline activity and detect idle conditions.

        Tracks frame activity and triggers idle timeout events when the
        pipeline hasn't received relevant frames within the timeout period.

        Note: Heartbeats are excluded from idle detection.
        """
        running = True
        while running:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._idle_timeout_secs)
                self._idle_event.clear()
            except TimeoutError:
                running = await self._idle_timeout_detected()

    async def _idle_timeout_detected(self) -> bool:
        """Handle idle timeout detection and optional cancellation.

        Returns:
            Whether the pipeline worker should continue running.
        """
        # If we are cancelling, just exit the worker.
        if self._cancelled:
            return False

        logger.warning("Idle timeout detected.")
        await self._call_event_handler("on_idle_timeout")
        if not self._cancel_on_idle_timeout:
            return True

        logger.warning("Idle pipeline detected, cancelling pipeline worker...")
        await self.cancel(reason="idle timeout")
        if self._cancel_runner_on_idle_timeout:
            logger.warning("...and cancelling the runner.")
            # ``BaseWorker.cancel`` sends ``BusCancelMessage`` on the bus
            # so the runner broadcasts cancellation to every other root
            # worker too. This worker's pipeline is already cancelling
            # from the call above.
            await BaseWorker.cancel(self, reason="idle timeout")
        return False

    async def _load_setup_files(self):
        """Run ``setup_pipeline_worker`` from each file in ``PIPECAT_SETUP_FILES``.

        A setup file may define ``setup_pipeline_worker(worker)`` to attach event
        handlers, observers, or other per-worker wiring. The legacy name
        ``setup_pipeline_task`` is still recognized but emits a
        ``DeprecationWarning``.
        """
        await run_setup_hook(
            target=self,
            function_name="setup_pipeline_worker",
            deprecated_function_name="setup_pipeline_task",
        )

    def _create_start_metadata(self) -> dict[str, Any]:
        """Build and return start metadata including user-provided values."""
        start_metadata = {}

        # Update with user provided metadata.
        start_metadata.update(self._params.start_metadata)

        return start_metadata

    def _find_processor(self, processor: FrameProcessor, processor_type: type[T]) -> T | None:
        """Recursively find a processor of the given type in the pipeline."""
        if isinstance(processor, processor_type):
            return processor

        for p in processor.processors:
            found = self._find_processor(p, processor_type)
            if found:
                return found
        return None


@deprecated(
    "`PipelineTask` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `PipelineWorker` instead."
)
class PipelineTask(PipelineWorker):
    """Deprecated alias for :class:`PipelineWorker`.

    .. deprecated:: 1.3.0
        Use :class:`PipelineWorker` instead. :class:`PipelineTask` will be removed
        in 2.0.0.
    """

    pass


@deprecated(
    "`PipelineTaskParams` is deprecated since 1.3.0 and will be removed in 2.0.0. "
    "Use `WorkerParams` instead."
)
@dataclass
class PipelineTaskParams(WorkerParams):
    """Deprecated alias for :class:`~pipecat.workers.base_worker.WorkerParams`.

    .. deprecated:: 1.3.0
        Use :class:`~pipecat.workers.base_worker.WorkerParams` instead.
        Will be removed in 2.0.0.
    """

    pass
