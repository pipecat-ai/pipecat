#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
from typing import Any, AsyncIterable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

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
    LLMFullResponseEndFrame,
    MetricsFrame,
    StartFrame,
    StopFrame,
    StopTaskFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.base_task import BaseTask
from pipecat.pipeline.task_observer import TaskObserver
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.utils.asyncio import BaseTaskManager, TaskManager
from pipecat.utils.tracing.setup import is_tracing_available
from pipecat.utils.tracing.turn_trace_observer import TurnTraceObserver

HEARTBEAT_SECONDS = 1.0
HEARTBEAT_MONITOR_SECONDS = HEARTBEAT_SECONDS * 5


class PipelineParams(BaseModel):
    """Configuration parameters for pipeline execution.

    Attributes:
        allow_interruptions: Whether to allow pipeline interruptions.
        audio_in_sample_rate: Input audio sample rate in Hz.
        audio_out_sample_rate: Output audio sample rate in Hz.
        enable_heartbeats: Whether to enable heartbeat monitoring.
        enable_metrics: Whether to enable metrics collection.
        enable_usage_metrics: Whether to enable usage metrics.
        heartbeats_period_secs: Period between heartbeats in seconds.
        observers: [deprecated] Use `observers` arg in `PipelineTask` class.
        report_only_initial_ttfb: Whether to report only initial time to first byte.
        send_initial_empty_metrics: Whether to send initial empty metrics.
        start_metadata: Additional metadata for pipeline start.
        interruption_strategies: Strategies for bot interruption behavior.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    allow_interruptions: bool = False
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    enable_heartbeats: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    heartbeats_period_secs: float = HEARTBEAT_SECONDS
    observers: List[BaseObserver] = Field(default_factory=list)
    report_only_initial_ttfb: bool = False
    send_initial_empty_metrics: bool = True
    start_metadata: Dict[str, Any] = Field(default_factory=dict)
    interruption_strategies: List[BaseInterruptionStrategy] = Field(default_factory=list)


class PipelineTaskSource(FrameProcessor):
    """Source processor for pipeline tasks that handles frame routing.

    This is the source processor that is linked at the beginning of the
    pipeline given to the pipeline task. It allows us to easily push frames
    downstream to the pipeline and also receive upstream frames coming from the
    pipeline.

    Args:
        up_queue: Queue for upstream frame processing.

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
    """Sink processor for pipeline tasks that handles final frame processing.

    This is the sink processor that is linked at the end of the pipeline
    given to the pipeline task. It allows us to receive downstream frames and
    act on them, for example, waiting to receive an EndFrame.

    Args:
        down_queue: Queue for downstream frame processing.
    """

    def __init__(self, down_queue: asyncio.Queue, **kwargs):
        super().__init__(**kwargs)
        self._down_queue = down_queue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self._down_queue.put(frame)


class PipelineTask(BaseTask):
    """Manages the execution of a pipeline, handling frame processing and task lifecycle.

    It has a couple of event handlers `on_frame_reached_upstream` and
    `on_frame_reached_downstream` that are called when upstream frames or
    downstream frames reach both ends of pipeline. By default, the events
    handlers will not be called unless some filters are set using
    `set_reached_upstream_filter` and `set_reached_downstream_filter`.

       @task.event_handler("on_frame_reached_upstream")
       async def on_frame_reached_upstream(task, frame):
           ...

       @task.event_handler("on_frame_reached_downstream")
       async def on_frame_reached_downstream(task, frame):
           ...

    It also has an event handler that detects when the pipeline is idle. By
    default, a pipeline is idle if no `BotSpeakingFrame` or
    `LLMFullResponseEndFrame` are received within `idle_timeout_secs`.

       @task.event_handler("on_idle_timeout")
       async def on_pipeline_idle_timeout(task):
           ...

    There are also events to know if a pipeline has been started, stopped, ended
    or cancelled.

       @task.event_handler("on_pipeline_started")
       async def on_pipeline_started(task, frame: StartFrame):
           ...

       @task.event_handler("on_pipeline_stopped")
       async def on_pipeline_stopped(task, frame: StopFrame):
           ...

       @task.event_handler("on_pipeline_ended")
       async def on_pipeline_ended(task, frame: EndFrame):
           ...

       @task.event_handler("on_pipeline_cancelled")
       async def on_pipeline_cancelled(task, frame: CancelFrame):
           ...

    Args:
        pipeline: The pipeline to execute.
        params: Configuration parameters for the pipeline.
        observers: List of observers for monitoring pipeline execution.
        clock: Clock implementation for timing operations.
        check_dangling_tasks: Whether to check for processors' tasks finishing properly.
        idle_timeout_secs: Timeout (in seconds) to consider pipeline idle or
            None. If a pipeline is idle the pipeline task will be cancelled
            automatically.
        idle_timeout_frames: A tuple with the frames that should trigger an idle
            timeout if not received withing `idle_timeout_seconds`.
        cancel_on_idle_timeout: Whether the pipeline task should be cancelled if
            the idle timeout is reached.
        enable_turn_tracking: Whether to enable turn tracking.
        enable_turn_tracing: Whether to enable turn tracing.
        conversation_id: Optional custom ID for the conversation.
        additional_span_attributes: Optional dictionary of attributes to propagate as
            OpenTelemetry conversation span attributes.
    """

    def __init__(
        self,
        pipeline: BasePipeline,
        *,
        params: Optional[PipelineParams] = None,
        observers: Optional[List[BaseObserver]] = None,
        clock: Optional[BaseClock] = None,
        task_manager: Optional[BaseTaskManager] = None,
        check_dangling_tasks: bool = True,
        idle_timeout_secs: Optional[float] = 300,
        idle_timeout_frames: Tuple[Type[Frame], ...] = (
            BotSpeakingFrame,
            LLMFullResponseEndFrame,
        ),
        cancel_on_idle_timeout: bool = True,
        enable_turn_tracking: bool = True,
        enable_tracing: bool = False,
        conversation_id: Optional[str] = None,
        additional_span_attributes: Optional[dict] = None,
    ):
        super().__init__()
        self._pipeline = pipeline
        self._clock = clock or SystemClock()
        self._params = params or PipelineParams()
        self._check_dangling_tasks = check_dangling_tasks
        self._idle_timeout_secs = idle_timeout_secs
        self._idle_timeout_frames = idle_timeout_frames
        self._cancel_on_idle_timeout = cancel_on_idle_timeout
        self._enable_turn_tracking = enable_turn_tracking
        self._enable_tracing = enable_tracing and is_tracing_available()
        self._conversation_id = conversation_id
        self._additional_span_attributes = additional_span_attributes or {}
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

        # This queue receives frames coming from the pipeline upstream.
        self._up_queue = asyncio.Queue()
        # This queue receives frames coming from the pipeline downstream.
        self._down_queue = asyncio.Queue()
        # This queue is the queue used to push frames to the pipeline.
        self._push_queue = asyncio.Queue()
        # This is the heartbeat queue. When a heartbeat frame is received in the
        # down queue we add it to the heartbeat queue for processing.
        self._heartbeat_queue = asyncio.Queue()
        # This is the idle queue. When frames are received downstream they are
        # put in the queue. If no frame is received the pipeline is considered
        # idle.
        self._idle_queue = asyncio.Queue()
        # This event is used to indicate a finalize frame (e.g. EndFrame,
        # StopFrame) has been received in the down queue.
        self._pipeline_end_event = asyncio.Event()

        # This is a source processor that we connect to the provided
        # pipeline. This source processor allows up to receive and react to
        # upstream frames.
        self._source = PipelineTaskSource(self._up_queue)
        self._source.link(pipeline)

        # This is a sink processor that we connect to the provided
        # pipeline. This sink processor allows up to receive and react to
        # downstream frames.
        self._sink = PipelineTaskSink(self._down_queue)
        pipeline.link(self._sink)

        # This task maneger will handle all the asyncio tasks created by this
        # PipelineTask and its frame processors.
        self._task_manager = task_manager or TaskManager()

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

    @property
    def params(self) -> PipelineParams:
        """Returns the pipeline parameters of this task."""
        return self._params

    @property
    def turn_tracking_observer(self) -> Optional[TurnTrackingObserver]:
        """Return the turn tracking observer if enabled."""
        return self._turn_tracking_observer

    @property
    def turn_trace_observer(self) -> Optional[TurnTraceObserver]:
        """Return the turn trace observer if enabled."""
        return self._turn_trace_observer

    def add_observer(self, observer: BaseObserver):
        self._observer.add_observer(observer)

    async def remove_observer(self, observer: BaseObserver):
        await self._observer.remove_observer(observer)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._task_manager.set_event_loop(loop)

    def set_reached_upstream_filter(self, types: Tuple[Type[Frame], ...]):
        """Sets which frames will be checked before calling the
        on_frame_reached_upstream event handler.

        """
        self._reached_upstream_types = types

    def set_reached_downstream_filter(self, types: Tuple[Type[Frame], ...]):
        """Sets which frames will be checked before calling the
        on_frame_reached_downstream event handler.

        """
        self._reached_downstream_types = types

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
        """Stops the running pipeline immediately."""
        await self._cancel()

    async def run(self):
        """Starts and manages the pipeline execution until completion or cancellation."""
        if self.has_finished():
            return
        cleanup_pipeline = True
        try:
            # Setup processors.
            await self._setup()

            # Create all main tasks and wait of the main push task. This is the
            # task that pushes frames to the very beginning of our pipeline (our
            # controlled PipelineTaskSource processor).
            push_task = await self._create_tasks()
            await self._task_manager.wait_for_task(push_task)

            # We have already cleaned up the pipeline inside the task.
            cleanup_pipeline = False
        except asyncio.CancelledError:
            # We are awaiting on the push task and it might be cancelled
            # (e.g. Ctrl-C). This means we will get a CancelledError here as
            # well, because you get a CancelledError in every place you are
            # awaiting a task.
            pass
        finally:
            # It's possibe that we get an asyncio.CancelledError from the
            # outside, if so we need to make sure everything gets cancelled
            # properly.
            if cleanup_pipeline:
                await self._cancel()
            await self._cancel_tasks()
            await self._cleanup(cleanup_pipeline)
            if self._check_dangling_tasks:
                self._print_dangling_tasks()
            self._finished = True

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
        if not self._cancelled:
            logger.debug(f"Canceling pipeline task {self}")
            self._cancelled = True
            # Make sure everything is cleaned up downstream. This is sent
            # out-of-band from the main streaming task which is what we want since
            # we want to cancel right away.
            await self._source.push_frame(CancelFrame())
            # Only cancel the push task. Everything else will be cancelled in run().
            await self._task_manager.cancel_task(self._process_push_task)

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

    def _maybe_start_idle_task(self):
        if self._idle_timeout_secs:
            self._idle_monitor_task = self._task_manager.create_task(
                self._idle_monitor_handler(), f"{self}::_idle_monitor_handler"
            )

    async def _cancel_tasks(self):
        await self._observer.stop()

        await self._task_manager.cancel_task(self._process_up_task)
        await self._task_manager.cancel_task(self._process_down_task)

        await self._maybe_cancel_heartbeat_tasks()
        await self._maybe_cancel_idle_task()

    async def _maybe_cancel_heartbeat_tasks(self):
        if self._params.enable_heartbeats:
            await self._task_manager.cancel_task(self._heartbeat_push_task)
            await self._task_manager.cancel_task(self._heartbeat_monitor_task)

    async def _maybe_cancel_idle_task(self):
        if self._idle_timeout_secs:
            await self._task_manager.cancel_task(self._idle_monitor_task)

    def _initial_metrics_frame(self) -> MetricsFrame:
        processors = self._pipeline.processors_with_metrics()
        data = []
        for p in processors:
            data.append(TTFBMetricsData(processor=p.name, value=0.0))
            data.append(ProcessingMetricsData(processor=p.name, value=0.0))
        return MetricsFrame(data=data)

    async def _wait_for_pipeline_end(self):
        await self._pipeline_end_event.wait()
        self._pipeline_end_event.clear()

    async def _setup(self):
        setup = FrameProcessorSetup(
            clock=self._clock,
            task_manager=self._task_manager,
            observer=self._observer,
        )
        await self._source.setup(setup)
        await self._pipeline.setup(setup)
        await self._sink.setup(setup)

    async def _cleanup(self, cleanup_pipeline: bool):
        # Cleanup base object.
        await self.cleanup()

        # End conversation tracing if it's active - this will also close any active turn span
        if self._enable_tracing and hasattr(self, "_turn_trace_observer"):
            self._turn_trace_observer.end_conversation_tracing()

        # Cleanup pipeline processors.
        await self._source.cleanup()
        if cleanup_pipeline:
            await self._pipeline.cleanup()
        await self._sink.cleanup()

    async def _process_push_queue(self):
        """This is the task that runs the pipeline for the first time by sending
        a StartFrame and by pushing any other frames queued by the user. It runs
        until the tasks is cancelled or stopped (e.g. with an EndFrame).

        """
        self._clock.start()

        self._maybe_start_heartbeat_tasks()
        self._maybe_start_idle_task()

        start_frame = StartFrame(
            allow_interruptions=self._params.allow_interruptions,
            audio_in_sample_rate=self._params.audio_in_sample_rate,
            audio_out_sample_rate=self._params.audio_out_sample_rate,
            enable_metrics=self._params.enable_metrics,
            enable_usage_metrics=self._params.enable_usage_metrics,
            report_only_initial_ttfb=self._params.report_only_initial_ttfb,
            interruption_strategies=self._params.interruption_strategies,
        )
        start_frame.metadata = self._params.start_metadata
        await self._source.queue_frame(start_frame, FrameDirection.DOWNSTREAM)

        if self._params.enable_metrics and self._params.send_initial_empty_metrics:
            await self._source.queue_frame(self._initial_metrics_frame(), FrameDirection.DOWNSTREAM)

        running = True
        cleanup_pipeline = True
        while running:
            frame = await self._push_queue.get()
            await self._source.queue_frame(frame, FrameDirection.DOWNSTREAM)
            if isinstance(frame, (EndFrame, StopFrame)):
                await self._wait_for_pipeline_end()
            running = not isinstance(frame, (CancelFrame, EndFrame, StopFrame))
            cleanup_pipeline = not isinstance(frame, StopFrame)
            self._push_queue.task_done()
        await self._cleanup(cleanup_pipeline)

    async def _process_up_queue(self):
        """This is the task that processes frames coming upstream from the
        pipeline. These frames might indicate, for example, that we want the
        pipeline to be stopped (e.g. EndTaskFrame) in which case we would send
        an EndFrame down the pipeline.

        """
        while True:
            frame = await self._up_queue.get()

            if isinstance(frame, self._reached_upstream_types):
                await self._call_event_handler("on_frame_reached_upstream", frame)

            if isinstance(frame, EndTaskFrame):
                # Tell the task we should end nicely.
                await self.queue_frame(EndFrame())
            elif isinstance(frame, CancelTaskFrame):
                # Tell the task we should end right away.
                await self.queue_frame(CancelFrame())
            elif isinstance(frame, StopTaskFrame):
                # Tell the task we should stop nicely.
                await self.queue_frame(StopFrame())
            elif isinstance(frame, ErrorFrame):
                if frame.fatal:
                    logger.error(f"A fatal error occurred: {frame}")
                    # Cancel all tasks downstream.
                    await self.queue_frame(CancelFrame())
                    # Tell the task we should stop.
                    await self.queue_frame(StopTaskFrame())
                else:
                    logger.warning(f"Something went wrong: {frame}")
            self._up_queue.task_done()

    async def _process_down_queue(self):
        """This tasks process frames coming downstream from the pipeline. For
        example, heartbeat frames or an EndFrame which would indicate all
        processors have handled the EndFrame and therefore we can exit the task
        cleanly.

        """
        while True:
            frame = await self._down_queue.get()

            # Queue received frame to the idle queue so we can monitor idle
            # pipelines.
            await self._idle_queue.put(frame)

            if isinstance(frame, self._reached_downstream_types):
                await self._call_event_handler("on_frame_reached_downstream", frame)

            if isinstance(frame, StartFrame):
                await self._call_event_handler("on_pipeline_started", frame)
            elif isinstance(frame, EndFrame):
                await self._call_event_handler("on_pipeline_ended", frame)
                self._pipeline_end_event.set()
            elif isinstance(frame, StopFrame):
                await self._call_event_handler("on_pipeline_stopped", frame)
                self._pipeline_end_event.set()
            elif isinstance(frame, CancelFrame):
                await self._call_event_handler("on_pipeline_cancelled", frame)
            elif isinstance(frame, HeartbeatFrame):
                await self._heartbeat_queue.put(frame)
            self._down_queue.task_done()

    async def _heartbeat_push_handler(self):
        """This tasks pushes a heartbeat frame every heartbeat period."""
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

    async def _idle_monitor_handler(self):
        """This tasks monitors activity in the pipeline. If no frames are
        received (heartbeats don't count) the pipeline is considered idle.

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

                self._idle_queue.task_done()
            except asyncio.TimeoutError:
                running = await self._idle_timeout_detected()

    async def _idle_timeout_detected(self) -> bool:
        """Logic for when the pipeline is idle.

        Returns:
            bool: Whther the pipeline task is being cancelled or not.
        """
        await self._call_event_handler("on_idle_timeout")
        if self._cancel_on_idle_timeout:
            logger.warning(f"Idle pipeline detected, cancelling pipeline task...")
            await self.cancel()
            return False
        return True

    def _print_dangling_tasks(self):
        tasks = [t.get_name() for t in self._task_manager.current_tasks()]
        if tasks:
            logger.warning(f"Dangling tasks detected: {tasks}")
