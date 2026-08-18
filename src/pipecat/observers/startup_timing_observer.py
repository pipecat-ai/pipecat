#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer for tracking pipeline startup timing.

This module provides an observer that measures how long each processor's takes
during pipeline startup (i.e. setup() and ``StartFrame``). It works by tracking
``setup()`` time and when the ``StartFrame`` arrives at a processor
(``on_process_frame``) versus when it leaves (``on_push_frame``), giving the
exact ``start()`` duration for each processor in the pipeline.

It also measures transport timing — the time from before ``setup()`` to the
first ``BotConnectedFrame`` (SFU transports only) and ``ClientConnectedFrame`` —
via a separate ``on_transport_timing_report`` event.

Example::

    observer = StartupTimingObserver()

    @observer.event_handler("on_startup_timing_report")
    async def on_report(observer, report):
        for t in report.processor_timings:
            print(f"{t.processor_name}: {t.duration_secs:.3f}s")

    @observer.event_handler("on_transport_timing_report")
    async def on_transport(observer, report):
        if report.bot_connected_secs is not None:
            print(f"Bot connected in {report.bot_connected_secs:.3f}s")
        print(f"Client connected in {report.client_connected_secs:.3f}s")

    worker = PipelineWorker(pipeline, observers=[observer])

"""

import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from pipecat.frames.frames import BotConnectedFrame, ClientConnectedFrame, StartFrame
from pipecat.observers.base_observer import (
    BaseObserver,
    FrameProcessed,
    FramePushed,
    ProcessorSetUp,
)
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import PipelineSource
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.asyncio.task_manager import BaseTaskManager

# Internal pipeline types excluded from tracking by default.
_INTERNAL_TYPES = (PipelineSource, BasePipeline)


@dataclass
class _StartFrameInfo:
    """Captured once when the first StartFrame arrives at a processor."""

    frame_id: int
    arrival_ns: int
    wall_clock: float


@dataclass
class _ArrivalInfo:
    """Internal record of when a StartFrame arrived at a processor."""

    processor: FrameProcessor
    arrival_ts_ns: int


class ProcessorStartupTiming(BaseModel):
    """Startup timing for a single processor.

    Parameters:
        processor_name: The name of the processor.
        start_offset_secs: Offset in seconds from the StartFrame to when this
            processor's start() began.
        duration_secs: What the processor cost to get ready, in seconds: its
            setup() and its start() together.
        setup_duration_secs: How long the processor's setup() took, in seconds,
            which is the part of ``duration_secs`` spent connecting.
    """

    processor_name: str
    start_offset_secs: float
    duration_secs: float
    setup_duration_secs: float


class StartupTimingReport(BaseModel):
    """Report of startup timings for all measured processors.

    Parameters:
        start_time: Unix timestamp when the pipeline began setting up.
        total_duration_secs: Wall-clock time from the pipeline starting to set
            up until it had started. Processors are set up concurrently, so
            this is the span rather than the sum of what each cost.
        processor_timings: Per-processor timing data, in pipeline order.
    """

    start_time: float
    total_duration_secs: float
    processor_timings: list[ProcessorStartupTiming] = Field(default_factory=list)


class TransportTimingReport(BaseModel):
    """Time from pipeline start to transport connection milestones.

    Parameters:
        start_time: Unix timestamp when the pipeline began setting up.
        bot_connected_secs: Seconds from the pipeline starting to set up until
            the first BotConnectedFrame (only set for SFU transports).
        client_connected_secs: Seconds from the pipeline starting to set up
            until the first ClientConnectedFrame.
    """

    start_time: float
    bot_connected_secs: float | None = None
    client_connected_secs: float | None = None


class StartupTimingObserver(BaseObserver):
    """Observer that measures processor startup times during pipeline initialization.

    Tracks what each processor costs to get ready: its ``setup()`` and its
    ``start()`` together, the latter measured from the ``StartFrame`` arriving
    at the processor to it being pushed downstream. This captures WebSocket
    connections, API authentication, model loading, and other initialization
    work, most of which happens while the processor is being set up.

    Also measures transport timing, the time from the pipeline starting to set
    up until each connection milestone. A transport connects while it is set
    up, so these can be reached before the ``StartFrame`` is pushed:

    - ``bot_connected_secs``: When the bot joins the transport room
      (SFU transports only, triggered by ``BotConnectedFrame``).
    - ``client_connected_secs``: When a remote participant connects
      (triggered by ``ClientConnectedFrame``).

    By default, internal pipeline processors (``PipelineSource``, ``Pipeline``)
    are excluded from the report. Pass ``processor_types`` to measure only
    specific types.

    Event handlers available:

    - on_startup_timing_report: Called once after startup completes with the full
      timing report.
    - on_transport_timing_report: Called once when the first client connects with a
      TransportTimingReport containing client_connected_secs and bot_connected_secs
      (if available).

    Example::

        observer = StartupTimingObserver(
            processor_types=(STTService, TTSService)
        )

        @observer.event_handler("on_startup_timing_report")
        async def on_report(observer, report):
            for t in report.processor_timings:
                logger.info(f"{t.processor_name}: {t.duration_secs:.3f}s")

        @observer.event_handler("on_transport_timing_report")
        async def on_transport(observer, report):
            if report.bot_connected_secs is not None:
                logger.info(f"Bot connected in {report.bot_connected_secs:.3f}s")
            logger.info(f"Client connected in {report.client_connected_secs:.3f}s")

        worker = PipelineWorker(pipeline, observers=[observer])

    Args:
        processor_types: Optional tuple of processor types to measure. If None,
            all non-internal processors are measured.
    """

    def __init__(
        self,
        *,
        processor_types: tuple[type[FrameProcessor], ...] | None = None,
        **kwargs,
    ):
        """Initialize the startup timing observer.

        Args:
            processor_types: Optional tuple of processor types to measure.
                If None, all non-internal processors are measured.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._processor_types = processor_types

        # Map processor ID -> arrival info.
        self._arrivals: dict[int, _ArrivalInfo] = {}

        # Collected timings in pipeline order.
        self._timings: list[ProcessorStartupTiming] = []

        # Captured once when the first StartFrame arrives.
        self._start_frame: _StartFrameInfo | None = None

        # When the pipeline began setting up, i.e. before any processor
        # connected, and how long each processor's setup() took.
        self._setup_started_ns: int = 0
        self._setup_wall_clock: float = 0.0
        self._setup_durations: dict[int, float] = {}

        # Whether we've already emitted the startup timing report.
        self._startup_timing_reported = False

        # Whether we've already measured transport timing.
        self._transport_timing_reported = False

        # Bot connected timing (stored for inclusion in the transport report).
        self._bot_connected_secs: float | None = None

        self._register_event_handler("on_startup_timing_report")
        self._register_event_handler("on_transport_timing_report")

    async def setup(self, task_manager: BaseTaskManager):
        """Start the clock, before any processor has been set up.

        Processors connect while they are being set up, so startup begins here
        rather than at the StartFrame.

        Args:
            task_manager: The task manager to run tasks on.
        """
        await super().setup(task_manager)
        self._setup_started_ns = time.monotonic_ns()
        self._setup_wall_clock = time.time()

    async def on_processor_setup(self, data: ProcessorSetUp):
        """Record how long a processor's setup() took.

        Args:
            data: The processor setup event data.
        """
        if self._startup_timing_reported or not self._should_track(data.processor):
            return

        self._setup_durations[data.processor.id] = (data.finished_at_ns - data.started_at_ns) / 1e9

    def _should_track(self, processor: FrameProcessor) -> bool:
        """Check if a processor should be tracked for timing.

        Args:
            processor: The processor to check.

        Returns:
            True if the processor matches the filter or no filter is set.
        """
        if self._processor_types is not None:
            return isinstance(processor, self._processor_types)
        # Default: exclude internal pipeline plumbing.
        return not isinstance(processor, _INTERNAL_TYPES)

    async def on_pipeline_started(self):
        """Emit the startup timing report when the pipeline has fully started.

        Called by the ``PipelineWorker`` after the ``StartFrame`` has been
        processed by all processors, including nested ``ParallelPipeline``
        branches.
        """
        if self._timings:
            await self._emit_report()

    async def on_process_frame(self, data: FrameProcessed):
        """Record when a StartFrame arrives at a processor.

        Args:
            data: The frame processing event data.
        """
        if self._startup_timing_reported:
            return

        if not isinstance(data.frame, StartFrame):
            return

        # Lock onto the first StartFrame.
        if self._start_frame is None:
            self._start_frame = _StartFrameInfo(
                frame_id=data.frame.id,
                arrival_ns=data.timestamp,
                wall_clock=time.time(),
            )
        elif data.frame.id != self._start_frame.frame_id:
            return

        if self._should_track(data.processor):
            self._arrivals[data.processor.id] = _ArrivalInfo(
                processor=data.processor, arrival_ts_ns=data.timestamp
            )

    async def on_push_frame(self, data: FramePushed):
        """Record when a StartFrame leaves a processor and compute the delta.

        Also handles ``BotConnectedFrame`` and ``ClientConnectedFrame`` to
        measure transport timing.

        Args:
            data: The frame push event data.
        """
        if isinstance(data.frame, BotConnectedFrame):
            self._handle_bot_connected(data)
            return

        if isinstance(data.frame, ClientConnectedFrame):
            await self._handle_client_connected(data)
            return

        if self._startup_timing_reported:
            return

        if not isinstance(data.frame, StartFrame):
            return

        if self._start_frame is not None and data.frame.id != self._start_frame.frame_id:
            return

        arrival = self._arrivals.pop(data.source.id, None)
        if arrival is None or self._start_frame is None:
            return

        duration_ns = data.timestamp - arrival.arrival_ts_ns
        duration_secs = duration_ns / 1e9
        start_offset_secs = (arrival.arrival_ts_ns - self._start_frame.arrival_ns) / 1e9

        setup_duration_secs = self._setup_durations.get(arrival.processor.id, 0.0)

        self._timings.append(
            ProcessorStartupTiming(
                processor_name=arrival.processor.name,
                start_offset_secs=start_offset_secs,
                # What the processor cost overall: it connects while being set
                # up, and start() is whatever is left to do once it has.
                duration_secs=setup_duration_secs + duration_secs,
                setup_duration_secs=setup_duration_secs,
            )
        )

    def _handle_bot_connected(self, data: FramePushed):
        """Record bot connected timing on first BotConnectedFrame."""
        if self._bot_connected_secs is not None:
            return

        self._bot_connected_secs = data.timestamp / 1e9

    async def _handle_client_connected(self, data: FramePushed):
        """Emit transport timing report on first ClientConnectedFrame."""
        if self._transport_timing_reported:
            return

        self._transport_timing_reported = True
        client_connected_secs = data.timestamp / 1e9
        report = TransportTimingReport(
            # Both offsets are elapsed pipeline-clock time, which starts before
            # this observer does, so the wall clock they are offsets from comes
            # from the same clock rather than from a second reading of its own.
            start_time=time.time() - client_connected_secs,
            bot_connected_secs=self._bot_connected_secs,
            client_connected_secs=client_connected_secs,
        )
        await self._call_event_handler("on_transport_timing_report", report)

    async def _emit_report(self):
        """Build and emit the startup timing report."""
        if self._startup_timing_reported:
            return
        self._startup_timing_reported = True

        # Processors are set up concurrently, so what they cost does not add
        # up to wall-clock time. Report the span instead.
        total = (time.monotonic_ns() - self._setup_started_ns) / 1e9

        report = StartupTimingReport(
            start_time=self._setup_wall_clock,
            total_duration_secs=total,
            processor_timings=self._timings,
        )

        await self._call_event_handler("on_startup_timing_report", report)
