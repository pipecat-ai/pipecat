#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Observer for tracking pipeline startup timing.

This module provides an observer that measures how long each processor's
``start()`` method takes during pipeline startup. It works by tracking
when a ``StartFrame`` arrives at a processor (``on_process_frame``) versus
when it leaves (``on_push_frame``), giving the exact ``start()`` duration
for each processor in the pipeline.

It also measures transport timing — the time from ``StartFrame`` to the
first ``BotConnectedFrame`` (SFU transports only) and ``ClientConnectedFrame``
— via a separate ``on_transport_timing_report`` event.

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

    task = PipelineTask(pipeline, observers=[observer])
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from pipecat.frames.frames import BotConnectedFrame, ClientConnectedFrame, StartFrame
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import PipelineSource
from pipecat.processors.frame_processor import FrameProcessor

# Internal pipeline types excluded from tracking by default.
_INTERNAL_TYPES = (PipelineSource, BasePipeline)


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
        duration_secs: How long the processor's start() took, in seconds.
    """

    processor_name: str
    start_offset_secs: float
    duration_secs: float


class StartupTimingReport(BaseModel):
    """Report of startup timings for all measured processors.

    Parameters:
        start_time: Unix timestamp when the first processor began starting.
        total_duration_secs: Total wall-clock time from first to last processor start.
        processor_timings: Per-processor timing data, in pipeline order.
    """

    start_time: float
    total_duration_secs: float
    processor_timings: List[ProcessorStartupTiming] = Field(default_factory=list)


class TransportTimingReport(BaseModel):
    """Time from pipeline start to transport connection milestones.

    Parameters:
        start_time: Unix timestamp of the StartFrame (pipeline start).
        bot_connected_secs: Seconds from StartFrame to first BotConnectedFrame
            (only set for SFU transports).
        client_connected_secs: Seconds from StartFrame to first ClientConnectedFrame.
    """

    start_time: float
    bot_connected_secs: Optional[float] = None
    client_connected_secs: Optional[float] = None


class StartupTimingObserver(BaseObserver):
    """Observer that measures processor startup times during pipeline initialization.

    Tracks how long each processor's ``start()`` method takes by measuring the
    time between when a ``StartFrame`` arrives at a processor and when it is
    pushed downstream. This captures WebSocket connections, API authentication,
    model loading, and other initialization work.

    Also measures transport timing, the time from ``StartFrame`` to connection
    milestones:

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

        task = PipelineTask(pipeline, observers=[observer])

    Args:
        processor_types: Optional tuple of processor types to measure. If None,
            all non-internal processors are measured.
    """

    def __init__(
        self,
        *,
        processor_types: Optional[Tuple[Type[FrameProcessor], ...]] = None,
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
        self._arrivals: Dict[int, _ArrivalInfo] = {}

        # Collected timings in pipeline order.
        self._timings: List[ProcessorStartupTiming] = []

        # Lock onto the first StartFrame we see (by frame ID).
        self._start_frame_id: Optional[str] = None

        # Whether we've already emitted the startup timing report.
        self._startup_timing_reported = False

        # Whether we've already measured transport timing.
        self._transport_timing_reported = False

        # Timestamp (ns) when we first see a StartFrame arrive at a processor.
        self._start_frame_arrival_ns: Optional[int] = None

        # Bot connected timing (stored for inclusion in the transport report).
        self._bot_connected_secs: Optional[float] = None

        # Wall clock time when the StartFrame was first seen.
        self._start_wall_clock: Optional[float] = None

        self._register_event_handler("on_startup_timing_report")
        self._register_event_handler("on_transport_timing_report")

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

        Called by the ``PipelineTask`` after the ``StartFrame`` has been
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
        if self._start_frame_id is None:
            self._start_frame_id = data.frame.id
            self._start_frame_arrival_ns = data.timestamp
            self._start_wall_clock = time.time()
        elif data.frame.id != self._start_frame_id:
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

        if self._start_frame_id is not None and data.frame.id != self._start_frame_id:
            return

        arrival = self._arrivals.pop(data.source.id, None)
        if arrival is None:
            return

        duration_ns = data.timestamp - arrival.arrival_ts_ns
        duration_secs = duration_ns / 1e9
        start_offset_secs = (arrival.arrival_ts_ns - self._start_frame_arrival_ns) / 1e9

        self._timings.append(
            ProcessorStartupTiming(
                processor_name=arrival.processor.name,
                start_offset_secs=start_offset_secs,
                duration_secs=duration_secs,
            )
        )

    def _handle_bot_connected(self, data: FramePushed):
        """Record bot connected timing on first BotConnectedFrame."""
        if self._bot_connected_secs is not None or self._start_frame_arrival_ns is None:
            return

        delta_ns = data.timestamp - self._start_frame_arrival_ns
        self._bot_connected_secs = delta_ns / 1e9

    async def _handle_client_connected(self, data: FramePushed):
        """Emit transport timing report on first ClientConnectedFrame."""
        if self._transport_timing_reported or self._start_frame_arrival_ns is None:
            return

        self._transport_timing_reported = True
        delta_ns = data.timestamp - self._start_frame_arrival_ns
        client_connected_secs = delta_ns / 1e9
        report = TransportTimingReport(
            start_time=self._start_wall_clock or 0.0,
            bot_connected_secs=self._bot_connected_secs,
            client_connected_secs=client_connected_secs,
        )
        await self._call_event_handler("on_transport_timing_report", report)

    async def _emit_report(self):
        """Build and emit the startup timing report."""
        if self._startup_timing_reported:
            return
        self._startup_timing_reported = True

        total = sum(t.duration_secs for t in self._timings)

        report = StartupTimingReport(
            start_time=self._start_wall_clock or 0.0,
            total_duration_secs=total,
            processor_timings=self._timings,
        )

        await self._call_event_handler("on_startup_timing_report", report)
