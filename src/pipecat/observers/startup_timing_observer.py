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

It also measures transport readiness — the time from ``StartFrame`` to the
first ``ClientConnectedFrame`` — via a separate ``on_transport_readiness_measured``
event.

Example::

    observer = StartupTimingObserver()

    @observer.event_handler("on_startup_timing_report")
    async def on_report(observer, report):
        for t in report.processor_timings:
            print(f"{t.processor_name}: {t.duration_secs:.3f}s")

    @observer.event_handler("on_transport_readiness_measured")
    async def on_readiness(observer, report):
        print(f"Transport ready in {report.readiness_secs:.3f}s")

    task = PipelineTask(pipeline, observers=[observer])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

from loguru import logger

from pipecat.frames.frames import ClientConnectedFrame, StartFrame
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import PipelineSink, PipelineSource
from pipecat.processors.frame_processor import FrameProcessor

# Internal pipeline types excluded from tracking by default.
_INTERNAL_TYPES = (PipelineSink, PipelineSource, BasePipeline)


@dataclass
class ProcessorStartupTiming:
    """Startup timing for a single processor.

    Parameters:
        processor_name: The name of the processor.
        duration_secs: How long the processor's start() took, in seconds.
    """

    processor_name: str
    duration_secs: float


@dataclass
class StartupTimingReport:
    """Report of startup timings for all measured processors.

    Parameters:
        total_duration_secs: Total wall-clock time from first to last processor start.
        processor_timings: Per-processor timing data, in pipeline order.
    """

    total_duration_secs: float
    processor_timings: List[ProcessorStartupTiming] = field(default_factory=list)


@dataclass
class TransportReadinessReport:
    """Time from pipeline start to first client connection.

    Parameters:
        readiness_secs: Seconds from StartFrame to first ClientConnectedFrame.
    """

    readiness_secs: float


class StartupTimingObserver(BaseObserver):
    """Observer that measures processor startup times during pipeline initialization.

    Tracks how long each processor's ``start()`` method takes by measuring the
    time between when a ``StartFrame`` arrives at a processor and when it is
    pushed downstream. This captures WebSocket connections, API authentication,
    model loading, and other initialization work.

    Also measures transport readiness — the time from ``StartFrame`` to the
    first ``ClientConnectedFrame`` — indicating how long it takes for a client
    to connect after the pipeline starts.

    By default, internal pipeline processors (``PipelineSource``, ``PipelineSink``,
    ``Pipeline``) are excluded from the report. Pass ``processor_types`` to
    measure only specific types.

    Event handlers available:

    - on_startup_timing_report: Called once after startup completes with the full
      timing report.
    - on_transport_readiness_measured: Called once when the first client connects with the
      transport readiness timing.

    Example::

        observer = StartupTimingObserver(
            processor_types=(STTService, TTSService)
        )

        @observer.event_handler("on_startup_timing_report")
        async def on_report(observer, report):
            for t in report.processor_timings:
                logger.info(f"{t.processor_name}: {t.duration_secs:.3f}s")

        @observer.event_handler("on_transport_readiness_measured")
        async def on_readiness(observer, report):
            logger.info(f"Transport ready in {report.readiness_secs:.3f}s")

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

        # Map processor ID -> (processor, arrival_timestamp_ns)
        self._arrivals: Dict[int, Tuple[FrameProcessor, int]] = {}

        # Collected timings in pipeline order.
        self._timings: List[ProcessorStartupTiming] = []

        # Lock onto the first StartFrame we see (by frame ID).
        self._start_frame_id: Optional[str] = None

        # Whether we've already emitted the startup timing report.
        self._startup_timing_reported = False

        # Whether we've already measured transport readiness.
        self._transport_readiness_measured = False

        # Timestamp (ns) when we first see a StartFrame arrive at a processor.
        self._start_frame_arrival_ns: Optional[int] = None

        self._register_event_handler("on_startup_timing_report")
        self._register_event_handler("on_transport_readiness_measured")

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

    async def on_process_frame(self, data: FrameProcessed):
        """Record when a StartFrame arrives at a processor.

        When a ``StartFrame`` reaches a ``PipelineSink``, startup is complete
        (the frame has traversed the entire pipeline) and the report is emitted.

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
        elif data.frame.id != self._start_frame_id:
            return

        # When the StartFrame reaches a PipelineSink, all processors have
        # completed start(). PipelineSinks use direct mode so the outermost
        # sink fires last within the same synchronous call chain.
        if isinstance(data.processor, PipelineSink):
            if self._timings:
                await self._emit_report()
            return

        if self._should_track(data.processor):
            self._arrivals[data.processor.id] = (data.processor, data.timestamp)

    async def on_push_frame(self, data: FramePushed):
        """Record when a StartFrame leaves a processor and compute the delta.

        Also handles ``ClientConnectedFrame`` to measure transport readiness.

        Args:
            data: The frame push event data.
        """
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

        processor, arrival_ts = arrival
        duration_ns = data.timestamp - arrival_ts
        duration_secs = duration_ns / 1e9

        self._timings.append(
            ProcessorStartupTiming(
                processor_name=processor.name,
                duration_secs=duration_secs,
            )
        )

    async def _handle_client_connected(self, data: FramePushed):
        """Measure transport readiness on first client connection."""
        if self._transport_readiness_measured or self._start_frame_arrival_ns is None:
            return

        self._transport_readiness_measured = True
        delta_ns = data.timestamp - self._start_frame_arrival_ns
        readiness_secs = delta_ns / 1e9
        report = TransportReadinessReport(readiness_secs=readiness_secs)
        await self._call_event_handler("on_transport_readiness_measured", report)

    async def _emit_report(self):
        """Build and emit the startup timing report."""
        if self._startup_timing_reported:
            return
        self._startup_timing_reported = True

        total = sum(t.duration_secs for t in self._timings)

        report = StartupTimingReport(
            total_duration_secs=total,
            processor_timings=self._timings,
        )

        await self._call_event_handler("on_startup_timing_report", report)
