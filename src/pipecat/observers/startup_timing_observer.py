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

Example::

    observer = StartupTimingObserver()

    @observer.event_handler("on_startup_timing_report")
    async def on_report(observer, report):
        for t in report.processor_timings:
            print(f"{t.processor_name}: {t.duration_secs:.3f}s")

    task = PipelineTask(pipeline, observers=[observer])
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

from loguru import logger

from pipecat.frames.frames import StartFrame
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.base_pipeline import BasePipeline
from pipecat.pipeline.pipeline import PipelineSink, PipelineSource
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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


class StartupTimingObserver(BaseObserver):
    """Observer that measures processor startup times during pipeline initialization.

    Tracks how long each processor's ``start()`` method takes by measuring the
    time between when a ``StartFrame`` arrives at a processor and when it is
    pushed downstream. This captures WebSocket connections, API authentication,
    model loading, and other initialization work.

    By default, internal pipeline processors (``PipelineSource``, ``PipelineSink``,
    ``Pipeline``) are excluded from the report. Pass ``processor_types`` to
    measure only specific types.

    Event handlers available:

    - on_startup_timing_report: Called once after startup completes with the full
      timing report.

    Example::

        observer = StartupTimingObserver(
            processor_types=(STTService, TTSService)
        )

        @observer.event_handler("on_startup_timing_report")
        async def on_report(observer, report):
            for t in report.processor_timings:
                logger.info(f"{t.processor_name}: {t.duration_secs:.3f}s")

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

        # Whether we've already emitted the report.
        self._reported = False

        self._register_event_handler("on_startup_timing_report")

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
        if self._reported:
            return

        if not isinstance(data.frame, StartFrame):
            return

        if data.direction != FrameDirection.DOWNSTREAM:
            return

        # Lock onto the first StartFrame.
        if self._start_frame_id is None:
            self._start_frame_id = data.frame.id
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

        Args:
            data: The frame push event data.
        """
        if self._reported:
            return

        if not isinstance(data.frame, StartFrame):
            return

        if data.direction != FrameDirection.DOWNSTREAM:
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

    async def _emit_report(self):
        """Build and emit the startup timing report."""
        if self._reported:
            return
        self._reported = True

        total = sum(t.duration_secs for t in self._timings)

        report = StartupTimingReport(
            total_duration_secs=total,
            processor_timings=self._timings,
        )

        logger.debug(f"Pipeline startup completed in {total:.3f}s")
        for t in self._timings:
            logger.debug(f"  {t.processor_name}: {t.duration_secs:.3f}s")

        await self._call_event_handler("on_startup_timing_report", report)
