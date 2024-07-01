#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time

from enum import Enum

from pipecat.frames.frames import ErrorFrame, Frame, MetricsFrame, StartFrame, StartInterruptionFrame, UserStoppedSpeakingFrame
from pipecat.utils.utils import obj_count, obj_id

from loguru import logger


class FrameDirection(Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2


class FrameProcessorMetrics:
    def __init__(self, name: str):
        self._name = name
        self._start_ttfb_time = 0
        self._start_processing_time = 0
        self._should_report_ttfb = True

    async def start_ttfb_metrics(self, report_only_initial_ttfb):
        if self._should_report_ttfb:
            self._start_ttfb_time = time.time()
            self._should_report_ttfb = not report_only_initial_ttfb

    async def stop_ttfb_metrics(self):
        if self._start_ttfb_time == 0:
            return None

        value = time.time() - self._start_ttfb_time
        logger.debug(f"{self._name} TTFB: {value}")
        ttfb = {
            "processor": self._name,
            "value": value
        }
        self._start_ttfb_time = 0
        return MetricsFrame(ttfb=[ttfb])

    async def start_processing_metrics(self):
        self._start_processing_time = time.time()

    async def stop_processing_metrics(self):
        if self._start_processing_time == 0:
            return None

        value = time.time() - self._start_processing_time
        logger.debug(f"{self._name} processing time: {value}")
        processing = {
            "processor": self._name,
            "value": value
        }
        self._start_processing_time = 0
        return MetricsFrame(processing=[processing])


class FrameProcessor:

    def __init__(
            self,
            *,
            name: str | None = None,
            loop: asyncio.AbstractEventLoop | None = None,
            **kwargs):
        self.id: int = obj_id()
        self.name = name or f"{self.__class__.__name__}#{obj_count(self)}"
        self._prev: "FrameProcessor" | None = None
        self._next: "FrameProcessor" | None = None
        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_running_loop()

        # Properties
        self._allow_interruptions = False
        self._enable_metrics = False
        self._report_only_initial_ttfb = False

        # Metrics
        self._metrics = FrameProcessorMetrics(name=self.name)

    @property
    def interruptions_allowed(self):
        return self._allow_interruptions

    @property
    def metrics_enabled(self):
        return self._enable_metrics

    @property
    def report_only_initial_ttfb(self):
        return self._report_only_initial_ttfb

    def can_generate_metrics(self) -> bool:
        return False

    async def start_ttfb_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_ttfb_metrics(self._report_only_initial_ttfb)

    async def stop_ttfb_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_ttfb_metrics()
            if frame:
                await self.push_frame(frame)

    async def start_processing_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            await self._metrics.start_processing_metrics()

    async def stop_processing_metrics(self):
        if self.can_generate_metrics() and self.metrics_enabled:
            frame = await self._metrics.stop_processing_metrics()
            if frame:
                await self.push_frame(frame)

    async def stop_all_metrics(self):
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    async def cleanup(self):
        pass

    def link(self, processor: 'FrameProcessor'):
        self._next = processor
        processor._prev = self
        logger.debug(f"Linking {self} -> {self._next}")

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, StartFrame):
            self._allow_interruptions = frame.allow_interruptions
            self._enable_metrics = frame.enable_metrics
            self._report_only_initial_ttfb = frame.report_only_initial_ttfb
        elif isinstance(frame, StartInterruptionFrame):
            await self.stop_all_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._should_report_ttfb = True

    async def push_error(self, error: ErrorFrame):
        await self.push_frame(error, FrameDirection.UPSTREAM)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        try:
            if direction == FrameDirection.DOWNSTREAM and self._next:
                logger.trace(f"Pushing {frame} from {self} to {self._next}")
                await self._next.process_frame(frame, direction)
            elif direction == FrameDirection.UPSTREAM and self._prev:
                logger.trace(f"Pushing {frame} upstream from {self} to {self._prev}")
                await self._prev.process_frame(frame, direction)
        except Exception as e:
            logger.exception(f"Uncaught exception in {self}: {e}")

    def __str__(self):
        return self.name
