#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame processor metrics collection and reporting."""

import time
from typing import Optional

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject


class FrameProcessorMetrics(BaseObject):
    """Metrics collection and reporting for frame processors.

    Provides comprehensive metrics tracking for frame processing operations,
    including timing measurements, resource usage, and performance analytics.
    Supports TTFB tracking, processing duration metrics, and usage statistics
    for LLM and TTS operations.
    """

    def __init__(self):
        """Initialize the frame processor metrics collector.

        Sets up internal state for tracking various metrics including TTFB,
        processing times, and usage statistics.
        """
        super().__init__()
        self._task_manager = None
        self._start_ttfb_time = 0
        self._start_processing_time = 0
        self._last_ttfb_time = 0
        self._should_report_ttfb = True

    async def setup(self, task_manager: BaseTaskManager):
        """Set up the metrics collector with a task manager.

        Args:
            task_manager: The task manager for handling async operations.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Clean up metrics collection resources."""
        await super().cleanup()

    @property
    def task_manager(self) -> BaseTaskManager:
        """Get the associated task manager.

        Returns:
            The task manager instance for async operations.
        """
        return self._task_manager

    @property
    def ttfb(self) -> Optional[float]:
        """Get the current TTFB value in seconds.

        Returns:
            The TTFB value in seconds, or None if not measured.
        """
        if self._last_ttfb_time > 0:
            return self._last_ttfb_time

        # If TTFB is in progress, calculate current value
        if self._start_ttfb_time > 0:
            return time.time() - self._start_ttfb_time

        return None

    def _processor_name(self):
        """Get the processor name from core metrics data."""
        return self._core_metrics_data.processor

    def _model_name(self):
        """Get the model name from core metrics data."""
        return self._core_metrics_data.model

    def set_core_metrics_data(self, data: MetricsData):
        """Set the core metrics data for this collector.

        Args:
            data: The core metrics data containing processor and model information.
        """
        self._core_metrics_data = data

    def set_processor_name(self, name: str):
        """Set the processor name for metrics reporting.

        Args:
            name: The name of the processor to use in metrics.
        """
        self._core_metrics_data = MetricsData(processor=name)

    async def start_ttfb_metrics(self, report_only_initial_ttfb):
        """Start measuring time-to-first-byte (TTFB).

        Args:
            report_only_initial_ttfb: Whether to report only the first TTFB measurement.
        """
        if self._should_report_ttfb:
            self._start_ttfb_time = time.time()
            self._last_ttfb_time = 0
            self._should_report_ttfb = not report_only_initial_ttfb

    async def stop_ttfb_metrics(self):
        """Stop TTFB measurement and generate metrics frame.

        Returns:
            MetricsFrame containing TTFB data, or None if not measuring.
        """
        if self._start_ttfb_time == 0:
            return None

        self._last_ttfb_time = time.time() - self._start_ttfb_time
        logger.debug(f"{self._processor_name()} TTFB: {self._last_ttfb_time}")
        ttfb = TTFBMetricsData(
            processor=self._processor_name(), value=self._last_ttfb_time, model=self._model_name()
        )
        self._start_ttfb_time = 0
        return MetricsFrame(data=[ttfb])

    async def start_processing_metrics(self):
        """Start measuring processing time."""
        self._start_processing_time = time.time()

    async def stop_processing_metrics(self):
        """Stop processing time measurement and generate metrics frame.

        Returns:
            MetricsFrame containing processing duration data, or None if not measuring.
        """
        if self._start_processing_time == 0:
            return None

        value = time.time() - self._start_processing_time
        logger.debug(f"{self._processor_name()} processing time: {value}")
        processing = ProcessingMetricsData(
            processor=self._processor_name(), value=value, model=self._model_name()
        )
        self._start_processing_time = 0
        return MetricsFrame(data=[processing])

    async def start_llm_usage_metrics(self, tokens: LLMTokenUsage):
        """Record LLM token usage metrics.

        Args:
            tokens: Token usage information including prompt and completion tokens.

        Returns:
            MetricsFrame containing LLM usage data.
        """
        logstr = f"{self._processor_name()} prompt tokens: {tokens.prompt_tokens}, completion tokens: {tokens.completion_tokens}"
        if tokens.cache_read_input_tokens:
            logstr += f", cache read input tokens: {tokens.cache_read_input_tokens}"
        if tokens.reasoning_tokens:
            logstr += f", reasoning tokens: {tokens.reasoning_tokens}"
        logger.debug(logstr)
        value = LLMUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=tokens
        )
        return MetricsFrame(data=[value])

    async def start_tts_usage_metrics(self, text: str):
        """Record TTS character usage metrics.

        Args:
            text: The text being processed by TTS.

        Returns:
            MetricsFrame containing TTS usage data.
        """
        characters = TTSUsageMetricsData(
            processor=self._processor_name(), model=self._model_name(), value=len(text)
        )
        logger.debug(f"{self._processor_name()} usage characters: {characters.value}")
        return MetricsFrame(data=[characters])
