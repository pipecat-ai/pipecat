#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Metrics logging observer for Pipecat.

This module provides an observer that logs metrics frames to the console,
allowing developers to monitor performance metrics, token usage, and other
statistics in real-time.
"""

from typing import Optional, Set, Type

from loguru import logger

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    SmartTurnMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


class MetricsLogObserver(BaseObserver):
    """Observer to log metrics activity to the console.

    Monitors and logs all MetricsFrame instances, including:

    - TTFBMetricsData (Time To First Byte)
    - ProcessingMetricsData (General processing time)
    - LLMUsageMetricsData (Token usage statistics)
    - TTSUsageMetricsData (Text-to-Speech character counts)
    - SmartTurnMetricsData (Turn prediction metrics)

    This allows developers to track performance metrics, token usage,
    and other statistics throughout the pipeline.

    Examples:
        Log all metrics types::

            observers = [MetricsLogObserver()]

        Log only LLM and TTS metrics::

            from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
            observers = [
                MetricsLogObserver(
                    include_metrics={LLMUsageMetricsData, TTSUsageMetricsData}
                )
            ]
    """

    def __init__(
        self,
        include_metrics: Optional[Set[Type[MetricsData]]] = None,
        **kwargs,
    ):
        """Initialize the metrics log observer.

        Args:
            include_metrics: Set of metrics types to include. If specified, only these
                metrics types will be logged. If None, all metrics are logged.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._include_metrics = include_metrics
        self._frames_seen = set()

    async def on_push_frame(self, data: FramePushed):
        """Handle frame push events and log metrics frames.

        Logs MetricsFrame instances with detailed information about the
        metrics data, formatted appropriately for each metrics type.

        Args:
            data: Frame push event data containing source, frame, and timestamp.
        """
        frame = data.frame
        timestamp = data.timestamp

        if not isinstance(frame, MetricsFrame):
            return

        # Skip frames we've already seen to avoid duplicate logging
        if frame.id in self._frames_seen:
            return

        self._frames_seen.add(frame.id)

        time_sec = timestamp / 1_000_000_000

        # Process each metrics data item in the frame
        for metrics_data in frame.data:
            # Check if this metrics type should be logged
            if not self._should_log_metrics(metrics_data):
                continue

            self._log_metrics_data(metrics_data, time_sec)

    def _should_log_metrics(self, metrics_data: MetricsData) -> bool:
        """Determine if a metrics data item should be logged based on filters.

        Args:
            metrics_data: The metrics data to check.

        Returns:
            True if the metrics should be logged, False otherwise.
        """
        # If include_metrics is specified, only log those types
        if self._include_metrics is not None:
            return type(metrics_data) in self._include_metrics

        # Otherwise, log all metrics
        return True

    def _log_metrics_data(self, metrics_data: MetricsData, time_sec: float):
        """Log a single metrics data item.

        Args:
            metrics_data: The metrics data to log.
            time_sec: Timestamp in seconds.
        """
        processor_info = f"[{metrics_data.processor}]"
        model_info = f" ({metrics_data.model})" if metrics_data.model else ""

        if isinstance(metrics_data, TTFBMetricsData):
            logger.debug(
                f"📊 {processor_info} TTFB{model_info}: {metrics_data.value}s at {time_sec:.3f}s"
            )
        elif isinstance(metrics_data, ProcessingMetricsData):
            logger.debug(
                f"📊 {processor_info} PROCESSING TIME{model_info}: {metrics_data.value}s at {time_sec:.3f}s"
            )
        elif isinstance(metrics_data, LLMUsageMetricsData):
            self._log_llm_usage(metrics_data, processor_info, model_info, time_sec)
        elif isinstance(metrics_data, TTSUsageMetricsData):
            logger.debug(
                f"📊 {processor_info} TTS USAGE{model_info}: {metrics_data.value} characters at {time_sec:.3f}s"
            )
        elif isinstance(metrics_data, SmartTurnMetricsData):
            self._log_smart_turn(metrics_data, processor_info, model_info, time_sec)
        else:
            # Generic fallback for unknown metrics types
            logger.debug(
                f"📊 {processor_info} METRICS{model_info}: {metrics_data} at {time_sec:.3f}s"
            )

    def _log_llm_usage(
        self,
        metrics_data: LLMUsageMetricsData,
        processor_info: str,
        model_info: str,
        time_sec: float,
    ):
        """Log LLM token usage metrics.

        Args:
            metrics_data: The LLM usage metrics data.
            processor_info: Formatted processor name string.
            model_info: Formatted model name string.
            time_sec: Timestamp in seconds.
        """
        usage: LLMTokenUsage = metrics_data.value

        # Build usage details
        details = [
            f"prompt: {usage.prompt_tokens}",
            f"completion: {usage.completion_tokens}",
            f"total: {usage.total_tokens}",
        ]

        if usage.cache_read_input_tokens is not None:
            details.append(f"cache_read: {usage.cache_read_input_tokens}")

        if usage.cache_creation_input_tokens is not None:
            details.append(f"cache_creation: {usage.cache_creation_input_tokens}")

        if usage.reasoning_tokens is not None:
            details.append(f"reasoning: {usage.reasoning_tokens}")

        usage_str = ", ".join(details)

        logger.debug(
            f"📊 {processor_info} LLM TOKEN USAGE{model_info}: {usage_str} at {time_sec:.2f}s"
        )

    def _log_smart_turn(
        self,
        metrics_data: SmartTurnMetricsData,
        processor_info: str,
        model_info: str,
        time_sec: float,
    ):
        """Log smart turn prediction metrics.

        Args:
            metrics_data: The smart turn metrics data.
            processor_info: Formatted processor name string.
            model_info: Formatted model name string.
            time_sec: Timestamp in seconds.
        """
        complete_str = "COMPLETE" if metrics_data.is_complete else "INCOMPLETE"

        logger.debug(
            f"📊 {processor_info} SMART TURN{model_info}: {complete_str} "
            f"(probability: {metrics_data.probability:.2%}, "
            f"inference: {metrics_data.inference_time_ms:.1f}ms, "
            f"server: {metrics_data.server_total_time_ms:.1f}ms, "
            f"e2e: {metrics_data.e2e_processing_time_ms:.1f}ms) "
            f"at {time_sec:.2f}s"
        )
