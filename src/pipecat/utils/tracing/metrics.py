#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Metrics collection utilities for OpenTelemetry tracing."""

from typing import Any, Dict, Optional

from pipecat.metrics.metrics import LLMTokenUsage


class TraceMetricsCollector:
    """Collects metrics for the current trace span.

    This class provides a clean way to collect metrics and add them to the current
    trace span without cluttering service implementations.
    """

    def __init__(self, service_name: str):
        """Initialize a metrics collector.

        Args:
            service_name: Name of the service (e.g., "cartesia")
        """
        self.service_name = service_name
        self.ttfb_ms: Optional[float] = None
        self.character_count: Optional[int] = None
        self.token_usage: Optional[LLMTokenUsage] = None
        self.additional_metrics: Dict[str, Any] = {}

    def set_ttfb(self, ttfb_ms: float) -> "TraceMetricsCollector":
        """Set the Time To First Byte metric.

        Args:
            ttfb_ms: Time to first byte in milliseconds

        Returns:
            self: For method chaining
        """
        self.ttfb_ms = ttfb_ms
        return self

    def set_character_count(self, count: int) -> "TraceMetricsCollector":
        """Set the TTS character count metric.

        Args:
            count: Number of characters synthesized

        Returns:
            self: For method chaining
        """
        self.character_count = count
        return self

    def set_token_usage(self, usage: LLMTokenUsage) -> "TraceMetricsCollector":
        """Set the LLM token usage metrics.

        Args:
            usage: LLM token usage data

        Returns:
            self: For method chaining
        """
        self.token_usage = usage
        return self

    def add_metric(self, name: str, value: Any) -> "TraceMetricsCollector":
        """Add a custom metric.

        Args:
            name: Metric name
            value: Metric value

        Returns:
            self: For method chaining
        """
        self.additional_metrics[name] = value
        return self
