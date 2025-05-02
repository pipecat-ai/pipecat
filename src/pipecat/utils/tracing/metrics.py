#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Metrics collection utilities for OpenTelemetry tracing."""

import contextlib
from typing import Any, Dict, Optional

from opentelemetry import trace

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.utils.tracing.tracing import is_tracing_available


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

    def apply_to_current_span(self) -> bool:
        """Apply all collected metrics to the current span.

        Returns:
            bool: True if metrics were applied, False otherwise
        """
        if not is_tracing_available():
            return False

        current_span = trace.get_current_span()
        return self.apply_to_span(current_span)

    def apply_to_span(self, span) -> bool:
        """Apply all collected metrics to the specified span.

        Args:
            span: OpenTelemetry span

        Returns:
            bool: True if metrics were applied, False otherwise
        """
        if not is_tracing_available():
            return False

        # Apply service type
        span.set_attribute("metrics.service", self.service_name)

        # Apply TTFB if set
        if self.ttfb_ms is not None:
            span.set_attribute("metrics.ttfb_ms", self.ttfb_ms)

        # Apply TTS character count if set
        if self.character_count is not None:
            span.set_attribute("metrics.tts.character_count", self.character_count)

        # Apply LLM token usage if set
        if self.token_usage is not None:
            span.set_attribute("metrics.llm.prompt_tokens", self.token_usage.prompt_tokens)
            span.set_attribute("metrics.llm.completion_tokens", self.token_usage.completion_tokens)
            span.set_attribute("metrics.llm.total_tokens", self.token_usage.total_tokens)

            if self.token_usage.cache_read_input_tokens is not None:
                span.set_attribute(
                    "metrics.llm.cache_read_input_tokens", self.token_usage.cache_read_input_tokens
                )

            if self.token_usage.cache_creation_input_tokens is not None:
                span.set_attribute(
                    "metrics.llm.cache_creation_input_tokens",
                    self.token_usage.cache_creation_input_tokens,
                )

        # Apply any additional metrics
        for name, value in self.additional_metrics.items():
            span.set_attribute(f"metrics.{name}", value)

        return True


@contextlib.contextmanager
def traced_operation(processor: Any, operation_name: str = None):
    """Context manager for traced operations with metric collection.

    Usage:
        with traced_operation(self, "synthesis") as metrics:
            metrics.set_character_count(len(text))
            result = await do_expensive_operation()

    Args:
        processor: The processor instance
        operation_name: Optional name for the operation
    """
    if not is_tracing_available():
        yield None
        return

    # Extract service name from processor class
    service_name = processor.__class__.__name__.replace("TTSService", "").lower()

    # Create metrics collector
    metrics = TraceMetricsCollector(service_name)

    try:
        yield metrics
    finally:
        # Add operation name if provided
        if operation_name:
            metrics.add_metric("operation.name", operation_name)

        # Apply metrics to current span
        metrics.apply_to_current_span()
