#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Core OpenTelemetry tracing utilities and setup for Pipecat.

This module provides functions to check availability and configure OpenTelemetry
tracing for Pipecat applications. It handles the optional nature of OpenTelemetry
dependencies and provides a safe setup process.
"""

import os

# Check if OpenTelemetry is available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


def is_tracing_available() -> bool:
    """Check if OpenTelemetry tracing is available and configured.

    Returns:
        True if tracing is available, False otherwise.
    """
    return OPENTELEMETRY_AVAILABLE


def setup_tracing(
    service_name: str = "pipecat",
    exporter=None,  # User-provided exporter
    console_export: bool = False,
) -> bool:
    """Set up OpenTelemetry tracing with a user-provided exporter.

    Args:
        service_name: The name of the service for traces.
        exporter: A pre-configured OpenTelemetry span exporter instance.
                  If None, only console export will be available if enabled.
        console_export: Whether to also export traces to console (useful for debugging).

    Returns:
        True if setup was successful, False otherwise.

    Example::

        # With OTLP exporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
        setup_tracing("my-service", exporter=exporter)
    """
    if not OPENTELEMETRY_AVAILABLE:
        return False

    try:
        # Create a resource with service info
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.instance.id": os.getenv("HOSTNAME", "unknown"),
                "deployment.environment": os.getenv("ENVIRONMENT", "development"),
            }
        )

        # Set up the tracer provider with the resource
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Add console exporter if requested (good for debugging)
        if console_export:
            console_exporter = ConsoleSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # Add user-provided exporter if available
        if exporter:
            tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        return True
    except Exception as e:
        print(f"Error setting up tracing: {e}")
        return False
