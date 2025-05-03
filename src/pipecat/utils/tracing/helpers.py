#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helper functions for adding OpenTelemetry trace attributes to Pipecat services."""

import json
from typing import Any, Dict, Optional

from pipecat.utils.tracing.metrics import TraceMetricsCollector
from pipecat.utils.tracing.tracing import is_tracing_available


def add_nested_settings_as_attributes(span, prefix, settings):
    """Add nested settings as flattened span attributes."""
    for key, value in settings.items():
        if isinstance(value, dict):
            # Recursively add nested dictionaries
            add_nested_settings_as_attributes(span, f"{prefix}.{key}", value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            # Add simple types directly
            span.set_attribute(f"{prefix}.{key}", value)
        else:
            # For other types, convert to string
            try:
                span.set_attribute(f"{prefix}.{key}", json.dumps(value))
            except (TypeError, ValueError):
                span.set_attribute(f"{prefix}.{key}", str(value))


def add_tts_span_attributes(
    span, service_name: str, model: str, voice_id: str, text: str, **kwargs
):
    """Add standard TTS attributes to a span.

    Args:
        span: The OpenTelemetry span
        service_name: Name of the TTS service (e.g., "cartesia")
        model: The TTS model name
        voice_id: The voice identifier
        text: The text being synthesized
        **kwargs: Additional TTS-specific attributes
    """
    # Add core TTS attributes
    span.set_attribute("service.type", "tts")
    span.set_attribute("tts.service", service_name)
    span.set_attribute("tts.model", model)
    span.set_attribute("tts.voice_id", voice_id)
    span.set_attribute("tts.text", text)

    # Add language if provided
    if "language" in kwargs:
        span.set_attribute("tts.language", str(kwargs["language"]))

    # Handle settings if provided (special case for comprehensive configuration)
    if "settings" in kwargs:
        settings = kwargs["settings"]

        # Add as a structured attribute if possible, otherwise stringify
        try:
            span.set_attribute("tts.settings", settings)
        except (TypeError, ValueError):
            # Fall back to JSON string
            try:
                span.set_attribute("tts.settings", json.dumps(settings))
            except (TypeError, ValueError):
                # Last resort
                span.set_attribute("tts.settings", str(settings))

        # Add flattened nested settings for easier querying
        add_nested_settings_as_attributes(span, "tts.setting", settings)

    # Add any additional attributes with tts. prefix
    for key, value in kwargs.items():
        if key not in ["language", "settings"]:  # Skip already handled values
            span.set_attribute(f"tts.{key}", value)


def add_stt_span_attributes(
    span,
    service_name: str,
    model: str,
    transcript: Optional[str] = None,
    is_final: Optional[bool] = None,
    language: Optional[str] = None,
    vad_enabled: Optional[bool] = None,
    settings: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Add standard STT attributes to a span.

    Args:
        span: The OpenTelemetry span
        service_name: Name of the STT service (e.g., "deepgram", "gladia")
        model: The STT model name
        transcript: The transcribed text
        is_final: Whether the transcription is final (not interim)
        language: The detected language
        vad_enabled: Whether voice activity detection is enabled
        settings: STT service settings/configuration
        **kwargs: Additional STT-specific attributes
    """
    # Add core STT attributes
    span.set_attribute("service.type", "stt")
    span.set_attribute("stt.service", service_name)
    span.set_attribute("stt.model", model)

    # Add transcription details
    if transcript is not None:
        span.set_attribute("stt.transcript", transcript)

    if is_final is not None:
        span.set_attribute("stt.is_final", is_final)

    # Add language if provided
    if language is not None:
        span.set_attribute("stt.language", str(language))

    # Add VAD status if provided
    if vad_enabled is not None:
        span.set_attribute("stt.vad_enabled", vad_enabled)

    # Add settings if provided
    if settings:
        for key, value in settings.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"stt.setting.{key}", value)
            else:
                # For complex types, try JSON serialization
                try:
                    span.set_attribute(f"stt.setting.{key}", json.dumps(value))
                except (TypeError, ValueError):
                    # Fall back to string representation
                    span.set_attribute(f"stt.setting.{key}", str(value))

    # Add any additional attributes with stt. prefix
    for key, value in kwargs.items():
        span.set_attribute(f"stt.{key}", value)


def add_service_span_attributes(
    service, span, metrics_collector: Optional[TraceMetricsCollector] = None, **kwargs
):
    """Add span attributes based on service type.

    This detects the service type and calls the appropriate attribute helper.

    Args:
        service: The service instance
        span: The OpenTelemetry span
        metrics_collector: Optional metrics collector with metrics data
        **kwargs: Service-specific attributes and values
    """
    if not is_tracing_available():
        return

    # Import inside function to avoid circular imports
    from pipecat.services.stt_service import STTService
    from pipecat.services.tts_service import TTSService

    if isinstance(service, TTSService) and hasattr(service, "get_trace_attributes"):
        # Get attributes from the service
        attributes = service.get_trace_attributes(**kwargs)

        # Apply to the span
        add_tts_span_attributes(span, **attributes)

        # Apply metrics if provided
        if metrics_collector:
            metrics_collector.apply_to_span(span)
    elif isinstance(service, STTService) and hasattr(service, "get_trace_attributes"):
        # Get attributes from the service
        attributes = service.get_trace_attributes(**kwargs)

        # Apply to the span
        add_stt_span_attributes(span, **attributes)

        # Apply metrics if provided
        if metrics_collector:
            metrics_collector.apply_to_span(span)
    else:
        # Fallback for unknown service types
        for key, value in kwargs.items():
            span.set_attribute(key, value)

        # Still apply metrics if provided
        if metrics_collector:
            metrics_collector.apply_to_span(span)
