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
    """Add standard TTS attributes to a span."""
    # Add core TTS attributes
    span.set_attribute("service.type", "tts")
    span.set_attribute("tts.service", service_name)
    span.set_attribute("tts.model", model)
    span.set_attribute("tts.voice_id", voice_id)
    span.set_attribute("tts.text", text)

    # Add language if provided
    if "language" in kwargs:
        span.set_attribute("tts.language", str(kwargs["language"]))

    # Add cartesia-specific attributes
    if "cartesia_version" in kwargs:
        span.set_attribute("tts.cartesia_version", kwargs["cartesia_version"])
    if "context_id" in kwargs and kwargs["context_id"]:
        span.set_attribute("tts.context_id", kwargs["context_id"])

    # Handle settings if provided (special case for comprehensive configuration)
    if "settings" in kwargs:
        settings = kwargs["settings"]
        # Add flattened nested settings for easier querying
        add_nested_settings_as_attributes(span, "tts.setting", settings)

    # Add any additional attributes with tts. prefix
    for key, value in kwargs.items():
        if key not in ["language", "settings", "cartesia_version", "context_id"]:
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
    """Add standard STT attributes to a span."""
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

    # Add TTFB if provided
    if "ttfb_ms" in kwargs:
        span.set_attribute("metrics.ttfb_ms", kwargs["ttfb_ms"])

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
        if key not in ["ttfb_ms"]:
            span.set_attribute(f"stt.{key}", value)


def add_llm_span_attributes(
    span,
    service_name: str,
    model: str,
    token_usage: Optional[Dict[str, int]] = None,
    **kwargs,
):
    """Add standard LLM attributes to a span."""
    # Add core LLM attributes
    span.set_attribute("service.type", "llm")
    span.set_attribute("llm.service", service_name)
    span.set_attribute("llm.model", model)

    # Add stream property if provided
    if "stream" in kwargs:
        span.set_attribute("llm.stream", kwargs["stream"])

    # Add messages if provided
    if "messages" in kwargs:
        try:
            span.set_attribute("llm.messages", kwargs["messages"])
        except Exception as e:
            span.set_attribute("llm.messages_error", f"Error serializing messages: {str(e)}")

    # Add tools if provided
    if "tools" in kwargs:
        try:
            span.set_attribute("llm.tools", kwargs["tools"])
        except Exception as e:
            span.set_attribute("llm.tools_error", f"Error serializing tools: {str(e)}")

    # Add tool_count if provided
    if "tool_count" in kwargs:
        span.set_attribute("llm.tool_count", kwargs["tool_count"])

    # Add tool_choice if provided
    if "tool_choice" in kwargs:
        span.set_attribute("llm.tool_choice", kwargs["tool_choice"])

    # Add token usage if available
    if token_usage:
        if "prompt_tokens" in token_usage:
            span.set_attribute("llm.prompt_tokens", token_usage["prompt_tokens"])
        if "completion_tokens" in token_usage:
            span.set_attribute("llm.completion_tokens", token_usage["completion_tokens"])

    # Add TTFB if provided
    if "ttfb_ms" in kwargs:
        span.set_attribute("metrics.ttfb_ms", kwargs["ttfb_ms"])

    # Add parameters if provided
    if "parameters" in kwargs:
        for param_name, param_value in kwargs["parameters"].items():
            if param_value is not "NOT_GIVEN" and isinstance(param_value, (int, float, bool)):
                span.set_attribute(f"llm.param.{param_name}", param_value)
            elif param_value == "NOT_GIVEN":
                span.set_attribute(f"llm.param.{param_name}", "NOT_GIVEN")

    # Add extra parameters if provided
    if "extra_parameters" in kwargs:
        for param_name, param_value in kwargs["extra_parameters"].items():
            if param_value is not "NOT_GIVEN":
                span.set_attribute(f"llm.param.extra.{param_name}", param_value)
            else:
                span.set_attribute(f"llm.param.extra.{param_name}", "NOT_GIVEN")

    # Add any additional attributes with llm. prefix
    for key, value in kwargs.items():
        if key not in [
            "stream",
            "messages",
            "tools",
            "tool_count",
            "tool_choice",
            "parameters",
            "extra_parameters",
            "ttfb_ms",
        ]:
            span.set_attribute(f"llm.{key}", value)


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
    from pipecat.services.llm_service import LLMService
    from pipecat.services.stt_service import STTService
    from pipecat.services.tts_service import TTSService

    # Get service name
    service_name = service.__class__.__name__
    if isinstance(service, TTSService):
        service_name = service_name.replace("TTSService", "").lower()
        # Get attributes from service
        attributes = service.get_trace_attributes(**kwargs)
        add_tts_span_attributes(span, **attributes)
    elif isinstance(service, STTService):
        service_name = service_name.replace("STTService", "").lower()
        # Get attributes from service
        attributes = service.get_trace_attributes(**kwargs)
        add_stt_span_attributes(span, **attributes)
    elif isinstance(service, LLMService):
        service_name = service_name.replace("LLMService", "").lower()
        # Get attributes from service
        attributes = service.get_trace_attributes(**kwargs)
        add_llm_span_attributes(span, **attributes)
    else:
        # Fallback for unknown service types
        for key, value in kwargs.items():
            span.set_attribute(key, value)

    # Apply metrics if provided
    if metrics_collector:
        metrics_collector.apply_to_span(span)
