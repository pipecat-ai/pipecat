#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Helper functions for adding OpenTelemetry trace attributes to Pipecat services."""

import json
from typing import Any, Dict, Optional


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
        span: The span to update with attributes
        service_name: Name of the TTS service (e.g. "cartesia")
        model: Model identifier
        voice_id: Voice identifier
        text: The text being synthesized
        **kwargs: Additional attributes including metrics
            - ttfb_ms: Time to first byte in milliseconds
            - character_count: Number of characters synthesized
            - language: Language for synthesis
            - settings: Dict of TTS settings
            - context_id: Optional context ID for services like Cartesia
            - cartesia_version: Optional version info
            - operation_name: Optional operation identifier
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

    # Add metrics attributes
    if "ttfb_ms" in kwargs and kwargs["ttfb_ms"] is not None:
        span.set_attribute("metrics.ttfb_ms", kwargs["ttfb_ms"])

    # Add character count metric
    if "character_count" in kwargs:
        span.set_attribute("metrics.tts.character_count", kwargs["character_count"])

    # Add operation name if provided
    if "operation_name" in kwargs:
        span.set_attribute("metrics.operation.name", kwargs["operation_name"])

    # Add any custom metrics
    for key, value in kwargs.items():
        if key.startswith("metric."):
            # Allow passing custom metrics with metric.* prefix
            metric_name = key[7:]  # Remove "metric." prefix
            span.set_attribute(f"metrics.{metric_name}", value)

    # Add any additional attributes with tts. prefix
    for key, value in kwargs.items():
        if key not in [
            "language",
            "settings",
            "cartesia_version",
            "context_id",
            "ttfb_ms",
            "character_count",
            "operation_name",
        ] and not key.startswith("metric."):
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
        span: The span to update with attributes
        service_name: Name of the STT service (e.g. "deepgram")
        model: Model identifier
        transcript: Optional transcript text
        is_final: Whether this is a final transcript
        language: Optional language detected/used
        vad_enabled: Whether VAD was enabled
        settings: Dict of STT settings
        **kwargs: Additional attributes including metrics
            - ttfb_ms: Time to first byte in milliseconds
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
    """Add standard LLM attributes to a span.

    Args:
        span: The span to update with attributes
        service_name: Name of the LLM service (e.g. "openai")
        model: Model identifier (e.g. "gpt-4")
        token_usage: Optional dict with token usage metrics
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
        **kwargs: Additional attributes including:
            - stream: Whether streaming was used
            - messages: The conversation messages
            - tools: Tools configuration
            - tool_count: Number of tools
            - tool_choice: Tool choice configuration
            - ttfb_ms: Time to first byte in milliseconds
            - parameters: Model parameters
            - extra_parameters: Additional parameters
    """
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
