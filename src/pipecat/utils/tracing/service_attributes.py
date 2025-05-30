#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Functions for adding attributes to OpenTelemetry spans."""

from typing import TYPE_CHECKING, Any, Dict, Optional

# Import for type checking only
if TYPE_CHECKING:
    from opentelemetry.trace import Span

from pipecat.utils.tracing.setup import is_tracing_available

if is_tracing_available():
    from opentelemetry.trace import Span


def _get_gen_ai_system_from_service_name(service_name: str) -> str:
    """Extract the standardized gen_ai.system value from a service class name.

    Source:
    https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system

    Uses standard OTel names where possible, with special case mappings for
    service names that don't follow the pattern.
    """
    SPECIAL_CASE_MAPPINGS = {
        # AWS
        "AWSBedrockLLMService": "aws.bedrock",
        # Azure
        "AzureLLMService": "az.ai.openai",
        # Google
        "GoogleLLMService": "gcp.gemini",
        "GoogleLLMOpenAIBetaService": "gcp.gemini",
        "GoogleVertexLLMService": "gcp.vertex_ai",
        # Others
        "GrokLLMService": "xai",
    }

    if service_name in SPECIAL_CASE_MAPPINGS:
        return SPECIAL_CASE_MAPPINGS[service_name]

    if service_name.endswith("LLMService"):
        provider = service_name[:-10].lower()
    else:
        provider = service_name.lower()

    return provider


def add_tts_span_attributes(
    span: "Span",
    service_name: str,
    model: str,
    voice_id: str,
    text: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    character_count: Optional[int] = None,
    operation_name: str = "tts",
    ttfb: Optional[float] = None,
    **kwargs,
) -> None:
    """Add TTS-specific attributes to a span.

    Args:
        span: The span to add attributes to
        service_name: Name of the TTS service (e.g., "cartesia")
        model: Model name/identifier
        voice_id: Voice identifier
        text: The text being synthesized
        settings: Service configuration settings
        character_count: Number of characters in the text
        operation_name: Name of the operation (default: "tts")
        ttfb: Time to first byte in seconds
        **kwargs: Additional attributes to add
    """
    # Add standard attributes
    span.set_attribute("gen_ai.system", service_name.replace("TTSService", "").lower())
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.operation.name", operation_name)
    span.set_attribute("gen_ai.output.type", "speech")
    span.set_attribute("voice_id", voice_id)

    # Add optional attributes
    if text:
        span.set_attribute("text", text)

    if character_count is not None:
        span.set_attribute("metrics.character_count", character_count)

    if ttfb is not None:
        span.set_attribute("metrics.ttfb", ttfb)

    # Add settings if provided
    if settings:
        for key, value in settings.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"settings.{key}", value)

    # Add any additional keyword arguments as attributes
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)


def add_stt_span_attributes(
    span: "Span",
    service_name: str,
    model: str,
    operation_name: str = "stt",
    transcript: Optional[str] = None,
    is_final: Optional[bool] = None,
    language: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    vad_enabled: bool = False,
    ttfb: Optional[float] = None,
    **kwargs,
) -> None:
    """Add STT-specific attributes to a span.

    Args:
        span: The span to add attributes to
        service_name: Name of the STT service (e.g., "deepgram")
        model: Model name/identifier
        operation_name: Name of the operation (default: "stt")
        transcript: The transcribed text
        is_final: Whether this is a final transcript
        language: Detected or configured language
        settings: Service configuration settings
        vad_enabled: Whether voice activity detection is enabled
        ttfb: Time to first byte in seconds
        **kwargs: Additional attributes to add
    """
    # Add standard attributes
    span.set_attribute("gen_ai.system", service_name.replace("STTService", "").lower())
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.operation.name", operation_name)
    span.set_attribute("vad_enabled", vad_enabled)

    # Add optional attributes
    if transcript:
        span.set_attribute("transcript", transcript)

    if is_final is not None:
        span.set_attribute("is_final", is_final)

    if language:
        span.set_attribute("language", language)

    if ttfb is not None:
        span.set_attribute("metrics.ttfb", ttfb)

    # Add settings if provided
    if settings:
        for key, value in settings.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"settings.{key}", value)

    # Add any additional keyword arguments as attributes
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)


def add_llm_span_attributes(
    span: "Span",
    service_name: str,
    model: str,
    stream: bool = True,
    messages: Optional[str] = None,
    output: Optional[str] = None,
    tools: Optional[str] = None,
    tool_count: Optional[int] = None,
    tool_choice: Optional[str] = None,
    system: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    extra_parameters: Optional[Dict[str, Any]] = None,
    ttfb: Optional[float] = None,
    **kwargs,
) -> None:
    """Add LLM-specific attributes to a span.

    Args:
        span: The span to add attributes to
        service_name: Name of the LLM service (e.g., "openai")
        model: Model name/identifier
        stream: Whether streaming is enabled
        messages: JSON-serialized messages
        output: Aggregated output text from the LLM
        tools: JSON-serialized tools configuration
        tool_count: Number of tools available
        tool_choice: Tool selection configuration
        system: System message
        parameters: Service parameters
        extra_parameters: Additional parameters
        ttfb: Time to first byte in seconds
        **kwargs: Additional attributes to add
    """
    # Add standard attributes
    span.set_attribute("gen_ai.system", _get_gen_ai_system_from_service_name(service_name))
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.output.type", "text")
    span.set_attribute("stream", stream)

    # Add optional attributes
    if messages:
        span.set_attribute("input", messages)

    if output:
        span.set_attribute("output", output)

    if tools:
        span.set_attribute("tools", tools)

    if tool_count is not None:
        span.set_attribute("tool_count", tool_count)

    if tool_choice:
        span.set_attribute("tool_choice", tool_choice)

    if system:
        span.set_attribute("system", system)

    if ttfb is not None:
        span.set_attribute("metrics.ttfb", ttfb)

    # Add parameters if provided
    if parameters:
        for key, value in parameters.items():
            if isinstance(value, (str, int, float, bool)):
                if key in [
                    "temperature",
                    "max_tokens",
                    "max_completion_tokens",
                    "top_p",
                    "top_k",
                    "frequency_penalty",
                    "presence_penalty",
                    "seed",
                ]:
                    span.set_attribute(f"gen_ai.request.{key}", value)
                else:
                    span.set_attribute(f"param.{key}", value)

    # Add extra parameters if provided
    if extra_parameters:
        for key, value in extra_parameters.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"extra.{key}", value)

    # Add any additional keyword arguments as attributes
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)
