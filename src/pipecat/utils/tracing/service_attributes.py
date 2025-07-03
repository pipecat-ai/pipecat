#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Functions for adding attributes to OpenTelemetry spans.

This module provides specialized functions for adding service-specific
attributes to OpenTelemetry spans, following standard semantic conventions
where applicable and Pipecat-specific conventions for additional context.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

    Args:
        service_name: The service class name to extract system name from.

    Returns:
        The standardized gen_ai.system value.
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
        span: The span to add attributes to.
        service_name: Name of the TTS service (e.g., "cartesia").
        model: Model name/identifier.
        voice_id: Voice identifier.
        text: The text being synthesized.
        settings: Service configuration settings.
        character_count: Number of characters in the text.
        operation_name: Name of the operation (default: "tts").
        ttfb: Time to first byte in seconds.
        **kwargs: Additional attributes to add.
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
    user_id: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    vad_enabled: bool = False,
    ttfb: Optional[float] = None,
    **kwargs,
) -> None:
    """Add STT-specific attributes to a span.

    Args:
        span: The span to add attributes to.
        service_name: Name of the STT service (e.g., "deepgram").
        model: Model name/identifier.
        operation_name: Name of the operation (default: "stt").
        transcript: The transcribed text.
        is_final: Whether this is a final transcript.
        language: Detected or configured language.
        user_id: User ID associated with the audio being transcribed.
        settings: Service configuration settings.
        vad_enabled: Whether voice activity detection is enabled.
        ttfb: Time to first byte in seconds.
        **kwargs: Additional attributes to add.
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

    if user_id:
        span.set_attribute("user_id", user_id)

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
        span: The span to add attributes to.
        service_name: Name of the LLM service (e.g., "openai").
        model: Model name/identifier.
        stream: Whether streaming is enabled.
        messages: JSON-serialized messages.
        output: Aggregated output text from the LLM.
        tools: JSON-serialized tools configuration.
        tool_count: Number of tools available.
        tool_choice: Tool selection configuration.
        system: System message.
        parameters: Service parameters.
        extra_parameters: Additional parameters.
        ttfb: Time to first byte in seconds.
        **kwargs: Additional attributes to add.
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


def add_gemini_live_span_attributes(
    span: "Span",
    service_name: str,
    model: str,
    operation_name: str,
    voice_id: Optional[str] = None,
    language: Optional[str] = None,
    modalities: Optional[str] = None,
    settings: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict]] = None,
    tools_serialized: Optional[str] = None,
    transcript: Optional[str] = None,
    is_input: Optional[bool] = None,
    text_output: Optional[str] = None,
    audio_data_size: Optional[int] = None,
    **kwargs,
) -> None:
    """Add Gemini Live specific attributes to a span.

    Args:
        span: The span to add attributes to.
        service_name: Name of the service.
        model: Model name/identifier.
        operation_name: Name of the operation (setup, model_turn, tool_call, etc.).
        voice_id: Voice identifier used for output.
        language: Language code for the session.
        modalities: Supported modalities (e.g., "AUDIO", "TEXT").
        settings: Service configuration settings.
        tools: Available tools/functions list.
        tools_serialized: JSON-serialized tools for detailed inspection.
        transcript: Transcription text.
        is_input: Whether transcript is input (True) or output (False).
        text_output: Text output from model.
        audio_data_size: Size of audio data in bytes.
        **kwargs: Additional attributes to add.
    """
    # Add standard attributes
    span.set_attribute("gen_ai.system", "gcp.gemini")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.operation.name", operation_name)
    span.set_attribute("service.operation", operation_name)

    # Add optional attributes
    if voice_id:
        span.set_attribute("voice_id", voice_id)

    if language:
        span.set_attribute("language", language)

    if modalities:
        span.set_attribute("modalities", modalities)

    if transcript:
        span.set_attribute("transcript", transcript)
        if is_input is not None:
            span.set_attribute("transcript.is_input", is_input)

    if text_output:
        span.set_attribute("text_output", text_output)

    if audio_data_size is not None:
        span.set_attribute("audio.data_size_bytes", audio_data_size)

    if tools:
        span.set_attribute("tools.count", len(tools))
        span.set_attribute("tools.available", True)

        # Add individual tool names for easier filtering
        tool_names = []
        for tool in tools:
            if isinstance(tool, dict) and "name" in tool:
                tool_names.append(tool["name"])
            elif hasattr(tool, "name"):
                tool_name = getattr(tool, "name", None)
                if tool_name is not None:
                    tool_names.append(tool_name)

        if tool_names:
            span.set_attribute("tools.names", ",".join(tool_names))

    if tools_serialized:
        span.set_attribute("tools.definitions", tools_serialized)

    # Add settings if provided
    if settings:
        for key, value in settings.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"settings.{key}", value)
            elif key == "vad" and value:
                # Handle VAD settings specially
                if hasattr(value, "disabled") and value.disabled is not None:
                    span.set_attribute("settings.vad.disabled", value.disabled)
                if hasattr(value, "start_sensitivity") and value.start_sensitivity:
                    span.set_attribute(
                        "settings.vad.start_sensitivity", value.start_sensitivity.value
                    )
                if hasattr(value, "end_sensitivity") and value.end_sensitivity:
                    span.set_attribute("settings.vad.end_sensitivity", value.end_sensitivity.value)

    # Add any additional keyword arguments as attributes
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)


def add_openai_realtime_span_attributes(
    span: "Span",
    service_name: str,
    model: str,
    operation_name: str,
    session_properties: Optional[Dict[str, Any]] = None,
    transcript: Optional[str] = None,
    is_input: Optional[bool] = None,
    context_messages: Optional[str] = None,
    function_calls: Optional[List[Dict]] = None,
    tools: Optional[List[Dict]] = None,
    tools_serialized: Optional[str] = None,
    audio_data_size: Optional[int] = None,
    **kwargs,
) -> None:
    """Add OpenAI Realtime specific attributes to a span.

    Args:
        span: The span to add attributes to.
        service_name: Name of the service.
        model: Model name/identifier.
        operation_name: Name of the operation (setup, transcription, response, etc.).
        session_properties: Session configuration properties.
        transcript: Transcription text.
        is_input: Whether transcript is input (True) or output (False).
        context_messages: JSON-serialized context messages.
        function_calls: Function calls being made.
        tools: Available tools/functions list.
        tools_serialized: JSON-serialized tools for detailed inspection.
        audio_data_size: Size of audio data in bytes.
        **kwargs: Additional attributes to add.
    """
    # Add standard attributes
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.operation.name", operation_name)
    span.set_attribute("service.operation", operation_name)

    # Add optional attributes
    if transcript:
        span.set_attribute("transcript", transcript)
        if is_input is not None:
            span.set_attribute("transcript.is_input", is_input)

    if context_messages:
        span.set_attribute("input", context_messages)

    if audio_data_size is not None:
        span.set_attribute("audio.data_size_bytes", audio_data_size)

    if tools:
        span.set_attribute("tools.count", len(tools))
        span.set_attribute("tools.available", True)

        # Add individual tool names for easier filtering
        tool_names = []
        for tool in tools:
            if isinstance(tool, dict) and "name" in tool:
                tool_names.append(tool["name"])
            elif hasattr(tool, "name"):
                tool_names.append(tool.name)
            elif isinstance(tool, dict) and "function" in tool and "name" in tool["function"]:
                tool_names.append(tool["function"]["name"])

        if tool_names:
            span.set_attribute("tools.names", ",".join(tool_names))

    if tools_serialized:
        span.set_attribute("tools.definitions", tools_serialized)

    if function_calls:
        span.set_attribute("function_calls.count", len(function_calls))
        if function_calls:
            call = function_calls[0]
            if hasattr(call, "name"):
                span.set_attribute("function_calls.first_name", call.name)
            elif isinstance(call, dict) and "name" in call:
                span.set_attribute("function_calls.first_name", call["name"])

    # Add session properties if provided
    if session_properties:
        for key, value in session_properties.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(f"session.{key}", value)
            elif key == "turn_detection" and value is not None:
                if isinstance(value, bool):
                    span.set_attribute("session.turn_detection.enabled", value)
                elif isinstance(value, dict):
                    span.set_attribute("session.turn_detection.enabled", True)
                    for td_key, td_value in value.items():
                        if isinstance(td_value, (str, int, float, bool)):
                            span.set_attribute(f"session.turn_detection.{td_key}", td_value)

    # Add any additional keyword arguments as attributes
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)
