#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service-specific OpenTelemetry tracing decorators for Pipecat.

This module provides specialized decorators that automatically capture
rich information about service execution including configuration,
parameters, and performance metrics.
"""

import contextlib
import copy
import functools
import inspect
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar

from loguru import logger

from pipecat.processors.frame_processor import FrameDirection

# Type imports for type checking only
if TYPE_CHECKING:
    from opentelemetry import context as context_api
    from opentelemetry import trace

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    NOT_GIVEN,
    LLMContext,
    LLMSpecificMessage,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.utils.tracing.service_attributes import (
    add_gemini_live_span_attributes,
    add_llm_span_attributes,
    add_openai_realtime_span_attributes,
    add_stt_span_attributes,
    add_tts_span_attributes,
)
from pipecat.utils.tracing.setup import is_tracing_available

if is_tracing_available():
    from opentelemetry import context as context_api
    from opentelemetry import trace

T = TypeVar("T")
R = TypeVar("R")


def _get_model_name(service) -> str:
    """Get the model name from a service instance.

    This is a bit of a mess — there were multiple places a model name could live.
    Soon, self._settings should be the only source of truth about model name.
    In fact...it might already be the case, but juuuuust to be safe, we'll
    check all the places we used to store it.
    """
    return (
        getattr(getattr(service, "_settings", None), "model", None)
        or getattr(service, "_full_model_name", None)
        or getattr(service, "model_name", None)
        or getattr(service, "_model_name", None)
        or "unknown"
    )


def _noop_decorator(func):
    """No-op fallback decorator when tracing is unavailable.

    Args:
        func: The function to pass through unchanged.

    Returns:
        The original function unchanged.
    """
    return func


def _sanitize_messages_for_logging(messages: List[Any]) -> List[Dict[str, Any]]:
    """Sanitize messages for logging by redacting sensitive data.

    Creates deep copies and redacts base64-encoded images, audio data,
    and other sensitive content to prevent logging large binary data.

    Args:
        messages: List of messages in standard ChatML format.

    Returns:
        List of sanitized messages safe for logging.
    """
    sanitized = []
    for message in messages:
        # Include LLM-specific messages with a marker so traces show something was there
        if isinstance(message, LLMSpecificMessage):
            sanitized.append(
                {
                    "role": "assistant",
                    "content": f"[LLM-specific message for {message.llm}]",
                    "_llm_specific": True,
                }
            )
            continue

        msg = copy.deepcopy(message)

        # Handle content field (can be string or list of content parts)
        if "content" in msg and isinstance(msg["content"], list):
            for item in msg["content"]:
                if not isinstance(item, dict):
                    continue
                # Redact base64 image data
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/"):
                        item["image_url"]["url"] = "data:image/..."
                # Redact audio data
                if item.get("type") == "input_audio":
                    if "input_audio" in item and "data" in item["input_audio"]:
                        item["input_audio"]["data"] = "..."

        # Handle legacy mime_type format
        if "mime_type" in msg and msg.get("mime_type", "").startswith("image/"):
            msg["data"] = "..."

        sanitized.append(msg)

    return sanitized


def _get_standard_tools_for_logging(tools: Any) -> Optional[List[Dict[str, Any]]]:
    """Convert tools to standard format for logging.

    Converts ToolsSchema to a list of standard tool definitions
    in OpenAI's function calling format.

    Args:
        tools: ToolsSchema instance or NOT_GIVEN.

    Returns:
        List of tools in standard format, or None if no tools.
    """
    if tools is NOT_GIVEN or tools is None:
        return None

    if isinstance(tools, ToolsSchema):
        standard_tools = []
        for func in tools.standard_tools:
            standard_tools.append({"type": "function", "function": func.to_default_dict()})
        return standard_tools if standard_tools else None

    # Fallback: if it's already a list, return as-is
    if isinstance(tools, list):
        return tools

    return None


def _get_turn_context(self):
    """Get the current turn's tracing context if available.

    Args:
        self: The service instance.

    Returns:
        The turn context, or None if unavailable.
    """
    tracing_ctx = getattr(self, "_tracing_context", None)
    return tracing_ctx.get_turn_context() if tracing_ctx else None


def _get_parent_service_context(self):
    """Get the parent service span context (internal use only).

    This looks for the service span that was created when the service was initialized,
    or falls back to the conversation context if available.

    Args:
        self: The service instance.

    Returns:
        The parent service context, or None if unavailable.
    """
    if not is_tracing_available():
        return None

    # TODO: Remove this block and delete class_decorators.py once Traceable is removed.
    # Legacy: support for classes inheriting from Traceable (currently unused, deprecated).
    if hasattr(self, "_span") and self._span:
        return trace.set_span_in_context(self._span)

    # Use the conversation context set by TurnTraceObserver via TracingContext.
    tracing_ctx = getattr(self, "_tracing_context", None)
    conversation_context = tracing_ctx.get_conversation_context() if tracing_ctx else None
    if conversation_context:
        return conversation_context

    # Last resort: use current context (may create orphan spans)
    return context_api.get_current()


def _add_token_usage_to_span(span, token_usage):
    """Add token usage metrics to a span (internal use only).

    Args:
        span: The span to add token metrics to.
        token_usage: Dictionary or object containing token usage information.
    """
    if not is_tracing_available() or not token_usage:
        return

    if isinstance(token_usage, dict):
        if "prompt_tokens" in token_usage:
            span.set_attribute("gen_ai.usage.input_tokens", token_usage["prompt_tokens"])
        if "completion_tokens" in token_usage:
            span.set_attribute("gen_ai.usage.output_tokens", token_usage["completion_tokens"])
        # Add cached token metrics for dictionary
        if (
            "cache_read_input_tokens" in token_usage
            and token_usage["cache_read_input_tokens"] is not None
        ):
            span.set_attribute(
                "gen_ai.usage.cache_read_input_tokens", token_usage["cache_read_input_tokens"]
            )
        if (
            "cache_creation_input_tokens" in token_usage
            and token_usage["cache_creation_input_tokens"] is not None
        ):
            span.set_attribute(
                "gen_ai.usage.cache_creation_input_tokens",
                token_usage["cache_creation_input_tokens"],
            )
        if "reasoning_tokens" in token_usage and token_usage["reasoning_tokens"] is not None:
            span.set_attribute("gen_ai.usage.reasoning_tokens", token_usage["reasoning_tokens"])
    else:
        # Handle LLMTokenUsage object
        span.set_attribute("gen_ai.usage.input_tokens", getattr(token_usage, "prompt_tokens", 0))
        span.set_attribute(
            "gen_ai.usage.output_tokens", getattr(token_usage, "completion_tokens", 0)
        )

        # Add cached token metrics for LLMTokenUsage object
        cache_read_tokens = getattr(token_usage, "cache_read_input_tokens", None)
        if cache_read_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_read_input_tokens", cache_read_tokens)

        cache_creation_tokens = getattr(token_usage, "cache_creation_input_tokens", None)
        if cache_creation_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_creation_input_tokens", cache_creation_tokens)

        reasoning_tokens = getattr(token_usage, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning_tokens)


def traced_tts(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Trace TTS service methods with TTS-specific attributes.

    Automatically captures and records:

    - Service name and model information
    - Voice ID and settings
    - Character count and text content
    - Performance metrics like TTFB

    Works with both async functions and generators.

    Args:
        func: The TTS method to trace.
        name: Custom span name. Defaults to service type and class name.

    Returns:
        Wrapped method with TTS-specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        is_async_generator = inspect.isasyncgenfunction(f)

        @contextlib.asynccontextmanager
        async def tracing_context(self, text):
            """Async context manager for TTS tracing.

            Args:
                self: The TTS service instance.
                text: The text being synthesized.

            Yields:
                The active span for the TTS operation.
            """
            # Check if tracing is enabled for this service instance
            if not getattr(self, "_tracing_enabled", False):
                yield None
                return

            service_class_name = self.__class__.__name__
            span_name = "tts"

            # Get parent context
            parent_context = _get_turn_context(self) or _get_parent_service_context(self)

            # Create span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(span_name, context=parent_context) as span:
                try:
                    # Enable sharing public traces
                    span.set_attribute("langfuse.trace.public", True)

                    settings = getattr(self, "_settings", None)
                    add_tts_span_attributes(
                        span=span,
                        service_name=service_class_name,
                        model=_get_model_name(self),
                        voice_id=getattr(settings, "voice", "unknown"),
                        text=text,
                        settings=settings,
                        character_count=len(text),
                        operation_name="tts",
                        cartesia_version=getattr(self, "_cartesia_version", None),
                        context_id=getattr(self, "_context_id", None),
                    )

                    yield span

                except Exception as e:
                    logger.warning(f"Error in TTS tracing: {e}")
                    raise
                finally:
                    # Update TTFB metric at the end
                    ttfb: Optional[float] = getattr(getattr(self, "_metrics", None), "ttfb", None)
                    if ttfb is not None:
                        span.set_attribute("metrics.ttfb", ttfb)

        if is_async_generator:

            @functools.wraps(f)
            async def gen_wrapper(self, text, *args, **kwargs):
                if not getattr(self, "_tracing_enabled", False):
                    async for item in f(self, text, *args, **kwargs):
                        yield item
                    return

                fn_called = False
                try:
                    async with tracing_context(self, text):
                        fn_called = True
                        async for item in f(self, text, *args, **kwargs):
                            yield item
                except Exception as e:
                    if fn_called:
                        raise
                    logger.error(f"Error in TTS tracing (continuing without tracing): {e}")
                    async for item in f(self, text, *args, **kwargs):
                        yield item

            return gen_wrapper
        else:

            @functools.wraps(f)
            async def wrapper(self, text, *args, **kwargs):
                if not getattr(self, "_tracing_enabled", False):
                    return await f(self, text, *args, **kwargs)

                fn_called = False
                try:
                    async with tracing_context(self, text):
                        fn_called = True
                        return await f(self, text, *args, **kwargs)
                except Exception as e:
                    if fn_called:
                        raise
                    logger.error(f"Error in TTS tracing (continuing without tracing): {e}")
                    return await f(self, text, *args, **kwargs)

            return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_stt(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Trace STT service methods with transcription attributes.

    Automatically captures and records:

    - Service name and model information
    - Transcription text and final status
    - Language information
    - Performance metrics like TTFB

    Args:
        func: The STT method to trace.
        name: Custom span name. Defaults to function name.

    Returns:
        Wrapped method with STT-specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, transcript, is_final, language=None):
            if not getattr(self, "_tracing_enabled", False):
                return await f(self, transcript, is_final, language)

            fn_called = False
            try:
                service_class_name = self.__class__.__name__
                span_name = "stt"

                # Get the turn context first, then fall back to service context
                parent_context = _get_turn_context(self) or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Enable sharing public traces
                        current_span.set_attribute("langfuse.trace.public", True)

                        # Get TTFB metric if available
                        ttfb: Optional[float] = getattr(
                            getattr(self, "_metrics", None), "ttfb", None
                        )

                        # Use settings from the service if available
                        settings = getattr(self, "_settings", None)

                        add_stt_span_attributes(
                            span=current_span,
                            service_name=service_class_name,
                            model=_get_model_name(self),
                            transcript=transcript,
                            is_final=is_final,
                            language=str(language) if language else None,
                            user_id=getattr(self, "_user_id", None),
                            vad_enabled=getattr(self, "vad_enabled", False),
                            settings=settings,
                            ttfb=ttfb,
                        )

                        # Call the original function
                        fn_called = True
                        return await f(self, transcript, is_final, language)
                    except Exception as e:
                        # Log any exception but don't disrupt the main flow
                        logger.warning(f"Error in STT transcription tracing: {e}")
                        raise
            except Exception as e:
                if fn_called:
                    raise
                logger.error(f"Error in STT tracing (continuing without tracing): {e}")
                return await f(self, transcript, is_final, language)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_llm(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Trace LLM service methods with LLM-specific attributes.

    Automatically captures and records:

    - Service name and model information
    - Context content and messages
    - Tool configurations
    - Token usage metrics
    - Performance metrics like TTFB
    - Aggregated output text

    Args:
        func: The LLM method to trace.
        name: Custom span name. Defaults to service type and class name.

    Returns:
        Wrapped method with LLM-specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, context, *args, **kwargs):
            if not getattr(self, "_tracing_enabled", False):
                return await f(self, context, *args, **kwargs)

            fn_called = False
            try:
                service_class_name = self.__class__.__name__

                # Build the span name. If a custom name was supplied to the decorator we
                # honour that. Otherwise, if the provided context exposes a node name we
                # append it to the default "llm" prefix so that the span becomes
                # "llm-{node_name}".
                span_name = "llm"
                if name is not None:
                    span_name += f"-{name}"
                else:
                    otel_span_name = None
                    try:
                        otel_span_name = context.get_otel_span_name()
                    except AttributeError:
                        otel_span_name = None

                    if otel_span_name:
                        # Replace whitespace with hyphens for cleaner span names.
                        span_name = str(otel_span_name).replace(" ", "-").lower()[:20]

                # Get the parent context - turn context if available, otherwise service context
                parent_context = _get_turn_context(self) or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Enable sharing public traces
                        current_span.set_attribute("langfuse.trace.public", True)

                        # Store original method and output aggregator
                        original_push_frame = self.push_frame
                        # Accumulator for plain text tokens streamed by the LLM
                        output_text = ""  # Simple string accumulation

                        # Accumulator for function call information emitted during the
                        # generation (captured from FunctionCallsStartedFrame frames)
                        function_calls_info = []

                        async def traced_push_frame(frame, direction=None):
                            nonlocal output_text, function_calls_info
                            # ------------------------------------------------------------------
                            # Capture text tokens streamed by the LLM
                            # ------------------------------------------------------------------
                            if (
                                hasattr(frame, "__class__")
                                and frame.__class__.__name__ == "LLMTextFrame"
                                and hasattr(frame, "text")
                            ):
                                output_text += frame.text

                            # ------------------------------------------------------------------
                            # Capture function call frames so that we can record the
                            # function name and its arguments in the tracing span.
                            # ------------------------------------------------------------------
                            if (
                                hasattr(frame, "__class__")
                                and frame.__class__.__name__ == "FunctionCallsFromLLMInfoFrame"
                                and direction == FrameDirection.DOWNSTREAM
                            ):
                                try:
                                    # frame.function_calls is a sequence of FunctionCallFromLLM
                                    for call in getattr(frame, "function_calls", []):
                                        function_calls_info.append(
                                            {
                                                "function_name": getattr(
                                                    call, "function_name", None
                                                ),
                                                "tool_call_id": getattr(call, "tool_call_id", None),
                                                "arguments": getattr(call, "arguments", None),
                                            }
                                        )
                                except Exception as e:
                                    logger.warning(f"Error serializing function call: {e}")

                            # Call original
                            if direction is not None:
                                return await original_push_frame(frame, direction)
                            else:
                                return await original_push_frame(frame)

                        # For token usage monitoring
                        original_start_llm_usage_metrics = None
                        if hasattr(self, "start_llm_usage_metrics"):
                            original_start_llm_usage_metrics = self.start_llm_usage_metrics

                            # Override the method to capture token usage
                            @functools.wraps(original_start_llm_usage_metrics)
                            async def wrapped_start_llm_usage_metrics(tokens):
                                # Call the original method
                                await original_start_llm_usage_metrics(tokens)

                                # Add token usage to the current span
                                _add_token_usage_to_span(current_span, tokens)

                            # Replace the method temporarily
                            self.start_llm_usage_metrics = wrapped_start_llm_usage_metrics

                        try:
                            # Replace push_frame to capture output
                            self.push_frame = traced_push_frame

                            # Get messages for logging in standard ChatML format
                            # For OpenAILLMContext: use context's own get_messages_for_logging() method
                            # For LLMContext: use context.messages directly (already in standard format)
                            #                 and sanitize for logging
                            messages = None

                            if isinstance(context, OpenAILLMContext):
                                # OpenAILLMContext and subclasses have their own method
                                messages = context.get_messages_for_logging()
                            elif isinstance(context, LLMContext):
                                # Universal LLMContext - use standard messages directly
                                # context.messages returns messages in standard ChatML format
                                messages = _sanitize_messages_for_logging(context.messages)

                            # Get tools in standard format for logging
                            # For OpenAILLMContext: tools property returns provider-specific format
                            # For LLMContext: use standard tools from ToolsSchema
                            tools = None

                            if isinstance(context, OpenAILLMContext):
                                # OpenAILLMContext: tools property handles adapter conversion internally
                                tools = context.tools
                            elif isinstance(context, LLMContext):
                                # Universal LLMContext - use standard tools format
                                tools = _get_standard_tools_for_logging(context.tools)

                            # Use given_fields() defensively in case a service doesn't
                            # initialize all settings.
                            params = {}
                            if hasattr(self, "_settings"):
                                for key, value in self._settings.given_fields().items():
                                    if isinstance(value, (int, float, bool, str)):
                                        params[key] = value
                                    elif value is None:
                                        params[key] = "NOT_GIVEN"

                            # Add all available attributes to the span
                            attribute_kwargs = {
                                "service_name": service_class_name,
                                "model": getattr(self, "model_name", "")
                                or getattr(self, "_full_model_name", "unknown"),
                                "stream": True,  # Most LLM services use streaming
                                "parameters": params,
                            }

                            # Add optional attributes only if they exist
                            attribute_kwargs["messages"] = messages
                            attribute_kwargs["tools"] = tools

                            # Add all gathered attributes to the span
                            add_llm_span_attributes(span=current_span, **attribute_kwargs)

                        except Exception as e:
                            logger.warning(f"Error setting up LLM tracing: {e}")
                            # Don't raise - let the function execute anyway

                        # Run function with modified push_frame to capture the output
                        fn_called = True
                        result = await f(self, context, *args, **kwargs)

                        # --------------------------------------------------------------
                        # Append JSON dump of function calls to the output text so that
                        # the consumer can see both in a single attribute.
                        # --------------------------------------------------------------
                        span_output = {"content": output_text}

                        if function_calls_info:
                            span_output["tool_calls"] = function_calls_info

                        try:
                            current_span.set_attribute("output", json.dumps(span_output))
                        except Exception:
                            logger.error(f"Unable to serialize span output: {span_output}")

                        return result

                    finally:
                        # Always restore the original methods
                        self.push_frame = original_push_frame

                        if (
                            "original_start_llm_usage_metrics" in locals()
                            and original_start_llm_usage_metrics
                        ):
                            self.start_llm_usage_metrics = original_start_llm_usage_metrics

                        # Update TTFB metric
                        ttfb: Optional[float] = getattr(
                            getattr(self, "_metrics", None), "ttfb", None
                        )
                        if ttfb is not None:
                            current_span.set_attribute("metrics.ttfb", ttfb)
            except Exception as e:
                if fn_called:
                    raise
                logger.error(f"Error in LLM tracing (continuing without tracing): {e}")
                return await f(self, context, *args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_gemini_live(operation: str) -> Callable:
    """Trace Gemini Live service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:

    - llm_setup: Configuration, tools definitions, and system instructions
    - llm_tool_call: Function call information
    - llm_tool_result: Function execution results
    - llm_response: Complete LLM response with usage and output

    Args:
        operation: The operation name (matches the event type being handled).

    Returns:
        Wrapped method with Gemini Live specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not getattr(self, "_tracing_enabled", False):
                return await func(self, *args, **kwargs)

            fn_called = False
            try:
                service_class_name = self.__class__.__name__
                span_name = f"{operation}"

                # Get the parent context - turn context if available, otherwise service context
                parent_context = _get_turn_context(self) or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Enable sharing public traces
                        current_span.set_attribute("langfuse.trace.public", True)

                        # Base service attributes
                        model_name = _get_model_name(self)
                        voice_id = getattr(self, "_voice_id", None)
                        language_code = getattr(self, "_language_code", None)
                        settings = getattr(self, "_settings", None)

                        # Get modalities if available
                        modalities = None
                        if settings and hasattr(settings, "modalities"):
                            modality_obj = settings.modalities
                            if hasattr(modality_obj, "value"):
                                modalities = modality_obj.value
                            else:
                                modalities = str(modality_obj)

                        # Operation-specific attribute collection
                        operation_attrs = {}

                        if operation == "llm_setup":
                            # Capture detailed tool information
                            tools = getattr(self, "_tools", None)
                            if tools:
                                # Handle different tool formats
                                tools_list = []
                                tools_serialized = None

                                try:
                                    if hasattr(tools, "standard_tools"):
                                        # ToolsSchema object
                                        tools_list = tools.standard_tools
                                        # Serialize the tools for detailed inspection
                                        tools_serialized = json.dumps(
                                            [
                                                {
                                                    "name": tool.name
                                                    if hasattr(tool, "name")
                                                    else tool.get("name", "unknown"),
                                                    "description": tool.description
                                                    if hasattr(tool, "description")
                                                    else tool.get("description", ""),
                                                    "properties": tool.properties
                                                    if hasattr(tool, "properties")
                                                    else tool.get("properties", {}),
                                                    "required": tool.required
                                                    if hasattr(tool, "required")
                                                    else tool.get("required", []),
                                                }
                                                for tool in tools_list
                                            ]
                                        )
                                    elif isinstance(tools, list):
                                        # List of tool dictionaries or objects
                                        tools_list = tools
                                        tools_serialized = json.dumps(
                                            [
                                                {
                                                    "name": tool.get("name", "unknown")
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "name", "unknown"),
                                                    "description": tool.get("description", "")
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "description", ""),
                                                    "properties": tool.get("properties", {})
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "properties", {}),
                                                    "required": tool.get("required", [])
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "required", []),
                                                }
                                                for tool in tools_list
                                            ]
                                        )

                                    if tools_list:
                                        operation_attrs["tools"] = tools_list
                                        operation_attrs["tools_serialized"] = tools_serialized

                                except Exception as e:
                                    logger.warning(f"Error serializing tools for tracing: {e}")
                                    # Fallback to basic tool count
                                    if tools_list:
                                        operation_attrs["tools"] = tools_list

                            # Capture system instruction information
                            system_instruction = getattr(self, "_system_instruction", None)
                            if system_instruction:
                                operation_attrs["system_instruction"] = system_instruction[
                                    :500
                                ]  # Truncate if very long

                            # Capture context system instructions if available
                            if hasattr(self, "_context") and self._context:
                                try:
                                    context_system = self._context.extract_system_instructions()
                                    if context_system:
                                        operation_attrs["context_system_instruction"] = (
                                            context_system[:500]
                                        )  # Truncate if very long
                                except Exception as e:
                                    logger.warning(
                                        f"Error extracting context system instructions: {e}"
                                    )

                        elif operation == "llm_tool_call" and args:
                            # Extract tool call information
                            msg = args[0] if args else None
                            if msg and hasattr(msg, "tool_call") and msg.tool_call.function_calls:
                                function_calls = msg.tool_call.function_calls
                                if function_calls:
                                    # Add information about the first function call
                                    call = function_calls[0]
                                    operation_attrs["tool.function_name"] = call.name
                                    operation_attrs["tool.call_id"] = call.id
                                    operation_attrs["tool.calls_count"] = len(function_calls)

                                    # Add all function names being called
                                    all_function_names = [c.name for c in function_calls]
                                    operation_attrs["tool.all_function_names"] = ",".join(
                                        all_function_names
                                    )

                                    # Add arguments for the first call (truncated if too long)
                                    try:
                                        args_str = json.dumps(call.args) if call.args else "{}"
                                        if len(args_str) > 1000:
                                            args_str = args_str[:1000] + "..."
                                        operation_attrs["tool.arguments"] = args_str
                                    except Exception:
                                        operation_attrs["tool.arguments"] = str(call.args)[:1000]

                        elif operation == "llm_tool_result" and args:
                            # Extract tool result information
                            tool_result_message = args[0] if args else None
                            if tool_result_message and isinstance(tool_result_message, dict):
                                # Extract the tool call information
                                tool_call_id = tool_result_message.get("tool_call_id")
                                tool_call_name = tool_result_message.get("tool_call_name")
                                result_content = tool_result_message.get("content")

                                if tool_call_id:
                                    operation_attrs["tool.call_id"] = tool_call_id
                                if tool_call_name:
                                    operation_attrs["tool.function_name"] = tool_call_name

                                # Parse and capture the result
                                if result_content:
                                    try:
                                        result = json.loads(result_content)
                                        # Serialize the result, truncating if too long
                                        result_str = json.dumps(result)
                                        if len(result_str) > 2000:  # Larger limit for results
                                            result_str = result_str[:2000] + "..."
                                        operation_attrs["tool.result"] = result_str

                                        # Add result status/success indicator if present
                                        if isinstance(result, dict):
                                            if "error" in result:
                                                operation_attrs["tool.result_status"] = "error"
                                            elif "success" in result:
                                                operation_attrs["tool.result_status"] = "success"
                                            else:
                                                operation_attrs["tool.result_status"] = "completed"

                                    except json.JSONDecodeError:
                                        operation_attrs["tool.result"] = (
                                            f"Invalid JSON: {str(result_content)[:500]}"
                                        )
                                        operation_attrs["tool.result_status"] = "parse_error"
                                    except Exception as e:
                                        operation_attrs["tool.result"] = (
                                            f"Error processing result: {str(e)}"
                                        )
                                        operation_attrs["tool.result_status"] = "processing_error"

                        elif operation == "llm_response" and args:
                            # Extract usage and response metadata from turn complete event
                            msg = args[0] if args else None
                            if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                                usage = msg.usage_metadata

                                # Token usage - basic attributes for span visibility
                                if hasattr(usage, "prompt_token_count"):
                                    operation_attrs["tokens.prompt"] = usage.prompt_token_count or 0
                                if hasattr(usage, "response_token_count"):
                                    operation_attrs["tokens.completion"] = (
                                        usage.response_token_count or 0
                                    )
                                if hasattr(usage, "total_token_count"):
                                    operation_attrs["tokens.total"] = usage.total_token_count or 0

                            # Get output text and modality from service state
                            text = getattr(self, "_bot_text_buffer", "")
                            audio_text = getattr(self, "_llm_output_buffer", "")

                            if text:
                                # TEXT modality
                                operation_attrs["output"] = text
                                operation_attrs["output_modality"] = "TEXT"
                            elif audio_text:
                                # AUDIO modality
                                operation_attrs["output"] = audio_text
                                operation_attrs["output_modality"] = "AUDIO"

                            # Add turn completion status
                            if (
                                msg
                                and hasattr(msg, "server_content")
                                and msg.server_content.turn_complete
                            ):
                                operation_attrs["turn_complete"] = True

                        # Add all attributes to the span
                        add_gemini_live_span_attributes(
                            span=current_span,
                            service_name=service_class_name,
                            model=model_name,
                            operation_name=operation,
                            voice_id=voice_id,
                            language=language_code,
                            modalities=modalities,
                            settings=settings,
                            **operation_attrs,
                        )

                        # For llm_response operation, also handle token usage metrics
                        if operation == "llm_response" and hasattr(self, "start_llm_usage_metrics"):
                            msg = args[0] if args else None
                            if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                                usage = msg.usage_metadata
                                # Create LLMTokenUsage object
                                from pipecat.metrics.metrics import LLMTokenUsage

                                tokens = LLMTokenUsage(
                                    prompt_tokens=usage.prompt_token_count or 0,
                                    completion_tokens=usage.response_token_count or 0,
                                    total_tokens=usage.total_token_count or 0,
                                )
                                _add_token_usage_to_span(current_span, tokens)

                        # Capture TTFB metric if available
                        ttfb = getattr(getattr(self, "_metrics", None), "ttfb", None)
                        if ttfb is not None:
                            current_span.set_attribute("metrics.ttfb", ttfb)

                        # Run the original function
                        fn_called = True
                        result = await func(self, *args, **kwargs)

                        return result

                    except Exception as e:
                        current_span.record_exception(e)
                        current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

            except Exception as e:
                if fn_called:
                    raise
                logger.error(f"Error in Gemini Live tracing (continuing without tracing): {e}")
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def traced_openai_realtime(operation: str) -> Callable:
    """Trace OpenAI Realtime service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:

    - llm_setup: Session configuration and tools
    - llm_request: Context and input messages
    - llm_response: Usage metadata, output, and function calls

    Args:
        operation: The operation name (matches the event type being handled).

    Returns:
        Wrapped method with OpenAI Realtime specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not getattr(self, "_tracing_enabled", False):
                return await func(self, *args, **kwargs)

            fn_called = False
            try:
                service_class_name = self.__class__.__name__
                span_name = f"{operation}"

                # Get the parent context - turn context if available, otherwise service context
                parent_context = _get_turn_context(self) or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Enable sharing public traces
                        current_span.set_attribute("langfuse.trace.public", True)

                        # Base service attributes
                        model_name = _get_model_name(self)

                        # Operation-specific attribute collection
                        operation_attrs = {}

                        if operation == "llm_setup":
                            # Capture session properties and tools
                            session_properties = getattr(self, "_session_properties", None)
                            if session_properties:
                                try:
                                    # Convert to dict for easier processing
                                    if hasattr(session_properties, "model_dump"):
                                        props_dict = session_properties.model_dump()
                                    elif hasattr(session_properties, "__dict__"):
                                        props_dict = session_properties.__dict__
                                    else:
                                        props_dict = {}

                                    operation_attrs["session_properties"] = props_dict

                                    # Extract tools if available
                                    tools = props_dict.get("tools")
                                    if tools:
                                        operation_attrs["tools"] = tools
                                        try:
                                            operation_attrs["tools_serialized"] = json.dumps(tools)
                                        except Exception as e:
                                            logger.warning(f"Error serializing OpenAI tools: {e}")

                                    # Extract instructions
                                    instructions = props_dict.get("instructions")
                                    if instructions:
                                        operation_attrs["instructions"] = instructions[:500]

                                except Exception as e:
                                    logger.warning(f"Error processing session properties: {e}")

                            # Also check context for tools
                            if hasattr(self, "_context") and self._context:
                                try:
                                    context_tools = getattr(self._context, "tools", None)
                                    if context_tools and not operation_attrs.get("tools"):
                                        operation_attrs["tools"] = context_tools
                                        operation_attrs["tools_serialized"] = json.dumps(
                                            context_tools
                                        )
                                except Exception as e:
                                    logger.warning(f"Error extracting context tools: {e}")

                        elif operation == "llm_request":
                            # Capture context messages being sent
                            if hasattr(self, "_context") and self._context:
                                try:
                                    messages = self.get_llm_adapter().get_messages_for_logging(
                                        self._context
                                    )
                                    if messages:
                                        operation_attrs["context_messages"] = json.dumps(messages)
                                except Exception as e:
                                    logger.warning(f"Error getting context messages: {e}")

                        elif operation == "llm_response" and args:
                            # Extract usage and response metadata
                            evt = args[0] if args else None
                            if evt and hasattr(evt, "response"):
                                response = evt.response

                                # Token usage - basic attributes for span visibility
                                if hasattr(response, "usage"):
                                    usage = response.usage
                                    if hasattr(usage, "input_tokens"):
                                        operation_attrs["tokens.prompt"] = usage.input_tokens
                                    if hasattr(usage, "output_tokens"):
                                        operation_attrs["tokens.completion"] = usage.output_tokens
                                    if hasattr(usage, "total_tokens"):
                                        operation_attrs["tokens.total"] = usage.total_tokens

                                # Response status and metadata
                                if hasattr(response, "status"):
                                    operation_attrs["response.status"] = response.status

                                if hasattr(response, "id"):
                                    operation_attrs["response.id"] = response.id

                                # Output items and extract transcript for output field
                                if hasattr(response, "output") and response.output:
                                    operation_attrs["response.output_items"] = len(response.output)

                                    # Extract assistant transcript and function calls
                                    assistant_transcript = ""
                                    function_calls = []

                                    for item in response.output:
                                        if (
                                            hasattr(item, "content")
                                            and item.content
                                            and hasattr(item, "role")
                                            and item.role == "assistant"
                                        ):
                                            for content in item.content:
                                                if (
                                                    hasattr(content, "transcript")
                                                    and content.transcript
                                                ):
                                                    assistant_transcript += content.transcript + " "

                                        elif hasattr(item, "type") and item.type == "function_call":
                                            function_call_info = {
                                                "name": getattr(item, "name", "unknown"),
                                                "call_id": getattr(item, "call_id", "unknown"),
                                            }
                                            if hasattr(item, "arguments"):
                                                args_str = item.arguments
                                                if len(args_str) > 500:
                                                    args_str = args_str[:500] + "..."
                                                function_call_info["arguments"] = args_str
                                            function_calls.append(function_call_info)

                                    if assistant_transcript.strip():
                                        operation_attrs["output"] = assistant_transcript.strip()

                                    if function_calls:
                                        operation_attrs["function_calls"] = function_calls
                                        operation_attrs["function_calls.count"] = len(
                                            function_calls
                                        )
                                        all_names = [call["name"] for call in function_calls]
                                        operation_attrs["function_calls.all_names"] = ",".join(
                                            all_names
                                        )

                        # Add all attributes to the span
                        add_openai_realtime_span_attributes(
                            span=current_span,
                            service_name=service_class_name,
                            model=model_name,
                            operation_name=operation,
                            **operation_attrs,
                        )

                        # For llm_response operation, also handle token usage metrics
                        if operation == "llm_response" and hasattr(self, "start_llm_usage_metrics"):
                            evt = args[0] if args else None
                            if evt and hasattr(evt, "response") and hasattr(evt.response, "usage"):
                                usage = evt.response.usage
                                # Create LLMTokenUsage object
                                from pipecat.metrics.metrics import LLMTokenUsage

                                tokens = LLMTokenUsage(
                                    prompt_tokens=getattr(usage, "input_tokens", 0),
                                    completion_tokens=getattr(usage, "output_tokens", 0),
                                    total_tokens=getattr(usage, "total_tokens", 0),
                                )
                                _add_token_usage_to_span(current_span, tokens)

                            # Capture TTFB metric if available
                            ttfb = getattr(getattr(self, "_metrics", None), "ttfb", None)
                            if ttfb is not None:
                                current_span.set_attribute("metrics.ttfb", ttfb)

                        # Run the original function
                        fn_called = True
                        result = await func(self, *args, **kwargs)

                        return result

                    except Exception as e:
                        current_span.record_exception(e)
                        current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

            except Exception as e:
                if fn_called:
                    raise
                logger.error(f"Error in OpenAI Realtime tracing (continuing without tracing): {e}")
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator
