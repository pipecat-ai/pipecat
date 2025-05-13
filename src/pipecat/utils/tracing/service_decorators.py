#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service-specific OpenTelemetry tracing decorators for Pipecat.

This module provides specialized decorators that automatically capture
rich information about service execution including configuration,
parameters, and performance metrics.
"""

import contextlib
import functools
import inspect
import json
import logging
from typing import Callable, Optional, TypeVar

from opentelemetry import context as context_api
from opentelemetry import trace

from pipecat.utils.tracing.service_attributes import (
    add_llm_span_attributes,
    add_stt_span_attributes,
    add_tts_span_attributes,
)
from pipecat.utils.tracing.setup import OPENTELEMETRY_AVAILABLE, is_tracing_available
from pipecat.utils.tracing.turn_context_provider import get_current_turn_context

T = TypeVar("T")
R = TypeVar("R")


# Internal helper functions
def _noop_decorator(func):
    """No-op fallback decorator when tracing is unavailable."""
    return func


def _get_parent_service_context(self):
    """Get the parent service span context (internal use only).

    This looks for the service span that was created when the service was initialized.

    Args:
        self: The service instance

    Returns:
        Context or None: The parent service context, or None if unavailable
    """
    if not is_tracing_available():
        return None

    # The parent span was created when Traceable was initialized and stored as self._span
    if hasattr(self, "_span") and self._span:
        return trace.set_span_in_context(self._span)

    # If we can't find a stored span, default to current context
    return context_api.get_current()


def _add_token_usage_to_span(span, token_usage):
    """Add token usage metrics to a span (internal use only).

    Args:
        span: The span to add token metrics to
        token_usage: Dictionary or object containing token usage information
    """
    if not is_tracing_available() or not token_usage:
        return

    if isinstance(token_usage, dict):
        if "prompt_tokens" in token_usage:
            span.set_attribute("llm.prompt_tokens", token_usage["prompt_tokens"])
        if "completion_tokens" in token_usage:
            span.set_attribute("llm.completion_tokens", token_usage["completion_tokens"])
    else:
        # Handle LLMTokenUsage object
        span.set_attribute("llm.prompt_tokens", getattr(token_usage, "prompt_tokens", 0))
        span.set_attribute("llm.completion_tokens", getattr(token_usage, "completion_tokens", 0))


def traced_tts(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Traces TTS service methods with TTS-specific attributes.

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
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        is_async_generator = inspect.isasyncgenfunction(f)

        @contextlib.asynccontextmanager
        async def tracing_context(self, text):
            """Async context manager for TTS tracing."""
            if not is_tracing_available():
                yield None
                return

            service_class_name = self.__class__.__name__
            span_name = f"tts_{service_class_name.lower()}"

            # Get parent context
            turn_context = get_current_turn_context()
            parent_context = turn_context or _get_parent_service_context(self)

            # Create span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(span_name, context=parent_context) as span:
                try:
                    add_tts_span_attributes(
                        span=span,
                        service_name=service_class_name,
                        model=getattr(self, "model_name", "unknown"),
                        voice_id=getattr(self, "_voice_id", "unknown"),
                        text=text,
                        settings=getattr(self, "_settings", {}),
                        character_count=len(text),
                        operation_name="tts",
                        cartesia_version=getattr(self, "_cartesia_version", None),
                        context_id=getattr(self, "_context_id", None),
                    )

                    yield span

                except Exception as e:
                    logging.warning(f"Error in TTS tracing: {e}")
                    raise
                finally:
                    # Update TTFB metric at the end
                    ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)
                    if ttfb_ms is not None:
                        span.set_attribute("metrics.ttfb_ms", ttfb_ms)

        if is_async_generator:

            @functools.wraps(f)
            async def gen_wrapper(self, text, *args, **kwargs):
                if not is_tracing_available():
                    async for item in f(self, text, *args, **kwargs):
                        yield item
                    return

                async with tracing_context(self, text):
                    async for item in f(self, text, *args, **kwargs):
                        yield item

            return gen_wrapper
        else:

            @functools.wraps(f)
            async def wrapper(self, text, *args, **kwargs):
                if not is_tracing_available():
                    return await f(self, text, *args, **kwargs)

                async with tracing_context(self, text):
                    result = await f(self, text, *args, **kwargs)
                    return result

            return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_stt(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Traces STT service methods with transcription attributes.

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
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, transcript, is_final, language=None):
            if not is_tracing_available():
                return await f(self, transcript, is_final, language)

            service_class_name = self.__class__.__name__
            span_name = f"stt_{service_class_name.lower()}"

            # Get the turn context first, then fall back to service context
            turn_context = get_current_turn_context()
            parent_context = turn_context or _get_parent_service_context(self)

            # Create a new span as child of the turn span or service span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(span_name, context=parent_context) as current_span:
                try:
                    # Get TTFB metric if available
                    ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)

                    # Use settings from the service if available
                    settings = getattr(self, "_settings", {})

                    add_stt_span_attributes(
                        span=current_span,
                        service_name=service_class_name,
                        model=getattr(self, "model_name", settings.get("model", "unknown")),
                        transcript=transcript,
                        is_final=is_final,
                        language=str(language) if language else None,
                        vad_enabled=getattr(self, "vad_enabled", False),
                        settings=settings,
                        ttfb_ms=ttfb_ms,
                    )

                    # Call the original function
                    return await f(self, transcript, is_final, language)
                except Exception as e:
                    # Log any exception but don't disrupt the main flow
                    logging.warning(f"Error in STT transcription tracing: {e}")
                    raise

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_llm(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Traces LLM service methods with LLM-specific attributes.

    Automatically captures and records:
    - Service name and model information
    - Context content and messages
    - Tool configurations
    - Token usage metrics
    - Performance metrics like TTFB

    Args:
        func: The LLM method to trace.
        name: Custom span name. Defaults to service type and class name.

    Returns:
        Wrapped method with LLM-specific tracing.
    """
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, context, *args, **kwargs):
            if not is_tracing_available():
                return await f(self, context, *args, **kwargs)

            service_class_name = self.__class__.__name__
            span_name = f"llm_{service_class_name.lower()}"

            # Get the parent context - turn context if available, otherwise service context
            turn_context = get_current_turn_context()
            parent_context = turn_context or _get_parent_service_context(self)

            # Create a new span as child of the turn span or service span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(span_name, context=parent_context) as current_span:
                try:
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
                        # Detect if we're using Google's service
                        is_google_service = "google" in service_class_name.lower()

                        # Try to get messages based on service type
                        messages = None
                        serialized_messages = None

                        if is_google_service:
                            # Handle Google service specifically
                            if hasattr(context, "get_messages_for_logging"):
                                messages = context.get_messages_for_logging()
                        else:
                            # Handle other services like OpenAI
                            if hasattr(context, "get_messages"):
                                messages = context.get_messages()
                            elif hasattr(context, "messages"):
                                messages = context.messages

                        # Serialize messages if available
                        if messages:
                            try:
                                serialized_messages = json.dumps(messages)
                            except Exception as e:
                                serialized_messages = f"Error serializing messages: {str(e)}"

                        # Get tools, system message, etc. based on the service type
                        tools = getattr(context, "tools", None)
                        serialized_tools = None
                        tool_count = 0

                        if tools:
                            try:
                                serialized_tools = json.dumps(tools)
                                tool_count = len(tools) if isinstance(tools, list) else 1
                            except Exception as e:
                                serialized_tools = f"Error serializing tools: {str(e)}"

                        # Handle system message for different services
                        system_message = None
                        if hasattr(context, "system"):
                            system_message = context.system
                        elif hasattr(context, "system_message"):
                            system_message = context.system_message
                        elif hasattr(self, "_system_instruction"):
                            system_message = self._system_instruction

                        # Get settings from the service
                        params = {}
                        if hasattr(self, "_settings"):
                            for key, value in self._settings.items():
                                if key == "extra":
                                    continue
                                # Add value directly if it's a basic type
                                if isinstance(value, (int, float, bool, str)):
                                    params[key] = value
                                elif value is None or (
                                    hasattr(value, "__name__") and value.__name__ == "NOT_GIVEN"
                                ):
                                    params[key] = "NOT_GIVEN"

                        # Add all available attributes to the span
                        attribute_kwargs = {
                            "service_name": service_class_name,
                            "model": getattr(self, "model_name", "unknown"),
                            "stream": True,  # Most LLM services use streaming
                            "parameters": params,
                        }

                        # Add optional attributes only if they exist
                        if serialized_messages:
                            attribute_kwargs["messages"] = serialized_messages
                        if serialized_tools:
                            attribute_kwargs["tools"] = serialized_tools
                            attribute_kwargs["tool_count"] = tool_count
                        if system_message:
                            attribute_kwargs["system"] = system_message

                        # Add all gathered attributes to the span
                        add_llm_span_attributes(span=current_span, **attribute_kwargs)
                    except Exception as e:
                        logging.warning(f"Error adding initial LLM attributes: {e}")

                    # Call the original function
                    return await f(self, context, *args, **kwargs)
                finally:
                    # Restore the original methods if we overrode them
                    if (
                        "original_start_llm_usage_metrics" in locals()
                        and original_start_llm_usage_metrics
                    ):
                        self.start_llm_usage_metrics = original_start_llm_usage_metrics

                    # Update TTFB metric
                    ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)
                    if ttfb_ms is not None:
                        current_span.set_attribute("metrics.ttfb_ms", ttfb_ms)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
