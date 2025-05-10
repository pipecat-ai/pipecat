#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenTelemetry service-specific tracing decorators for Pipecat.

This module provides specialized tracing decorators for different service types.
"""

import functools
import inspect
import logging
from typing import Callable, Optional, TypeVar

from opentelemetry import context as context_api
from opentelemetry import trace

from pipecat.utils.tracing.context_provider import get_current_turn_context
from pipecat.utils.tracing.helpers import (
    add_llm_span_attributes,
    add_stt_span_attributes,
    add_tts_span_attributes,
)
from pipecat.utils.tracing.tracing import (
    OPENTELEMETRY_AVAILABLE,
    is_tracing_available,
)

T = TypeVar("T")
R = TypeVar("R")


# Fallback for when OpenTelemetry is not available
def _noop_decorator(func):
    return func


def get_parent_service_context(self):
    """Get the parent service span context.

    This looks for the service span that was created when the service was initialized.
    """
    if not is_tracing_available():
        return None

    # The parent span was created when Traceable was initialized and stored as self._span
    if hasattr(self, "_span") and self._span:
        return trace.set_span_in_context(self._span)

    # If we can't find a stored span, default to current context
    return context_api.get_current()


def traced_tts(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Decorator specifically for TTS service methods that automatically adds TTS attributes."""
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else func

    def decorator(f):
        # Check if we're dealing with a coroutine or an async generator
        is_async_generator = inspect.isasyncgenfunction(f)

        if is_async_generator:

            @functools.wraps(f)
            async def gen_wrapper(self, text, *args, **kwargs):
                if not is_tracing_available():
                    async for item in f(self, text, *args, **kwargs):
                        yield item
                    return

                # Get the turn context first, then fall back to service context
                turn_context = get_current_turn_context()
                parent_context = turn_context or get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    name or f.__name__, context=parent_context
                ) as current_span:
                    try:
                        # Immediately add attributes to the span
                        service_name = self.__class__.__name__.replace("TTSService", "").lower()

                        # Add TTS attributes right away
                        add_tts_span_attributes(
                            span=current_span,
                            service_name=service_name,
                            model=getattr(self, "model_name", "unknown"),
                            voice_id=getattr(self, "_voice_id", "unknown"),
                            text=text,
                            settings=getattr(self, "_settings", {}),
                            character_count=len(text),
                            operation_name="tts",
                            cartesia_version=getattr(self, "_cartesia_version", None),
                            context_id=getattr(self, "_context_id", None),
                        )

                        # For async generators, we need to yield from it
                        async for item in f(self, text, *args, **kwargs):
                            yield item

                    except Exception as e:
                        # Log any exception but don't disrupt the main flow
                        logging.warning(f"Error in TTS tracing: {e}")
                        raise
                    finally:
                        # Update TTFB metric at the end
                        ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)
                        if ttfb_ms is not None:
                            current_span.set_attribute("metrics.ttfb_ms", ttfb_ms)

            return gen_wrapper
        else:

            @functools.wraps(f)
            async def wrapper(self, text, *args, **kwargs):
                if not is_tracing_available():
                    return await f(self, text, *args, **kwargs)

                # Get the parent service context
                parent_context = get_parent_service_context(self)

                # Create a new span as child of the service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    name or f.__name__, context=parent_context
                ) as current_span:
                    try:
                        # Immediately add attributes to the span
                        service_name = self.__class__.__name__.replace("TTSService", "").lower()

                        # Add TTS attributes right away
                        add_tts_span_attributes(
                            span=current_span,
                            service_name=service_name,
                            model=getattr(self, "model_name", "unknown"),
                            voice_id=getattr(self, "_voice_id", "unknown"),
                            text=text,
                            settings=getattr(self, "_settings", {}),
                            character_count=len(text),
                            operation_name="tts",
                            cartesia_version=getattr(self, "_cartesia_version", None),
                            context_id=getattr(self, "_context_id", None),
                        )

                        # Call the function
                        return await f(self, text, *args, **kwargs)
                    except Exception as e:
                        # Log any exception but don't disrupt the main flow
                        logging.warning(f"Error in TTS tracing: {e}")
                        raise
                    finally:
                        # Update TTFB metric at the end
                        ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)
                        if ttfb_ms is not None:
                            current_span.set_attribute("metrics.ttfb_ms", ttfb_ms)

            return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_stt(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """Decorator specifically for STT service methods that automatically adds STT attributes."""
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else func

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, audio, *args, **kwargs):
            if not is_tracing_available():
                return await f(self, audio, *args, **kwargs)

            # Get the turn context first, then fall back to service context
            turn_context = get_current_turn_context()
            parent_context = turn_context or get_parent_service_context(self)

            # Create a new span as child of the turn span or service span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(
                name or f.__name__, context=parent_context
            ) as current_span:
                try:
                    # Immediately add attributes to the span
                    service_name = self.__class__.__name__.replace("STTService", "").lower()
                    settings = getattr(self, "_settings", {})

                    # Add STT attributes
                    add_stt_span_attributes(
                        span=current_span,
                        service_name=service_name,
                        model=getattr(self, "model_name", settings.get("model", "unknown")),
                        settings=settings,
                        vad_enabled=getattr(self, "vad_enabled", False),
                    )

                    # Call the function
                    return await f(self, audio, *args, **kwargs)
                finally:
                    # Update TTFB metric at the end
                    ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)
                    if ttfb_ms is not None:
                        current_span.set_attribute("metrics.ttfb_ms", ttfb_ms)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_stt_transcription(
    func: Optional[Callable] = None, *, name: Optional[str] = None
) -> Callable:
    """Decorator for STT transcription handling that automatically adds STT attributes."""
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else func

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, transcript, is_final, language=None):
            if not is_tracing_available():
                return await f(self, transcript, is_final, language)

            # Get the turn context first, then fall back to service context
            turn_context = get_current_turn_context()
            parent_context = turn_context or get_parent_service_context(self)

            # Create a new span as child of the turn span or service span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(
                name or f.__name__, context=parent_context
            ) as current_span:
                try:
                    # Get service name from class name
                    service_name = self.__class__.__name__.replace("STTService", "").lower()

                    # Get TTFB metric if available
                    ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)

                    # Use settings from the service if available
                    settings = getattr(self, "_settings", {})

                    # Add all STT attributes immediately
                    add_stt_span_attributes(
                        span=current_span,
                        service_name=service_name,
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
    """Decorator specifically for LLM service methods that automatically adds LLM attributes."""
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else func

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, context, *args, **kwargs):
            if not is_tracing_available():
                return await f(self, context, *args, **kwargs)

            # Store the current turn context before function call
            # This will be used to restore function call traces
            turn_context = get_current_turn_context()

            # Get the service parent context
            parent_context = turn_context or get_parent_service_context(self)

            # Create a new span as child of the turn span or service span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(
                name or f.__name__, context=parent_context
            ) as current_span:
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
                            add_token_usage_to_span(current_span, tokens)

                        # Replace the method temporarily
                        self.start_llm_usage_metrics = wrapped_start_llm_usage_metrics

                    # Add basic attributes immediately
                    service_name = self.__class__.__name__.replace("LLMService", "").lower()

                    try:
                        import json

                        # Detect if we're using Google's service
                        is_google_service = "google" in service_name.lower()

                        # Try to get messages based on service type
                        messages = None
                        serialized_messages = None

                        if is_google_service:
                            # Handle Google service specifically
                            if hasattr(context, "get_messages_for_logging"):
                                messages = context.get_messages_for_logging()
                                # Google messages are already pre-formatted for logging, so we can just use them
                                try:
                                    serialized_messages = json.dumps(messages)
                                except Exception as e:
                                    serialized_messages = (
                                        f"Error serializing Google messages: {str(e)}"
                                    )
                        else:
                            # Handle other services like OpenAI
                            if hasattr(context, "get_messages"):
                                messages = context.get_messages()
                            elif hasattr(context, "messages"):
                                messages = context.messages

                            # For OpenAI-style messages, use standard JSON serialization
                            if messages:
                                try:
                                    # Standard message format should serialize fine
                                    serialized_messages = json.dumps(messages)
                                except Exception as e:
                                    serialized_messages = f"Error serializing messages: {str(e)}"

                        # Get tools, system message, etc. based on the service type
                        tools = getattr(context, "tools", None)
                        serialized_tools = None
                        tool_count = 0

                        if tools:
                            try:
                                if is_google_service:
                                    # Special handling for Google tools
                                    serialized_tools = json.dumps(tools)
                                else:
                                    # Standard serialization for other services
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
                            "service_name": service_name,
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

                    # Patch the LLM service's call_function method to ensure function calls use the same turn context
                    original_call_function = None
                    if hasattr(self, "call_function"):
                        original_call_function = self.call_function

                        @functools.wraps(original_call_function)
                        async def wrapped_call_function(*call_args, **call_kwargs):
                            # Use the turn_context when making function calls
                            if turn_context and is_tracing_available():
                                # This is key: temporarily set the global current span to be in the turn context
                                # before executing the function call
                                ctx_token = context_api.attach(turn_context)
                                try:
                                    return await original_call_function(*call_args, **call_kwargs)
                                finally:
                                    context_api.detach(ctx_token)
                            else:
                                return await original_call_function(*call_args, **call_kwargs)

                        # Replace temporarily
                        self.call_function = wrapped_call_function

                    # Call the original function
                    return await f(self, context, *args, **kwargs)
                finally:
                    # Restore the original methods if we overrode them
                    if (
                        "original_start_llm_usage_metrics" in locals()
                        and original_start_llm_usage_metrics
                    ):
                        self.start_llm_usage_metrics = original_start_llm_usage_metrics

                    if "original_call_function" in locals() and original_call_function:
                        self.call_function = original_call_function

                    # Update TTFB metric
                    ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)
                    if ttfb_ms is not None:
                        current_span.set_attribute("metrics.ttfb_ms", ttfb_ms)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_llm_chat_completion(
    func: Optional[Callable] = None, *, name: Optional[str] = None
) -> Callable:
    """Decorator for LLM chat completion that adds detailed attributes about the request."""
    if not OPENTELEMETRY_AVAILABLE:
        return _noop_decorator if func is None else func

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, context, messages, *args, **kwargs):
            if not is_tracing_available():
                return await f(self, context, messages, *args, **kwargs)

            # Get the turn context first, then fall back to service context
            turn_context = get_current_turn_context()
            parent_context = turn_context or get_parent_service_context(self)

            # Create a new span as child of the turn span or service span
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(
                name or f.__name__, context=parent_context
            ) as current_span:
                try:
                    try:
                        import json

                        # Helper function for serialization
                        def prepare_for_json(obj):
                            if isinstance(obj, dict):
                                return {k: prepare_for_json(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [prepare_for_json(item) for item in obj]
                            elif obj is None:
                                return None
                            elif hasattr(obj, "__name__") and obj.__name__ == "NOT_GIVEN":
                                return "NOT_GIVEN"
                            else:
                                return obj

                        # Get service name
                        service_name = self.__class__.__name__.replace("LLMService", "").lower()

                        # Try to serialize messages
                        serialized_messages = None
                        try:
                            serialized_messages = json.dumps(prepare_for_json(messages))
                        except Exception as e:
                            serialized_messages = f"Error serializing messages: {str(e)}"

                        # Get tools and tool_choice from context
                        serialized_tools = None
                        serialized_tool_choice = None
                        tool_count = 0

                        if hasattr(context, "tools") and context.tools:
                            try:
                                serialized_tools = json.dumps(prepare_for_json(context.tools))
                                tool_count = len(context.tools)
                            except Exception as e:
                                serialized_tools = f"Error serializing tools: {str(e)}"

                        # Handle tool_choice
                        if hasattr(context, "tool_choice"):
                            tool_choice = context.tool_choice
                            if tool_choice is None or (
                                hasattr(tool_choice, "__name__")
                                and tool_choice.__name__ == "NOT_GIVEN"
                            ):
                                serialized_tool_choice = "NOT_GIVEN"
                            else:
                                try:
                                    serialized_tool_choice = json.dumps(
                                        prepare_for_json(tool_choice)
                                    )
                                except Exception as e:
                                    serialized_tool_choice = (
                                        f"Error serializing tool_choice: {str(e)}"
                                    )

                        # Get parameters from settings
                        parameters = {}
                        extra_parameters = {}

                        if hasattr(self, "_settings"):
                            for key, value in self._settings.items():
                                if key == "extra":
                                    continue
                                if value is None or (
                                    hasattr(value, "__name__") and value.__name__ == "NOT_GIVEN"
                                ):
                                    parameters[key] = "NOT_GIVEN"
                                elif isinstance(value, (int, float, bool)):
                                    parameters[key] = value

                            # Add extra parameters
                            if "extra" in self._settings and isinstance(
                                self._settings["extra"], dict
                            ):
                                for key, value in self._settings["extra"].items():
                                    if value is None or (
                                        hasattr(value, "__name__") and value.__name__ == "NOT_GIVEN"
                                    ):
                                        extra_parameters[key] = "NOT_GIVEN"
                                    elif isinstance(value, (int, float, bool, str)):
                                        extra_parameters[key] = value

                        # Add all attributes to the span immediately
                        add_llm_span_attributes(
                            span=current_span,
                            service_name=service_name,
                            model=getattr(self, "model_name", "unknown"),
                            stream=True,  # Assuming streaming is always enabled
                            messages=serialized_messages,
                            tools=serialized_tools,
                            tool_count=tool_count,
                            tool_choice=serialized_tool_choice,
                            parameters=parameters,
                            extra_parameters=extra_parameters,
                        )
                    except Exception as e:
                        # If anything goes wrong with tracing, log it but don't fail the main function
                        logging.warning(f"Error adding LLM chat completion span attributes: {e}")

                    # Call the original function
                    return await f(self, context, messages, *args, **kwargs)
                finally:
                    # Update TTFB metric
                    ttfb_ms = getattr(getattr(self, "_metrics", None), "ttfb_ms", None)
                    if ttfb_ms is not None:
                        current_span.set_attribute("metrics.ttfb_ms", ttfb_ms)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def add_token_usage_to_span(span, token_usage):
    """Helper function to add token usage metrics to a span."""
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
