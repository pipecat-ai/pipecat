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
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

# Type imports for type checking only
if TYPE_CHECKING:
    from opentelemetry import context as context_api
    from opentelemetry import trace

from pipecat.utils.tracing.service_attributes import (
    add_gemini_live_span_attributes,
    add_llm_span_attributes,
    add_openai_realtime_span_attributes,
    add_stt_span_attributes,
    add_tts_span_attributes,
)
from pipecat.utils.tracing.setup import is_tracing_available
from pipecat.utils.tracing.turn_context_provider import get_current_turn_context

if is_tracing_available():
    from opentelemetry import context as context_api
    from opentelemetry import trace

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
            span.set_attribute("gen_ai.usage.input_tokens", token_usage["prompt_tokens"])
        if "completion_tokens" in token_usage:
            span.set_attribute("gen_ai.usage.output_tokens", token_usage["completion_tokens"])
    else:
        # Handle LLMTokenUsage object
        span.set_attribute("gen_ai.usage.input_tokens", getattr(token_usage, "prompt_tokens", 0))
        span.set_attribute(
            "gen_ai.usage.output_tokens", getattr(token_usage, "completion_tokens", 0)
        )


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
    if not is_tracing_available():
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
            span_name = "tts"

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
                    ttfb: Optional[float] = getattr(getattr(self, "_metrics", None), "ttfb", None)
                    if ttfb is not None:
                        span.set_attribute("metrics.ttfb", ttfb)

        if is_async_generator:

            @functools.wraps(f)
            async def gen_wrapper(self, text, *args, **kwargs):
                try:
                    if not is_tracing_available():
                        async for item in f(self, text, *args, **kwargs):
                            yield item
                        return

                    async with tracing_context(self, text):
                        async for item in f(self, text, *args, **kwargs):
                            yield item
                except Exception as e:
                    logging.error(f"Error in TTS tracing (continuing without tracing): {e}")
                    # If tracing fails, fall back to the original function
                    async for item in f(self, text, *args, **kwargs):
                        yield item

            return gen_wrapper
        else:

            @functools.wraps(f)
            async def wrapper(self, text, *args, **kwargs):
                try:
                    if not is_tracing_available():
                        return await f(self, text, *args, **kwargs)

                    async with tracing_context(self, text):
                        return await f(self, text, *args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in TTS tracing (continuing without tracing): {e}")
                    # If tracing fails, fall back to the original function
                    return await f(self, text, *args, **kwargs)

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
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, transcript, is_final, language=None):
            try:
                if not is_tracing_available():
                    return await f(self, transcript, is_final, language)

                service_class_name = self.__class__.__name__
                span_name = "stt"

                # Get the turn context first, then fall back to service context
                turn_context = get_current_turn_context()
                parent_context = turn_context or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Get TTFB metric if available
                        ttfb: Optional[float] = getattr(
                            getattr(self, "_metrics", None), "ttfb", None
                        )

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
                            ttfb=ttfb,
                        )

                        # Call the original function
                        return await f(self, transcript, is_final, language)
                    except Exception as e:
                        # Log any exception but don't disrupt the main flow
                        logging.warning(f"Error in STT transcription tracing: {e}")
                        raise
            except Exception as e:
                logging.error(f"Error in STT tracing (continuing without tracing): {e}")
                # If tracing fails, fall back to the original function
                return await f(self, transcript, is_final, language)

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
            try:
                if not is_tracing_available():
                    return await f(self, context, *args, **kwargs)

                service_class_name = self.__class__.__name__
                span_name = "llm"

                # Get the parent context - turn context if available, otherwise service context
                turn_context = get_current_turn_context()
                parent_context = turn_context or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Store original method and output aggregator
                        original_push_frame = self.push_frame
                        output_text = ""  # Simple string accumulation

                        async def traced_push_frame(frame, direction=None):
                            nonlocal output_text
                            # Capture text from LLMTextFrame during streaming
                            if (
                                hasattr(frame, "__class__")
                                and frame.__class__.__name__ == "LLMTextFrame"
                                and hasattr(frame, "text")
                            ):
                                output_text += frame.text

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

                            # Detect if we're using Google's service
                            is_google_service = "google" in service_class_name.lower()

                            # Try to get messages based on service type
                            messages = None
                            serialized_messages = None

                            # TODO: Revisit once we unify the messages across services
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
                            logging.warning(f"Error setting up LLM tracing: {e}")
                            # Don't raise - let the function execute anyway

                        # Run function with modified push_frame to capture the output
                        result = await f(self, context, *args, **kwargs)

                        # Add aggregated output after function completes, if available
                        if output_text:
                            current_span.set_attribute("output", output_text)

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
                logging.error(f"Error in LLM tracing (continuing without tracing): {e}")
                # If tracing fails, fall back to the original function
                return await f(self, context, *args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_gemini_live(operation: str) -> Callable:
    """Traces Gemini Live service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:
    - llm_setup: Configuration, tools definitions, and system instructions
    - llm_tool_call: Function call information
    - llm_tool_result: Function execution results
    - llm_response: Complete LLM response with usage and output

    Args:
        operation: The operation name (matches the event type being handled)

    Returns:
        Wrapped method with Gemini Live specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                if not is_tracing_available():
                    return await func(self, *args, **kwargs)

                service_class_name = self.__class__.__name__
                span_name = f"{operation}"

                # Get the parent context - turn context if available, otherwise service context
                turn_context = get_current_turn_context()
                parent_context = turn_context or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Base service attributes
                        model_name = getattr(
                            self, "model_name", getattr(self, "_model_name", "unknown")
                        )
                        voice_id = getattr(self, "_voice_id", None)
                        language_code = getattr(self, "_language_code", None)
                        settings = getattr(self, "_settings", {})

                        # Get modalities if available
                        modalities = None
                        if hasattr(self, "_settings") and "modalities" in self._settings:
                            modality_obj = self._settings["modalities"]
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
                                    logging.warning(f"Error serializing tools for tracing: {e}")
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
                                    logging.warning(
                                        f"Error extracting context system instructions: {e}"
                                    )

                        elif operation == "llm_tool_call" and args:
                            # Extract tool call information
                            evt = args[0] if args else None
                            if evt and hasattr(evt, "toolCall") and evt.toolCall.functionCalls:
                                function_calls = evt.toolCall.functionCalls
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

                                    except json.JSONDecodeError as e:
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
                            evt = args[0] if args else None
                            if evt and hasattr(evt, "usageMetadata") and evt.usageMetadata:
                                usage = evt.usageMetadata

                                # Token usage - basic attributes for span visibility
                                if hasattr(usage, "promptTokenCount"):
                                    operation_attrs["tokens.prompt"] = usage.promptTokenCount or 0
                                if hasattr(usage, "responseTokenCount"):
                                    operation_attrs["tokens.completion"] = (
                                        usage.responseTokenCount or 0
                                    )
                                if hasattr(usage, "totalTokenCount"):
                                    operation_attrs["tokens.total"] = usage.totalTokenCount or 0

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
                                evt
                                and hasattr(evt, "serverContent")
                                and evt.serverContent.turnComplete
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
                            evt = args[0] if args else None
                            if evt and hasattr(evt, "usageMetadata") and evt.usageMetadata:
                                usage = evt.usageMetadata
                                # Create LLMTokenUsage object
                                from pipecat.metrics.metrics import LLMTokenUsage

                                tokens = LLMTokenUsage(
                                    prompt_tokens=usage.promptTokenCount or 0,
                                    completion_tokens=usage.responseTokenCount or 0,
                                    total_tokens=usage.totalTokenCount or 0,
                                )
                                _add_token_usage_to_span(current_span, tokens)

                        # Capture TTFB metric if available
                        ttfb = getattr(getattr(self, "_metrics", None), "ttfb", None)
                        if ttfb is not None:
                            current_span.set_attribute("metrics.ttfb", ttfb)

                        # Run the original function
                        result = await func(self, *args, **kwargs)

                        return result

                    except Exception as e:
                        current_span.record_exception(e)
                        current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

            except Exception as e:
                logging.error(f"Error in Gemini Live tracing (continuing without tracing): {e}")
                # If tracing fails, fall back to the original function
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def traced_openai_realtime(operation: str) -> Callable:
    """Traces OpenAI Realtime service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:
    - llm_setup: Session configuration and tools
    - llm_request: Context and input messages
    - llm_response: Usage metadata, output, and function calls

    Args:
        operation: The operation name (matches the event type being handled)

    Returns:
        Wrapped method with OpenAI Realtime specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                if not is_tracing_available():
                    return await func(self, *args, **kwargs)

                service_class_name = self.__class__.__name__
                span_name = f"{operation}"

                # Get the parent context - turn context if available, otherwise service context
                turn_context = get_current_turn_context()
                parent_context = turn_context or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Base service attributes
                        model_name = getattr(
                            self, "model_name", getattr(self, "_model_name", "unknown")
                        )

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
                                            logging.warning(f"Error serializing OpenAI tools: {e}")

                                    # Extract instructions
                                    instructions = props_dict.get("instructions")
                                    if instructions:
                                        operation_attrs["instructions"] = instructions[:500]

                                except Exception as e:
                                    logging.warning(f"Error processing session properties: {e}")

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
                                    logging.warning(f"Error extracting context tools: {e}")

                        elif operation == "llm_request":
                            # Capture context messages being sent
                            if hasattr(self, "_context") and self._context:
                                try:
                                    messages = self._context.get_messages_for_logging()
                                    if messages:
                                        operation_attrs["context_messages"] = json.dumps(messages)
                                except Exception as e:
                                    logging.warning(f"Error getting context messages: {e}")

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
                        result = await func(self, *args, **kwargs)

                        return result

                    except Exception as e:
                        current_span.record_exception(e)
                        current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

            except Exception as e:
                logging.error(f"Error in OpenAI Realtime tracing (continuing without tracing): {e}")
                # If tracing fails, fall back to the original function
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator
