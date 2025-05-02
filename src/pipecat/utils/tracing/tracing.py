#
# Copyright (c) 2024–2025, Daily
# Portions Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenTelemetry tracing utilities for Pipecat.

This module provides optional tracing functionality using OpenTelemetry.
"""

import asyncio
import contextlib
import enum
import functools
import inspect
import os
from typing import Any, Callable, Optional, TypeVar

# Type variables for better typing support
T = TypeVar("T")
C = TypeVar("C", bound=type)

# Check if OpenTelemetry is available
try:
    import opentelemetry.trace
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


class AttachmentStrategy(enum.Enum):
    """Attachment strategy for the @traced annotation.

    CHILD: span will be attached to the class span if no parent or to its parent otherwise.
    LINK : span will be attached to the class span but linked to its parent.
    NONE : span will be attached to the class span even if nested under another span.
    """

    CHILD = enum.auto()
    LINK = enum.auto()
    NONE = enum.auto()


class Traceable:
    """Base class for traceable objects.

    This class provides tracing functionality using OpenTelemetry. It initializes tracing and metrics
    components and manages spans for tracing operations.
    """

    def __init__(self, name: str, **kwargs):
        """Initialize a traceable object.

        Args:
            name (str): Name of the traceable object used for the initial span
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(**kwargs)

        if not OPENTELEMETRY_AVAILABLE:
            self._tracer = self._meter = self._parent_span_id = self._span = None
            return

        self._tracer = trace.get_tracer("pipecat")
        self._meter = metrics.get_meter("pipecat")
        self._parent_span_id = trace.get_current_span().get_span_context().span_id
        self._span = self._tracer.start_span(name)
        self._span.end()

    @property
    def meter(self):
        """Returns the OpenTelemetry meter instance.

        The meter is used for collecting metrics and measurements in OpenTelemetry.

        Returns:
            opentelemetry.metrics.Meter: The OpenTelemetry meter instance used by this traceable object.
        """
        return self._meter


@contextlib.contextmanager
def __traced_context_manager(
    self: Traceable, func: Callable, name: str | None, attachment_strategy: AttachmentStrategy
):
    """Internal context manager for the traced decorator.

    Creates spans based on the attachment strategy.
    """
    if not isinstance(self, Traceable):
        raise RuntimeError(
            "@traced annotation can only be used in classes inheriting from Traceable"
        )

    stack = contextlib.ExitStack()
    try:
        current_span = trace.get_current_span()
        is_span_class_parent_span = current_span.get_span_context().span_id == self._parent_span_id
        match attachment_strategy:
            case AttachmentStrategy.CHILD if not is_span_class_parent_span:
                stack.enter_context(
                    self._tracer.start_as_current_span(func.__name__ if name is None else name)  # type: ignore
                )
            case AttachmentStrategy.LINK:
                if is_span_class_parent_span:
                    link = trace.Link(self._span.get_span_context())  # type: ignore
                else:
                    link = trace.Link(current_span.get_span_context())
                stack.enter_context(
                    opentelemetry.trace.use_span(span=self._span, end_on_exit=False)  # type: ignore
                )
                stack.enter_context(
                    self._tracer.start_as_current_span(  # type: ignore
                        func.__name__ if name is None else name, links=[link]
                    )
                )
            case AttachmentStrategy.NONE | AttachmentStrategy.CHILD:
                stack.enter_context(
                    opentelemetry.trace.use_span(span=self._span, end_on_exit=False)  # type: ignore
                )
                stack.enter_context(
                    self._tracer.start_as_current_span(func.__name__ if name is None else name)  # type: ignore
                )
        yield
    finally:
        stack.close()


def __traced_decorator(func, name, attachment_strategy: AttachmentStrategy):
    """Implementation of the traced decorator.

    Handles both coroutines and async generators.
    """

    @functools.wraps(func)
    async def coroutine_wrapper(self: Traceable, *args, **kwargs):
        exception = None
        with __traced_context_manager(self, func, name, attachment_strategy):
            try:
                return await func(self, *args, **kwargs)
            except asyncio.CancelledError as e:
                exception = e
        if exception:
            raise exception

    @functools.wraps(func)
    async def generator_wrapper(self: Traceable, *args, **kwargs):
        exception = None
        with __traced_context_manager(self, func, name, attachment_strategy):
            try:
                async for v in func(self, *args, **kwargs):
                    yield v
            except asyncio.CancelledError as e:
                exception = e
        if exception:
            raise exception

    if inspect.iscoroutinefunction(func):
        return coroutine_wrapper
    if inspect.isasyncgenfunction(func):
        return generator_wrapper

    raise ValueError("@traced annotation can only be used on async or async generator functions")


def traced(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    attachment_strategy: AttachmentStrategy = AttachmentStrategy.CHILD,
) -> Callable:
    """Decorator that adds tracing to an async function.

    This decorator wraps an async function to add OpenTelemetry tracing capabilities.
    It creates a new span for the function and maintains proper context propagation.
    For FrameProcessor process_frame method, it also ensures the parent span is properly set.

    Args:
        func: The async function to be traced.
        name: Name for the span. Defaults to the name of the function.
        attachment_strategy (AttachmentStrategy): The attachment strategy to use (Possible values are
        CHILD, LINK, NONE)

    Returns:
        A wrapped async function with tracing capabilities.

    Raises:
        RuntimeError: If used on a function in a class that doesn't inherit from Traceable.

    Example:
        @traceable
        class MyClass:
            @traced
            async def my_function(self):
                pass
    """
    if not OPENTELEMETRY_AVAILABLE:
        # Just return the original function or a simple decorator
        def decorator(f):
            return f

        return decorator if func is None else func

    if func is not None:
        return __traced_decorator(func, name=name, attachment_strategy=attachment_strategy)
    else:
        return functools.partial(
            __traced_decorator, name=name, attachment_strategy=attachment_strategy
        )


def traceable(cls: C) -> C:
    """Class decorator that makes a class traceable for OpenTelemetry instrumentation.

    This decorator creates a new class that inherits from both the original class and
    the Traceable base class. The new class will be ready to be instrumented for tracing using
    the @traced decorator.

    Args:
        cls: The class to make traceable.

    Returns:
        TracedClass: A new class that inherits from both the input class and Traceable,
            with tracing capabilities added.

    Example:
        @traceable
        class MyClass:
            @traced
            def my_method(self):
                pass
    """
    if not OPENTELEMETRY_AVAILABLE:
        return cls

    @functools.wraps(cls, updated=())
    class TracedClass(cls, Traceable):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            if hasattr(self, "name"):
                Traceable.__init__(self, self.name)
            else:
                Traceable.__init__(self, cls.__name__)

    return TracedClass


def traced_function(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
) -> Callable:
    """Decorator for standalone functions (not methods of Traceable classes).

    This is simpler than the class-based traced decorator and doesn't require
    the function to be in a Traceable class.

    Args:
        func: The function to trace
        name: Optional name for the span. Defaults to the function name.

    Returns:
        The decorated function
    """
    if not OPENTELEMETRY_AVAILABLE:

        def decorator(f):
            return f

        return decorator if func is None else func

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer("pipecat")
            with tracer.start_as_current_span(name or f.__name__):
                return await f(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def is_tracing_available() -> bool:
    """Returns True if OpenTelemetry tracing is available and configured.

    Returns:
        bool: True if tracing is available, False otherwise.
    """
    return OPENTELEMETRY_AVAILABLE


def setup_tracing(
    service_name: str = "pipecat",
    exporter=None,  # User-provided exporter
    console_export: bool = False,
) -> bool:
    """Set up OpenTelemetry tracing with a user-provided exporter.

    Args:
        service_name: The name of the service for traces
        exporter: A pre-configured OpenTelemetry span exporter instance.
                  If None, only console export will be available if enabled.
        console_export: Whether to also export traces to console (useful for debugging)

    Returns:
        bool: True if setup was successful, False otherwise

    Example:
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
