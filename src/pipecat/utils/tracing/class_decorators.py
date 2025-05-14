#
# Copyright (c) 2024–2025, Daily
# Portions Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base OpenTelemetry tracing decorators and utilities for Pipecat.

This module provides class and method level tracing capabilities
similar to the original NVIDIA implementation.
"""

import asyncio
import contextlib
import enum
import functools
import inspect
from typing import Callable, Optional, TypeVar

from pipecat.utils.tracing.setup import is_tracing_available

# Import OpenTelemetry if available
if is_tracing_available():
    import opentelemetry.trace
    from opentelemetry import metrics, trace

# Type variables for better typing support
T = TypeVar("T")
C = TypeVar("C", bound=type)


class AttachmentStrategy(enum.Enum):
    """Controls how spans are attached to the trace hierarchy.

    Attributes:
        CHILD: Attached to class span if no parent, otherwise to parent.
        LINK: Attached to class span with link to parent.
        NONE: Always attached to class span regardless of context.
    """

    CHILD = enum.auto()
    LINK = enum.auto()
    NONE = enum.auto()


class Traceable:
    """Base class for objects that can be traced with OpenTelemetry.

    Provides the foundational tracing capabilities used by @traced methods.
    """

    def __init__(self, name: str, **kwargs):
        """Initialize a traceable object.

        Args:
            name: Name of the traceable object for the span.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        if not is_tracing_available():
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

        Returns:
            Meter: The OpenTelemetry meter instance for this object.
        """
        return self._meter


@contextlib.contextmanager
def __traced_context_manager(
    self: Traceable, func: Callable, name: str | None, attachment_strategy: AttachmentStrategy
):
    """Internal context manager for the traced decorator."""
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
    """Implementation of the traced decorator."""

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
    """Adds tracing to an async function in a Traceable class.

    Args:
        func: The async function to trace.
        name: Custom span name. Defaults to function name.
        attachment_strategy: How to attach this span (CHILD, LINK, NONE).

    Returns:
        Wrapped async function with tracing.

    Raises:
        RuntimeError: If used in a class not inheriting from Traceable.
        ValueError: If used on a non-async function.
    """
    if not is_tracing_available():
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
    """Makes a class traceable for OpenTelemetry.

    Creates a new class that inherits from both the original class
    and Traceable, enabling tracing for class methods.

    Args:
        cls: The class to make traceable.

    Returns:
        A new class with tracing capabilities.
    """
    if not is_tracing_available():
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
