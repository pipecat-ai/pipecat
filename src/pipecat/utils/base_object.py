#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base object class providing event handling and lifecycle management.

This module provides the foundational BaseObject class that offers common
functionality including unique identification, naming, event handling,
and async cleanup for all Pipecat components.
"""

import asyncio
import inspect
from abc import ABC
from typing import Optional

from loguru import logger

from pipecat.utils.utils import obj_count, obj_id


class BaseObject(ABC):
    """Abstract base class providing common functionality for Pipecat objects.

    Provides unique identification, naming, event handling capabilities,
    and async lifecycle management for all Pipecat components. All major
    classes in the framework should inherit from this base class.
    """

    def __init__(self, *, name: Optional[str] = None, **kwargs):
        """Initialize the base object.

        Args:
            name: Optional custom name for the object. If not provided,
                generates a name using the class name and instance count.
            **kwargs: Additional arguments passed to parent class.
        """
        self._id: int = obj_id()
        self._name = name or f"{self.__class__.__name__}#{obj_count(self)}"

        # Registered event handlers.
        self._event_handlers: dict = {}

        # Set of tasks being executed. When a task finishes running it gets
        # automatically removed from the set. When we cleanup we wait for all
        # event tasks still being executed.
        self._event_tasks = set()

    @property
    def id(self) -> int:
        """Get the unique identifier for this object.

        Returns:
            The unique integer ID assigned to this object instance.
        """
        return self._id

    @property
    def name(self) -> str:
        """Get the name of this object.

        Returns:
            The object's name, either custom-provided or auto-generated.
        """
        return self._name

    async def cleanup(self):
        """Clean up resources and wait for running event handlers to complete.

        This method should be called when the object is no longer needed.
        It waits for all currently executing event handler tasks to finish
        before returning.
        """
        if self._event_tasks:
            event_names, tasks = zip(*self._event_tasks)
            logger.debug(f"{self} waiting on event handlers to finish {list(event_names)}...")
            await asyncio.wait(tasks)

    def event_handler(self, event_name: str):
        """Decorator for registering event handlers.

        Args:
            event_name: The name of the event to handle.

        Returns:
            The decorator function that registers the handler.
        """

        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def add_event_handler(self, event_name: str, handler):
        """Add an event handler for the specified event.

        Args:
            event_name: The name of the event to handle.
            handler: The function to call when the event occurs.
                Can be sync or async.
        """
        if event_name in self._event_handlers:
            self._event_handlers[event_name].append(handler)
        else:
            logger.warning(f"Event handler {event_name} not registered")

    def _register_event_handler(self, event_name: str):
        """Register an event handler type.

        Args:
            event_name: The name of the event type to register.
        """
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        else:
            logger.warning(f"Event handler {event_name} not registered")

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        """Call all registered handlers for the specified event.

        Args:
            event_name: The name of the event to trigger.
            *args: Positional arguments to pass to event handlers.
            **kwargs: Keyword arguments to pass to event handlers.
        """
        # If we haven't registered an event handler, we don't need to do
        # anything.
        if not self._event_handlers.get(event_name):
            return

        # Create the task.
        task = asyncio.create_task(self._run_task(event_name, *args, **kwargs))

        # Add it to our list of event tasks.
        self._event_tasks.add((event_name, task))

        # Remove the task from the event tasks list when the task completes.
        task.add_done_callback(self._event_task_finished)

    async def _run_task(self, event_name: str, *args, **kwargs):
        """Execute all handlers for an event.

        Args:
            event_name: The name of the event being handled.
            *args: Positional arguments to pass to handlers.
            **kwargs: Keyword arguments to pass to handlers.
        """
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    await handler(self, *args, **kwargs)
                else:
                    handler(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in event handler {event_name}: {e}")

    def _event_task_finished(self, task: asyncio.Task):
        """Clean up completed event handler tasks.

        Args:
            task: The completed asyncio Task to remove from tracking.
        """
        tuple_to_remove = next((t for t in self._event_tasks if t[1] == task), None)
        if tuple_to_remove:
            self._event_tasks.discard(tuple_to_remove)

    def __str__(self):
        """Return the string representation of this object.

        Returns:
            The object's name as its string representation.
        """
        return self.name
