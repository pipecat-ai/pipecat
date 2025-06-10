#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
from abc import ABC
from typing import Optional

from loguru import logger

from pipecat.utils.utils import obj_count, obj_id


class BaseObject(ABC):
    def __init__(self, *, name: Optional[str] = None):
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
        return self._id

    @property
    def name(self) -> str:
        return self._name

    async def cleanup(self):
        if self._event_tasks:
            event_names, tasks = zip(*self._event_tasks)
            logger.debug(f"{self} waiting on event handlers to finish {list(event_names)}...")
            await asyncio.wait(tasks)

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def add_event_handler(self, event_name: str, handler):
        if event_name in self._event_handlers:
            self._event_handlers[event_name].append(handler)
        else:
            logger.warning(f"Event handler {event_name} not registered")

    def _register_event_handler(self, event_name: str):
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        else:
            logger.warning(f"Event handler {event_name} not registered")

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
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
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    await handler(self, *args, **kwargs)
                else:
                    handler(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in event handler {event_name}: {e}")

    def _event_task_finished(self, task: asyncio.Task):
        tuple_to_remove = next((t for t in self._event_tasks if t[1] == task), None)
        if tuple_to_remove:
            self._event_tasks.discard(tuple_to_remove)

    def __str__(self):
        return self.name
