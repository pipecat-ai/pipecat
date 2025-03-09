#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import inspect
from abc import ABC
from typing import Optional

from loguru import logger

from pipecat.utils.utils import obj_count, obj_id


class BaseObject(ABC):
    def __init__(self, *, name: Optional[str] = None):
        self._id: int = obj_id()
        self._name = name or f"{self.__class__.__name__}#{obj_count(self)}"
        self._event_handlers: dict = {}

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def add_event_handler(self, event_name: str, handler):
        if event_name not in self._event_handlers:
            raise Exception(f"Event handler {event_name} not registered")
        self._event_handlers[event_name].append(handler)

    def _register_event_handler(self, event_name: str):
        if event_name in self._event_handlers:
            raise Exception(f"Event handler {event_name} already registered")
        self._event_handlers[event_name] = []

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    await handler(self, *args, **kwargs)
                else:
                    handler(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in event handler {event_name}: {e}")

    def __str__(self):
        return self.name
