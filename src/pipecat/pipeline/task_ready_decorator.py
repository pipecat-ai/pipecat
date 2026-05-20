#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking methods as task-ready handlers."""

from collections.abc import Callable


def task_ready(*, name: str):
    """Mark a method as a handler for a specific task becoming ready.

    Decorated methods are automatically collected by `BaseTask` at
    initialization. When the task starts, it calls `watch_task` for
    each decorated handler. When the watched task registers, the
    decorated method is called with the ready data.

    Example::

        @task_ready(name="greeter")
        async def on_greeter_ready(self, data: TaskReadyData) -> None:
            await self.activate_task("greeter", args=...)

    Args:
        name: The name of the task to watch.
    """

    def decorator(fn):
        fn.task_ready_name = name
        return fn

    return decorator


def _collect_task_ready_handlers(obj) -> dict:
    """Collect all ``@task_ready`` decorated bound methods from an object.

    Returns a dict mapping task name to the bound method.

    Raises:
        ValueError: If two handlers watch the same task name.
    """
    seen: set[str] = set()
    handlers: dict[str, Callable] = {}
    for cls in type(obj).__mro__:
        for attr_name, val in cls.__dict__.items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            if callable(val) and hasattr(val, "task_ready_name"):
                task_name = val.task_ready_name
                if task_name in handlers:
                    existing = handlers[task_name].__name__
                    raise ValueError(
                        f"Duplicate @task_ready handler for '{task_name}': "
                        f"'{attr_name}' conflicts with '{existing}'"
                    )
                handlers[task_name] = getattr(obj, attr_name)
    return handlers
