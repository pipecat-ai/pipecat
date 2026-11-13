#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking methods as worker-ready handlers."""

from collections.abc import Callable


def worker_ready(*, name: str):
    """Mark a method as a handler for a specific worker becoming ready.

    Decorated methods are automatically collected by `BaseWorker` at
    initialization. When the worker starts, it calls `watch_workers` for
    the decorated handlers. When a watched worker registers, the
    decorated method is called with the ready data.

    Example::

        @worker_ready(name="greeter")
        async def on_greeter_ready(self, data: WorkerReadyData) -> None:
            await self.activate_worker("greeter", args=...)

    Args:
        name: The name of the worker to watch.
    """

    def decorator(fn):
        fn.worker_ready_name = name
        return fn

    return decorator


def _collect_worker_ready_handlers(obj) -> dict:
    """Collect all ``@worker_ready`` decorated bound methods from an object.

    Returns a dict mapping worker name to the bound method.

    Raises:
        ValueError: If two handlers watch the same worker name.
    """
    seen: set[str] = set()
    handlers: dict[str, Callable] = {}
    for cls in type(obj).__mro__:
        for attr_name, val in cls.__dict__.items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            if callable(val) and hasattr(val, "worker_ready_name"):
                worker_name = val.worker_ready_name
                if worker_name in handlers:
                    existing = handlers[worker_name].__name__
                    raise ValueError(
                        f"Duplicate @worker_ready handler for '{worker_name}': "
                        f"'{attr_name}' conflicts with '{existing}'"
                    )
                handlers[worker_name] = getattr(obj, attr_name)
    return handlers
