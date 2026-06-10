#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking worker methods as job handlers."""

from collections.abc import Callable


def job(*, name: str, sequential: bool = False):
    """Mark a worker method as a job handler.

    Decorated methods are automatically collected by ``BaseWorker`` at
    initialization and dispatched when matching job requests arrive.
    Each request runs in its own asyncio task so the bus message loop
    is never blocked.

    Example::

        @job(name="research")
        async def on_research(self, message):
            ...

        @job(name="write", sequential=True)
        async def on_write(self, message):
            ...

    Args:
        name: Job name to match. The handler only receives requests
            with a matching name.
        sequential: When ``True``, requests with this name run one at
            a time in FIFO order. Concurrent requests wait for the
            previous one to finish before running. When ``False`` (the
            default), multiple requests run concurrently. The wait
            time counts against the requester's timeout, so a slow
            predecessor can cause queued requests to time out before
            they start.
    """

    def decorator(fn: Callable) -> Callable:
        fn.is_job_handler = True  # type: ignore[attr-defined]
        fn.job_name = name  # type: ignore[attr-defined]
        fn.job_sequential = sequential  # type: ignore[attr-defined]
        return fn

    return decorator


def _collect_job_handlers(obj) -> dict[str, Callable]:
    seen: set[str] = set()
    handlers: dict[str, Callable] = {}
    for cls in type(obj).__mro__:
        for attr_name, val in cls.__dict__.items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            if callable(val) and getattr(val, "is_job_handler", False):
                job_name: str = getattr(val, "job_name")
                if job_name in handlers:
                    existing = handlers[job_name].__name__
                    raise ValueError(
                        f"Duplicate @job handler for '{job_name}': "
                        f"'{attr_name}' conflicts with '{existing}'"
                    )
                handlers[job_name] = getattr(obj, attr_name)
    return handlers
