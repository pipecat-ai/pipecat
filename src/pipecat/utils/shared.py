#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorators for operations on a resource shared by several owners.

An input and an output transport share a single client, so both set it up, both
join its room, and both tear it down. Processors are set up and cleaned up
concurrently, so those calls overlap.

Decorate the paired operations with :func:`acquires` and :func:`releases` and
the work runs once: the first owner to acquire runs the body while the rest wait
for it, and only the last owner to release runs the undo.
"""

import asyncio
import functools
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

_RESOURCES_ATTR = "__shared_resources"


class _SharedResource:
    """How many owners hold a resource, and the lock serializing them."""

    def __init__(self):
        self.owners = 0
        self.lock = asyncio.Lock()
        # What acquiring raised, if it did, so later owners fail the same way.
        self.error: BaseException | None = None


def _shared_resource(obj: Any, name: str) -> _SharedResource:
    """Get the named resource's state, creating it on first use.

    The state lives on the instance so a class doesn't need to initialize it,
    and so two instances of the same class count their owners separately.
    """
    resources = obj.__dict__.setdefault(_RESOURCES_ATTR, {})
    if name not in resources:
        resources[name] = _SharedResource()
    return resources[name]


def acquires(
    name: str,
) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Coroutine[Any, Any, _T | None]]]:
    """Run the decorated method for the first owner to acquire ``name``.

    Owners that arrive while the first is still running wait for it to finish,
    so no caller continues against a half-built resource. Later owners return
    ``None`` without running the method again.

    A method that raises leaves the resource unbuilt, and every owner of it is
    told: the exception is re-raised to each one that arrives later, rather
    than the method being attempted again. So the owners of a resource share
    its verdict, and a pair of processors sharing a client either both come up
    or both fail.

    Args:
        name: The resource being acquired. Names are per instance, so paired
            operations on the same object (e.g. ``setup``/``cleanup`` and
            ``join``/``leave``) count their owners separately.

    Note:
        The lock is not reentrant, so a method must not call another method
        that acquires the same resource.
    """

    def decorator(
        func: Callable[_P, Awaitable[_T]],
    ) -> Callable[_P, Coroutine[Any, Any, _T | None]]:
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T | None:
            resource = _shared_resource(args[0], name)
            async with resource.lock:
                if resource.error is not None:
                    raise resource.error
                resource.owners += 1
                if resource.owners == 1:
                    try:
                        return await func(*args, **kwargs)
                    except BaseException as e:
                        # Nothing was built, so nothing is owned and nothing is
                        # released later.
                        resource.owners -= 1
                        resource.error = e
                        raise
            return None

        return wrapper

    return decorator


def releases(
    name: str,
) -> Callable[[Callable[_P, Awaitable[_T]]], Callable[_P, Coroutine[Any, Any, _T | None]]]:
    """Run the decorated method for the last owner to release ``name``.

    Every other owner returns ``None``, as does a release that no acquire
    matched.

    Args:
        name: The resource being released, matching the :func:`acquires` name.

    Note:
        The lock is not reentrant, so a method must not call another method
        that releases the same resource.
    """

    def decorator(
        func: Callable[_P, Awaitable[_T]],
    ) -> Callable[_P, Coroutine[Any, Any, _T | None]]:
        @functools.wraps(func)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T | None:
            resource = _shared_resource(args[0], name)
            async with resource.lock:
                if resource.owners == 0:
                    return None
                resource.owners -= 1
                if resource.owners == 0:
                    return await func(*args, **kwargs)
            return None

        return wrapper

    return decorator
