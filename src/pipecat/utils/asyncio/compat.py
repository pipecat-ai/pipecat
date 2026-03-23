#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async backend compatibility layer.

This module provides backend-agnostic async primitives that work on both
``asyncio`` and ``trio`` via `anyio <https://anyio.readthedocs.io/>`_.

Pipecat is historically asyncio-only. This module is the first step toward
supporting `trio <https://trio.readthedocs.io/>`_ as an alternative event
loop. New code should prefer these primitives over raw ``asyncio`` where
possible so that it can run under either backend.

Usage::

    from pipecat.utils.asyncio.compat import Event, Queue, sleep, current_backend

    if current_backend() == "trio":
        ...

    event = Event()
    queue: Queue[int] = Queue()

The primitives here intentionally mirror the ``asyncio`` API surface that
Pipecat already uses so they can be dropped in with minimal churn.
"""

from __future__ import annotations

import heapq
import math
from typing import Generic, TypeVar

import anyio
import anyio.lowlevel
import sniffio

T = TypeVar("T")


def current_backend() -> str:
    """Return the name of the currently running async backend.

    Returns:
        ``"asyncio"`` or ``"trio"``. Returns ``"asyncio"`` if called
        outside an async context (the historical default for Pipecat).
    """
    try:
        return sniffio.current_async_library()
    except sniffio.AsyncLibraryNotFoundError:
        return "asyncio"


def get_cancelled_exc_class() -> type[BaseException]:
    """Return the cancellation exception class for the current backend.

    Code that catches ``asyncio.CancelledError`` directly will not work
    under trio (which raises ``trio.Cancelled``). Use this helper in
    ``except`` clauses that need to be backend-agnostic::

        try:
            await something()
        except get_cancelled_exc_class():
            raise  # always re-raise cancellation
    """
    return anyio.get_cancelled_exc_class()


async def sleep(seconds: float) -> None:
    """Backend-agnostic sleep."""
    await anyio.sleep(seconds)


async def checkpoint() -> None:
    """Yield control to the scheduler (equivalent to ``asyncio.sleep(0)``)."""
    await anyio.lowlevel.checkpoint()


# Re-export anyio primitives that already match the asyncio API closely
# enough to be drop-in replacements for Pipecat's usage.
Event = anyio.Event
Lock = anyio.Lock
Semaphore = anyio.Semaphore
CapacityLimiter = anyio.CapacityLimiter
CancelScope = anyio.CancelScope
fail_after = anyio.fail_after
move_on_after = anyio.move_on_after
to_thread = anyio.to_thread
from_thread = anyio.from_thread
create_task_group = anyio.create_task_group


class Queue(Generic[T]):
    """Backend-agnostic FIFO queue with an ``asyncio.Queue``-like API.

    Built on top of :func:`anyio.create_memory_object_stream` so it works
    under both asyncio and trio. Only the subset of the ``asyncio.Queue``
    API that Pipecat actually uses is implemented.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """Initialize the queue.

        Args:
            maxsize: Maximum number of items. ``0`` means unbounded.
        """
        buffer_size = math.inf if maxsize <= 0 else maxsize
        self._send, self._recv = anyio.create_memory_object_stream(max_buffer_size=buffer_size)

    async def put(self, item: T) -> None:
        """Put an item into the queue, blocking if full."""
        await self._send.send(item)

    def put_nowait(self, item: T) -> None:
        """Put an item without blocking.

        Raises:
            anyio.WouldBlock: If the queue is full.
        """
        self._send.send_nowait(item)

    async def get(self) -> T:
        """Remove and return an item, blocking until one is available."""
        return await self._recv.receive()

    def get_nowait(self) -> T:
        """Remove and return an item without blocking.

        Raises:
            anyio.WouldBlock: If the queue is empty.
        """
        return self._recv.receive_nowait()

    def qsize(self) -> int:
        """Return the approximate number of items in the queue."""
        return self._send.statistics().current_buffer_used

    def empty(self) -> bool:
        """Return ``True`` if the queue is empty."""
        return self.qsize() == 0

    def task_done(self) -> None:
        """No-op for API compatibility with ``asyncio.Queue``.

        anyio memory streams don't track unfinished tasks; callers that
        need ``join()`` semantics should use an :class:`Event` instead.
        """
        pass


class PriorityQueue(Queue[T]):
    """Backend-agnostic priority queue.

    anyio doesn't provide a priority queue, so this implements one using a
    heap guarded by a :class:`Lock` plus an :class:`Event` for wake-ups.
    Items are returned lowest-priority-first, matching
    ``asyncio.PriorityQueue``.
    """

    def __init__(self, maxsize: int = 0) -> None:
        """Initialize the priority queue.

        Args:
            maxsize: Maximum number of items. ``0`` means unbounded.
        """
        # Intentionally do NOT call super().__init__ - we don't use a
        # memory stream for priority ordering.
        self._maxsize = maxsize
        self._heap: list[T] = []
        self._lock = anyio.Lock()
        self._not_empty = anyio.Event()
        self._not_full = anyio.Event()
        self._not_full.set()

    async def put(self, item: T) -> None:
        """Put an item into the queue, blocking if full."""
        while True:
            await self._not_full.wait()
            async with self._lock:
                if self._maxsize > 0 and len(self._heap) >= self._maxsize:
                    # Lost the race; loop and wait again.
                    continue
                heapq.heappush(self._heap, item)
                self._not_empty.set()
                if self._maxsize > 0 and len(self._heap) >= self._maxsize:
                    self._not_full = anyio.Event()
                return

    def put_nowait(self, item: T) -> None:
        """Put an item without blocking.

        Raises:
            anyio.WouldBlock: If the queue is full.
        """
        if self._maxsize > 0 and len(self._heap) >= self._maxsize:
            raise anyio.WouldBlock()
        heapq.heappush(self._heap, item)
        self._not_empty.set()
        if self._maxsize > 0 and len(self._heap) >= self._maxsize:
            self._not_full = anyio.Event()

    async def get(self) -> T:
        """Remove and return the lowest-priority item, blocking if empty."""
        while True:
            await self._not_empty.wait()
            async with self._lock:
                if not self._heap:
                    # Lost the race; loop and wait again.
                    continue
                item = heapq.heappop(self._heap)
                if not self._heap:
                    self._not_empty = anyio.Event()
                self._not_full.set()
                return item

    def get_nowait(self) -> T:
        """Remove and return the lowest-priority item without blocking.

        Raises:
            anyio.WouldBlock: If the queue is empty.
        """
        if not self._heap:
            raise anyio.WouldBlock()
        item = heapq.heappop(self._heap)
        if not self._heap:
            self._not_empty = anyio.Event()
        self._not_full.set()
        return item

    def qsize(self) -> int:
        """Return the number of items in the queue."""
        return len(self._heap)

    def empty(self) -> bool:
        """Return ``True`` if the queue is empty."""
        return not self._heap
