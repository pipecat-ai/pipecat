#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Watchdog-enabled async iterator wrapper for task monitoring.

This module provides an async iterator wrapper that automatically resets
watchdog timers while waiting for iterator items, preventing false positive
watchdog timeouts during legitimate waiting periods.
"""

import asyncio
from typing import AsyncIterator, Optional

from pipecat.utils.asyncio.task_manager import BaseTaskManager


class WatchdogAsyncIterator:
    """Watchdog async iterator wrapper.

    An asynchronous iterator that monitors activity and resets the current
    task watchdog timer. This is necessary to avoid task watchdog timers to
    expire while we are waiting to get an item from the iterator.
    """

    def __init__(
        self,
        async_iterable,
        *,
        manager: BaseTaskManager,
        timeout: float = 2.0,
    ):
        """Initialize the watchdog async iterator.

        Args:
            async_iterable: The async iterable to wrap with watchdog monitoring.
            manager: The task manager for watchdog timer control.
            timeout: Timeout in seconds between watchdog resets while waiting.
        """
        self._async_iterable = async_iterable
        self._manager = manager
        self._timeout = timeout
        self._iter: Optional[AsyncIterator] = None
        self._current_anext_task: Optional[asyncio.Task] = None

    def __aiter__(self):
        """Return self as the async iterator.

        Returns:
            This iterator instance.
        """
        return self

    async def __anext__(self):
        """Get the next item from the iterator with watchdog monitoring.

        Returns:
            The next item from the wrapped async iterator.

        Raises:
            StopAsyncIteration: When the iterator is exhausted.
        """
        if not self._iter:
            self._iter = await self._ensure_async_iterator(self._async_iterable)

        if self._manager.task_watchdog_enabled:
            return await self._watchdog_anext()
        else:
            return await self._iter.__anext__()

    async def _watchdog_anext(self):
        """Get next item while periodically resetting watchdog timer."""
        while True:
            try:
                if not self._current_anext_task:
                    self._current_anext_task = asyncio.create_task(self._iter.__anext__())

                item = await asyncio.wait_for(
                    asyncio.shield(self._current_anext_task),
                    timeout=self._timeout,
                )

                self._manager.task_reset_watchdog()

                # The task has finished, so we will create a new one for the
                # next item.
                self._current_anext_task = None

                return item
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()
            except StopAsyncIteration:
                self._current_anext_task = None
                raise

    async def _ensure_async_iterator(self, obj) -> AsyncIterator:
        """Ensure the object is an async iterator, awaiting if necessary."""
        aiter = obj.__aiter__()
        if asyncio.iscoroutine(aiter):
            aiter = await aiter
        return aiter
