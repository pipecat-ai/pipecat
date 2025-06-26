#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import AsyncIterator, Optional

from pipecat.utils.asyncio.task_manager import BaseTaskManager


class WatchdogAsyncIterator:
    """An asynchronous iterator that monitors activity and resets the current
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
        self._async_iterable = async_iterable
        self._manager = manager
        self._timeout = timeout
        self._iter: Optional[AsyncIterator] = None
        self._current_anext_task: Optional[asyncio.Task] = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._iter:
            self._iter = await self._ensure_async_iterator(self._async_iterable)

        if self._manager.task_watchdog_enabled:
            return await self._watchdog_anext()
        else:
            return await self._iter.__anext__()

    async def _watchdog_anext(self):
        while True:
            try:
                if not self._current_anext_task:
                    self._current_anext_task = asyncio.create_task(self._iter.__anext__())

                item = await asyncio.wait_for(
                    asyncio.shield(self._current_anext_task),
                    timeout=self._timeout,
                )

                self._manager.task_reset_watchdog()

                # The task has finish, so we will create a new one for th next item.
                self._current_anext_task = None

                return item
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()
            except StopAsyncIteration:
                self._current_anext_task = None
                raise

    async def _ensure_async_iterator(self, obj) -> AsyncIterator:
        aiter = obj.__aiter__()
        if asyncio.iscoroutine(aiter):
            aiter = await aiter
        return aiter
