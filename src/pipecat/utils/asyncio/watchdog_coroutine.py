#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Watchdog-enabled coroutine wrapper for task monitoring.

This module provides a coroutine wrapper that automatically resets watchdog
timers while waiting for coroutine completion, preventing false positive
watchdog timeouts during legitimate operations.
"""

import asyncio
from typing import Optional

from pipecat.pipeline import task
from pipecat.utils.asyncio.task_manager import BaseTaskManager


class WatchdogCoroutine:
    """Watchdog-enabled coroutine wrapper.

    An asynchronous iterator that monitors activity and resets the current
    task watchdog timer. This is necessary to avoid task watchdog timers to
    expire while we are waiting to get an item from the iterator.
    """

    def __init__(
        self,
        coroutine,
        *,
        manager: BaseTaskManager,
        timeout: float = 2.0,
    ):
        """Initialize the watchdog coroutine wrapper.

        Args:
            coroutine: The coroutine to wrap with watchdog monitoring.
            manager: The task manager for watchdog timer control.
            timeout: Timeout in seconds between watchdog resets while waiting.
        """
        self._coroutine = coroutine
        self._manager = manager
        self._timeout = timeout
        self._current_coro_task: Optional[asyncio.Task] = None

    async def __call__(self):
        """Execute the wrapped coroutine with watchdog monitoring."""
        if self._manager.task_watchdog_enabled:
            return await self._watchdog_call()
        else:
            return await self._coroutine

    async def _watchdog_call(self):
        """Execute coroutine while periodically resetting watchdog timer."""
        while True:
            try:
                if not self._current_coro_task:
                    self._current_coro_task = asyncio.create_task(self._coroutine)

                result = await asyncio.wait_for(
                    asyncio.shield(self._current_coro_task),
                    timeout=self._timeout,
                )

                self._manager.task_reset_watchdog()

                # The task has finished.
                self._current_coro_task = None

                return result
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()


async def watchdog_coroutine(coroutine, *, manager: BaseTaskManager, timeout: float = 2.0):
    """Execute a coroutine with watchdog monitoring support.

    Args:
        coroutine: The coroutine to execute with watchdog monitoring.
        manager: The task manager for watchdog timer control.
        timeout: Timeout in seconds between watchdog resets while waiting.

    Returns:
        The result of the coroutine execution.
    """
    watchdog_coro = WatchdogCoroutine(coroutine, manager=manager, timeout=timeout)
    return await watchdog_coro()
