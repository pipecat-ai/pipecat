#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Watchdog-enabled asyncio PriorityQueue for task monitoring.

This module provides an asyncio PriorityQueue subclass that automatically resets
watchdog timers while waiting for items, preventing false positive watchdog
timeouts during legitimate queue operations.
"""

import asyncio
from dataclasses import dataclass

from loguru import logger

from pipecat.utils.asyncio.task_manager import BaseTaskManager


@dataclass
class _WatchdogPriorityCancelSentinel:
    def __lt__(self, other):
        return True


class WatchdogPriorityQueue(asyncio.PriorityQueue):
    """Watchdog-enabled asyncio PriorityQueue.

    An asynchronous priority queue that resets the current task watchdog
    timer. This is necessary to avoid task watchdog timers to expire while we
    are waiting to get an item from the queue.
    """

    def __init__(
        self,
        manager: BaseTaskManager,
        *,
        maxsize: int = 0,
        timeout: float = 2.0,
    ) -> None:
        """Initialize the watchdog priority queue.

        Args:
            manager: The task manager for watchdog timer control.
            maxsize: Maximum queue size. 0 means unlimited.
            timeout: Timeout in seconds between watchdog resets while waiting.
        """
        super().__init__(maxsize)
        self._manager = manager
        self._timeout = timeout

    async def get(self):
        """Get an item from the queue with watchdog monitoring.

        Returns:
            The next item from the priority queue.
        """
        if self._manager.task_watchdog_enabled:
            get_result = await self._watchdog_get()
        else:
            get_result = await super().get()

        if isinstance(get_result, _WatchdogPriorityCancelSentinel):
            logger.trace(
                "Received WatchdogPriorityCancelSentinel, throwing CancelledError to force cancelling"
            )
            raise asyncio.CancelledError("Cancelling watchdog queue get() call.")
        else:
            return get_result

    def task_done(self):
        """Mark a task as done and reset watchdog if enabled.

        Should be called after processing each item retrieved from the queue.
        """
        if self._manager.task_watchdog_enabled:
            self._manager.task_reset_watchdog()
        super().task_done()

    def cancel(self):
        """Ensures reliable task cancellation by preventing a common race condition.

        The race condition occurs in Python 3.10+ when:
        1. A value is put in the queue just before task cancellation
        2. queue.get() completes before the cancellation signal is delivered
        3. The task misses the CancelledError and continues running indefinitely

        This method prevents the issue by injecting a special sentinel value that
        forces the task to raise CancelledError when consumed, ensuring proper
        task termination.
        """
        super().put_nowait(_WatchdogPriorityCancelSentinel())

    async def _watchdog_get(self):
        """Get item from queue while periodically resetting watchdog timer."""
        while True:
            try:
                item = await asyncio.wait_for(super().get(), timeout=self._timeout)
                self._manager.task_reset_watchdog()
                return item
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()
