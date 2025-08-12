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
class WatchdogPriorityCancelSentinel:
    """Sentinel object used in priority queues to force cancellation.

    An instance of this class is typically inserted into a
    `WatchdogPriorityQueue` to act as a high-priority marker asyncio task
    cancellation.

    """

    pass


class WatchdogPriorityQueue(asyncio.PriorityQueue):
    """Class for watchdog-enabled asyncio PriorityQueue.

    An asynchronous priority queue that resets the current task watchdog
    timer. This is necessary to avoid task watchdog timers to expire while we
    are waiting to get an item from the queue.

    This queue expects items to be tuples, with the actual payload stored
    in the last element. All preceding elements are treated as numeric
    priority fields. For example:

        (0, 1, "foo")

    The tuple length must be specified at creation time so the queue can
    correctly construct special items, such as the watchdog cancel sentinel,
    with the proper tuple structure.

    """

    def __init__(
        self,
        manager: BaseTaskManager,
        *,
        tuple_size: int,
        maxsize: int = 0,
        timeout: float = 2.0,
    ) -> None:
        """Initialize the watchdog priority queue.

        Args:
            manager: The task manager for watchdog timer control.
            tuple_size: The number of values in each inserted tuple.
            maxsize: Maximum queue size. 0 means unlimited.
            timeout: Timeout in seconds between watchdog resets while waiting.
        """
        super().__init__(maxsize)
        self._manager = manager
        self._timeout = timeout
        self._tuple_size = tuple_size

    async def get(self):
        """Get an item from the queue with watchdog monitoring.

        Returns:
            The next item from the priority queue.
        """
        if self._manager.task_watchdog_enabled:
            get_result = await self._watchdog_get()
        else:
            get_result = await super().get()

        # Value is always at the end of the tuple.
        item = get_result[-1]

        if isinstance(item, WatchdogPriorityCancelSentinel):
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
        item = [float("-inf")] * self._tuple_size
        # Values go always at the end.
        item[-1] = WatchdogPriorityCancelSentinel()
        super().put_nowait(tuple(item))

    async def _watchdog_get(self):
        """Get item from queue while periodically resetting watchdog timer."""
        while True:
            try:
                item = await asyncio.wait_for(super().get(), timeout=self._timeout)
                self._manager.task_reset_watchdog()
                return item
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()
