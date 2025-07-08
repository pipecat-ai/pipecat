#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Watchdog-enabled asyncio Event for task monitoring.

This module provides an asyncio Event subclass that automatically resets
watchdog timers while waiting for the event, preventing false positive
watchdog timeouts during legitimate waiting periods.
"""

import asyncio

from pipecat.utils.asyncio.task_manager import BaseTaskManager


class WatchdogEvent(asyncio.Event):
    """Watchdog-enabled asyncio Event.

    An asynchronous event that resets the current task watchdog timer. This
    is necessary to avoid task watchdog timers to expire while we are waiting on
    the event.
    """

    def __init__(
        self,
        manager: BaseTaskManager,
        *,
        timeout: float = 2.0,
    ) -> None:
        """Initialize the watchdog event.

        Args:
            manager: The task manager for watchdog timer control.
            timeout: Timeout in seconds between watchdog resets while waiting.
        """
        super().__init__()
        self._manager = manager
        self._timeout = timeout

    async def wait(self):
        """Wait for the event to be set with watchdog monitoring.

        Returns:
            True when the event is set.
        """
        if self._manager.task_watchdog_enabled:
            return await self._watchdog_wait()
        else:
            return await super().wait()

    async def _watchdog_wait(self):
        """Wait for event while periodically resetting watchdog timer."""
        while True:
            try:
                await asyncio.wait_for(super().wait(), timeout=self._timeout)
                self._manager.task_reset_watchdog()
                return True
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()

    def clear(self):
        """Clear the event while resetting watchdog timer."""
        if self._manager.task_watchdog_enabled:
            self._manager.task_reset_watchdog()
        super().clear()
