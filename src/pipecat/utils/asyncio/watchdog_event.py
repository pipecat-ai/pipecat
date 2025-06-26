#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

from pipecat.utils.asyncio.task_manager import BaseTaskManager


class WatchdogEvent(asyncio.Event):
    """An asynchronous event that resets the current task watchdog timer. This
    is necessary to avoid task watchdog timers to expire while we are waiting on
    the event.

    """

    def __init__(
        self,
        manager: BaseTaskManager,
        *,
        timeout: float = 2.0,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._timeout = timeout

    async def wait(self):
        if self._manager.task_watchdog_enabled:
            return await self._watchdog_wait()
        else:
            return await super().wait()

    async def _watchdog_wait(self):
        while True:
            try:
                await asyncio.wait_for(super().wait(), timeout=self._timeout)
                self._manager.task_reset_watchdog()
                return True
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()
