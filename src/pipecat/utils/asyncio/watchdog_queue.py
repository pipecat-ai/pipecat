#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass

from loguru import logger

from pipecat.utils.asyncio.task_manager import BaseTaskManager


@dataclass
class WatchdogQueueCancelSentinel:
    pass


class WatchdogQueue(asyncio.Queue):
    """An asynchronous queue that resets the current task watchdog timer. This
    is necessary to avoid task watchdog timers to expire while we are waiting to
    get an item from the queue.

    """

    def __init__(
        self,
        manager: BaseTaskManager,
        *,
        maxsize: int = 0,
        timeout: float = 2.0,
    ) -> None:
        super().__init__(maxsize)
        self._manager = manager
        self._timeout = timeout

    async def get(self):
        if self._manager.task_watchdog_enabled:
            get_result = await self._watchdog_get()
        else:
            get_result = await super().get()

        if isinstance(get_result, WatchdogQueueCancelSentinel):
            logger.debug(
                "Received WatchdogQueueCancelFrame, throwing CancelledError to force cancelling"
            )
            raise asyncio.CancelledError("Cancelling watchdog queue get() call.")
        else:
            return get_result

    def task_done(self):
        if self._manager.task_watchdog_enabled:
            self._manager.task_reset_watchdog()
        super().task_done()

    def cancel(self):
        super().put_nowait(WatchdogQueueCancelSentinel())

    async def _watchdog_get(self):
        while True:
            try:
                item = await asyncio.wait_for(super().get(), timeout=self._timeout)
                self._manager.task_reset_watchdog()
                return item
            except asyncio.TimeoutError:
                self._manager.task_reset_watchdog()
