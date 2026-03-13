#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event-based notifier implementation using asyncio Condition primitives."""

import asyncio

from pipecat.utils.sync.base_notifier import BaseNotifier


class EventNotifier(BaseNotifier):
    """Event-based notifier using asyncio.Condition for task synchronization.

    Provides a simple notification mechanism where one task can signal
    an event and other tasks can wait for that event to occur. Uses
    asyncio.Condition to ensure atomic check-and-reset, preventing
    lost notifications in multi-consumer scenarios.
    """

    def __init__(self):
        """Initialize the event notifier.

        Creates an internal asyncio.Condition and boolean flag for
        managing notifications.
        """
        self._condition = asyncio.Condition()
        self._notified = False

    async def notify(self):
        """Signal the event to notify waiting tasks.

        Sets the notification flag and wakes all waiting tasks.
        Only one waiter will consume each notification.
        """
        async with self._condition:
            self._notified = True
            self._condition.notify_all()

    async def wait(self):
        """Wait for the event to be signaled.

        Blocks until another task calls notify(). Atomically checks
        and resets the notification flag under the condition lock,
        preventing races between multiple consumers.
        """
        async with self._condition:
            await self._condition.wait_for(lambda: self._notified)
            self._notified = False
