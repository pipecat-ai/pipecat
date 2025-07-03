#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event-based notifier implementation using asyncio Event primitives."""

import asyncio

from pipecat.sync.base_notifier import BaseNotifier


class EventNotifier(BaseNotifier):
    """Event-based notifier using asyncio.Event for task synchronization.

    Provides a simple notification mechanism where one task can signal
    an event and other tasks can wait for that event to occur. The event
    is automatically cleared after each wait operation.
    """

    def __init__(self):
        """Initialize the event notifier.

        Creates an internal asyncio.Event for managing notifications.
        """
        self._event = asyncio.Event()

    async def notify(self):
        """Signal the event to notify waiting tasks.

        Sets the internal event, causing any tasks waiting on this
        notifier to be awakened.
        """
        self._event.set()

    async def wait(self):
        """Wait for the event to be signaled.

        Blocks until another task calls notify(). Automatically clears
        the event after being awakened so subsequent calls will wait
        for the next notification.
        """
        await self._event.wait()
        self._event.clear()
