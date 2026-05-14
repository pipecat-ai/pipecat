#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Priority queue for bus messages."""

import asyncio
from typing import Any

from pipecat.frames.frames import SystemFrame

HIGH_PRIORITY = 1
LOW_PRIORITY = 2


class BusMessageQueue(asyncio.PriorityQueue):
    """Priority queue that delivers system messages before normal messages.

    Messages that extend ``SystemFrame`` (e.g. cancel messages) get high
    priority. All other messages are delivered in FIFO order at normal
    priority.
    """

    def __init__(self):
        """Initialize the BusMessageQueue."""
        super().__init__()
        self._high_counter = 0
        self._low_counter = 0

    def put_nowait(self, item) -> None:
        """Add a message to the queue with automatic priority assignment.

        Args:
            item: The bus message to enqueue.
        """
        if isinstance(item, SystemFrame):
            self._high_counter += 1
            super().put_nowait((HIGH_PRIORITY, self._high_counter, item))
        else:
            self._low_counter += 1
            super().put_nowait((LOW_PRIORITY, self._low_counter, item))

    async def put(self, item) -> None:
        """Add a message to the queue with automatic priority assignment.

        Args:
            item: The bus message to enqueue.
        """
        if isinstance(item, SystemFrame):
            self._high_counter += 1
            await super().put((HIGH_PRIORITY, self._high_counter, item))
        else:
            self._low_counter += 1
            await super().put((LOW_PRIORITY, self._low_counter, item))

    async def get(self) -> Any:
        """Get the next message, with system messages prioritized."""
        _, _, message = await super().get()
        return message
