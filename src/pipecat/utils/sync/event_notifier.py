#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Event-based notifier implementation using asyncio Event primitives."""

import asyncio

from pipecat.utils.sync.base_notifier import BaseNotifier


class EventNotifier(BaseNotifier):
    """Event-based notifier using asyncio.Event for task synchronization.

    Provides a notification mechanism where one task can signal an event and
    waiting tasks can receive it. The event is automatically cleared after
    each wait operation (auto-reset semantics).

    Safety:
        The wait-then-clear pattern is safe in single-threaded asyncio for
        single-consumer repeated cycles: there is no yield point between
        ``event.wait()`` returning and ``event.clear()`` executing.

    Multi-consumer one-shot usage:
        For one-shot scenarios (e.g., voicemail gating where ``notify()`` is
        called exactly once), ``set()`` resolves all currently waiting futures
        before any consumer can ``clear()``, so all consumers wake correctly.

    Limitations:
        Multi-consumer repeated-notify cycles are NOT safe: if waiter A
        clears the event before waiter B runs, a subsequent ``notify()``
        between A's ``clear()`` and B's ``clear()`` can be lost. Current
        codebase usages are either single-consumer repeated or multi-consumer
        one-shot, which are both safe.
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

        Note:
            Safe in single-threaded asyncio: no yield point exists between
            the internal ``event.wait()`` return and ``event.clear()``.
        """
        await self._event.wait()
        self._event.clear()
