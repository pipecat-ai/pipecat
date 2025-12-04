#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base notifier interface for Pipecat."""

from abc import ABC, abstractmethod


class BaseNotifier(ABC):
    """Abstract base class for notification mechanisms.

    Provides a standard interface for implementing notification and waiting
    patterns used for event coordination and signaling between components
    in the Pipecat framework.
    """

    @abstractmethod
    async def notify(self):
        """Send a notification signal.

        Implementations should trigger any waiting coroutines or processes
        that are blocked on this notifier.
        """
        pass

    @abstractmethod
    async def wait(self):
        """Wait for a notification signal.

        Implementations should block until a notification is received
        from the corresponding notify() call.
        """
        pass
