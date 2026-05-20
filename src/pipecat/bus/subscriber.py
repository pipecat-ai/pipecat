#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bus subscriber mixin for receiving messages from an WorkerBus."""

from pipecat.bus.messages import BusMessage


class BusSubscriber:
    """Mixin for objects that receive messages from an `WorkerBus`.

    Implementors override `on_bus_message()` to handle incoming messages.
    Concrete subscribers must provide a ``name`` property (typically
    inherited from ``BaseObject``).
    """

    @property
    def name(self) -> str:
        """Unique name identifying this subscriber on the bus."""
        raise NotImplementedError

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle an incoming bus message.

        Args:
            message: The bus message to handle.
        """
        ...
