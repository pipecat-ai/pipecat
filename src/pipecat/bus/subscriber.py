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

    def accepts_bus_message(self, message: BusMessage) -> bool:
        """Whether this subscriber should be handed this message.

        Checked by the bus before every delivery. Returning False drops
        the message for this subscriber alone; others still receive it.
        Subscribers that take everything, which is the default, need not
        override this.

        Args:
            message: The bus message about to be delivered.

        Returns:
            Whether to deliver the message.
        """
        return True

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle an incoming bus message.

        Only called for messages :meth:`accepts_bus_message` allowed.

        Args:
            message: The bus message to handle.
        """
        ...
