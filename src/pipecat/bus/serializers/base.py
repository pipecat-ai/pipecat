#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base classes for bus message serialization."""

from abc import ABC, abstractmethod

from pipecat.bus.messages import BusMessage


class MessageSerializer(ABC):
    """Serialize and deserialize `BusMessage` instances for network transport.

    Network bus implementations use a `MessageSerializer` to convert messages
    to bytes for transmission and reconstruct them on the receiving end.
    """

    @abstractmethod
    def serialize(self, message: BusMessage) -> bytes:
        """Convert a bus message to bytes.

        Args:
            message: The bus message to serialize.

        Returns:
            The serialized bytes.
        """
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> BusMessage | None:
        """Reconstruct a bus message from bytes.

        Args:
            data: The serialized bytes produced by `serialize()`.

        Returns:
            The reconstructed `BusMessage`.
        """
        pass
