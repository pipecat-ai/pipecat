#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Abstract base class for type adapters used by message serializers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

SerializeFunc = Callable[[Any], Any]
DeserializeFunc = Callable[[Any], Any]


class TypeAdapter(ABC):
    """Serialize and deserialize instances of a specific type for network transport.

    Each adapter handles one or more types, converting them to/from a
    JSON-compatible dict. Register adapters on a ``JSONMessageSerializer``
    to handle non-JSON-native field values (e.g. ``LLMContext``, ``ToolsSchema``).

    Adapters receive ``serialize_value`` and ``deserialize_value`` callbacks
    from the serializer so they can recursively serialize nested fields
    without importing the serializer itself.
    """

    @abstractmethod
    def serialize(self, obj: Any, serialize_value: SerializeFunc) -> dict[str, Any]:
        """Convert an object to a JSON-compatible dict.

        Args:
            obj: The object to serialize.
            serialize_value: Callback to recursively serialize nested values.

        Returns:
            A dict representation of the object.
        """
        pass

    @abstractmethod
    def deserialize(
        self,
        data: dict[str, Any],
        deserialize_value: DeserializeFunc,
        target_type: type | None = None,
    ) -> Any:
        """Reconstruct an object from a dict.

        Args:
            data: The dict representation produced by ``serialize()``.
            deserialize_value: Callback to recursively deserialize nested values.
            target_type: The resolved target class. Defaults to None.

        Returns:
            The reconstructed object.
        """
        pass
