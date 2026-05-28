#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapter for ToolsSchema serialization."""

from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.bus.adapters.base import DeserializeFunc, SerializeFunc, TypeAdapter


class ToolsSchemaAdapter(TypeAdapter):
    """Serialize and deserialize ``ToolsSchema`` instances for network transport."""

    def serialize(self, obj: Any, serialize_value: SerializeFunc) -> dict[str, Any]:
        """Serialize a ``ToolsSchema`` to a JSON-compatible dict.

        Args:
            obj: A ``ToolsSchema`` instance.
            serialize_value: Callback to recursively serialize nested values.

        Returns:
            A dict with a ``standard_tools`` list.
        """
        return {"standard_tools": [tool.to_default_dict() for tool in obj.standard_tools]}

    def deserialize(
        self,
        data: dict[str, Any],
        deserialize_value: DeserializeFunc,
        target_type: type | None = None,
    ) -> Any:
        """Reconstruct a ``ToolsSchema`` from a serialized dict.

        Args:
            data: A dict produced by ``serialize()``.
            deserialize_value: Callback to recursively deserialize nested values.
            target_type: Unused. ``ToolsSchema`` is always the target.

        Returns:
            A new ``ToolsSchema`` instance.
        """
        tools = []
        for item in data["standard_tools"]:
            params = item.get("parameters", {})
            tools.append(
                FunctionSchema(
                    name=item["name"],
                    description=item.get("description", ""),
                    properties=params.get("properties", {}),
                    required=params.get("required", []),
                )
            )
        return ToolsSchema(standard_tools=tools)
