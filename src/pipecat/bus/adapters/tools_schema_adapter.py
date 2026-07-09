#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapter for ToolsSchema serialization."""

from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.bus.adapters.base import DeserializeFunc, SerializeFunc, TypeAdapter


class ToolsSchemaAdapter(TypeAdapter):
    """Serialize and deserialize ``ToolsSchema`` instances for network transport."""

    def serialize(self, obj: Any, serialize_value: SerializeFunc) -> dict[str, Any]:
        """Serialize a ``ToolsSchema`` to a JSON-compatible dict.

        Args:
            obj: A ``ToolsSchema`` instance.
            serialize_value: Callback to recursively serialize nested values.

        Returns:
            A dict with a ``standard_tools`` list and, if present, a
            ``custom_tools`` mapping.
        """
        result: dict[str, Any] = {
            "standard_tools": [tool.to_default_dict() for tool in obj.standard_tools]
        }
        if obj.custom_tools:
            result["custom_tools"] = {
                adapter_type.value: tools for adapter_type, tools in obj.custom_tools.items()
            }
        return result

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
            A new ``ToolsSchema`` instance, including ``custom_tools`` if
            present in ``data``.
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
        custom_tools = None
        if "custom_tools" in data:
            custom_tools = {AdapterType(key): value for key, value in data["custom_tools"].items()}
        return ToolsSchema(standard_tools=tools, custom_tools=custom_tools)
