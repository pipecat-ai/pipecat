#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type adapter for LLMContext serialization."""

from typing import Any

from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.bus.adapters.base import DeserializeFunc, SerializeFunc, TypeAdapter
from pipecat.bus.adapters.tools_schema_adapter import ToolsSchemaAdapter
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    NotGiven,
)


class LLMContextAdapter(TypeAdapter):
    """Serialize and deserialize ``LLMContext`` instances.

    The ``NOT_GIVEN`` sentinel is preserved across serialization: missing
    keys are restored as ``NOT_GIVEN`` on deserialization.
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._tools_schema_adapter = ToolsSchemaAdapter()

    def serialize(self, obj: Any, serialize_value: SerializeFunc) -> dict[str, Any]:
        """Serialize an ``LLMContext`` to a JSON-compatible dict.

        Args:
            obj: An ``LLMContext`` instance.
            serialize_value: Callback to recursively serialize nested values.

        Returns:
            A dict with ``messages`` and, optionally, ``tools`` and
            ``tool_choice`` keys.
        """
        result: dict[str, Any] = {
            "messages": [self._serialize_message(m, serialize_value) for m in obj.messages],
        }
        if not isinstance(obj.tools, NotGiven):
            result["tools"] = self._tools_schema_adapter.serialize(obj.tools, serialize_value)
        if not isinstance(obj.tool_choice, NotGiven):
            result["tool_choice"] = serialize_value(obj.tool_choice)
        return result

    def deserialize(
        self,
        data: dict[str, Any],
        deserialize_value: DeserializeFunc,
        target_type: type | None = None,
    ) -> Any:
        """Reconstruct an ``LLMContext`` from a serialized dict.

        Missing ``tools`` and ``tool_choice`` keys are restored as
        OpenAI's ``NOT_GIVEN`` sentinel.

        Args:
            data: A dict produced by ``serialize()``.
            deserialize_value: Callback to recursively deserialize nested values.
            target_type: Unused. ``LLMContext`` is always the target.

        Returns:
            A new ``LLMContext`` instance.
        """
        messages = [self._deserialize_message(m, deserialize_value) for m in data["messages"]]
        tools = (
            self._tools_schema_adapter.deserialize(data["tools"], deserialize_value)
            if "tools" in data
            else OPENAI_NOT_GIVEN
        )
        tool_choice = (
            deserialize_value(data["tool_choice"]) if "tool_choice" in data else OPENAI_NOT_GIVEN
        )
        return LLMContext(messages=messages, tools=tools, tool_choice=tool_choice)

    def _serialize_message(self, msg: Any, serialize_value: SerializeFunc) -> dict[str, Any]:
        if isinstance(msg, LLMSpecificMessage):
            return {
                "__specific__": True,
                "llm": msg.llm,
                "message": serialize_value(msg.message),
            }
        return serialize_value(msg)

    def _deserialize_message(self, data: Any, deserialize_value: DeserializeFunc) -> Any:
        if isinstance(data, dict) and data.get("__specific__"):
            return LLMSpecificMessage(
                llm=data["llm"],
                message=deserialize_value(data["message"]),
            )
        return deserialize_value(data)
