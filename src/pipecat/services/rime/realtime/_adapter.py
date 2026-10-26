#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Universal text history and tool conversion for Rime Prism."""

from typing import Any

from rimelabs_sdk import realtime as r

from pipecat.adapters.llm.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.types import is_given


def _text(value: Any, *, history_message: bool = True) -> str:
    if not isinstance(value, str):
        if not isinstance(value, list) or any(
            not isinstance(part, dict) or part.get("type") != "text" for part in value
        ):
            raise ValueError("Prism history supports text content only")
        value = "".join(part["text"] for part in value)
    if history_message and len(value) > 4000:
        raise ValueError("Prism history messages must contain at most 4000 characters")
    value.encode("utf-8")
    return value


def history_items(messages: list) -> list[r.HistoryItem]:
    """Validate a complete history, including completed, ordered tool pairs."""
    result = []
    calls: set[str] = set()
    pending: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Prism does not support opaque provider history")
        role = message.get("role")
        if role in ("user", "assistant", "system"):
            if message.get("content") is not None and (
                message.get("content") or not message.get("tool_calls")
            ):
                result.append(r.HistoryMessage(role=role, text=_text(message["content"])))
            elif not message.get("tool_calls"):
                raise ValueError("Prism history messages require text")
            for call in message.get("tool_calls", []):
                if role != "assistant" or call.get("type", "function") != "function":
                    raise ValueError("Prism supports assistant function calls only")
                call_id = call["id"]
                if not isinstance(call_id, str) or not call_id or call_id in calls:
                    raise ValueError("Imported calls require unique call IDs")
                function = call["function"]
                arguments = function.get("arguments", "{}")
                name = function["name"]
                if not isinstance(name, str) or not name or not isinstance(arguments, str):
                    raise ValueError("Imported calls require a name and string arguments")
                for value in (call_id, name, arguments):
                    value.encode("utf-8")
                result.append(
                    r.HistoryFunctionCall(call_id=call_id, name=name, arguments=arguments)
                )
                calls.add(call_id)
                pending.add(call_id)
        elif role == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError("Imported results must follow their calls exactly once")
            pending.remove(message["tool_call_id"])
            result.append(
                r.HistoryFunctionResult(
                    call_id=message["tool_call_id"],
                    output=_text(message["content"], history_message=False),
                )
            )
        else:
            raise ValueError(f"Prism does not support history role {role!r}")
    if pending:
        raise ValueError("Cannot import pending tool calls into Prism")
    return result


class RimeRealtimeLLMAdapter(BaseLLMAdapter):
    """Convert universal context without submitting live turns or executing tools."""

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Provider identity for context filtering."""
        return "rime-realtime"

    def get_messages_for_logging(self, context: LLMContext) -> list:
        """Return text history with large values shortened."""
        return self.get_messages(context, truncate_large_values=True)

    async def get_llm_invocation_params(self, context: LLMContext, **kwargs) -> dict:
        """Return validated history and the advertised tools."""
        if is_given(context.tool_choice) and context.tool_choice != "auto":
            raise ValueError("Prism only supports automatic tool selection")
        return {
            "messages": history_items(context.get_messages()),
            "tools": self.tools(context.tools),
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[r.ToolDefinition]:
        """Convert standard function definitions to SDK values."""
        if tools_schema.custom_tools:
            raise ValueError("Prism does not support provider-native tool overrides")
        return [r.ToolDefinition(**tool.to_default_dict()) for tool in tools_schema.standard_tools]

    def tools(self, tools) -> list[r.ToolDefinition]:
        """Normalize tools and include the service's built-in definitions."""
        tools = LLMContext._normalize_and_validate_tools(tools if tools is not None else [])
        values = self.to_provider_tools_format(tools) if is_given(tools) else []
        names = {value.name for value in values}
        values.extend(
            r.ToolDefinition(**tool.to_default_dict())
            for name, tool in self.builtin_tools.items()
            if name not in names
        )
        if len({value.name for value in values}) != len(values):
            raise ValueError("Prism tool names must be unique")
        return values
