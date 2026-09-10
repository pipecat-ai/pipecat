#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live LLM adapter for Pipecat."""

from typing import Any, TypedDict, cast

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.live import events
from pipecat.utils.types import NotGiven, is_given


class OpenAILiveLLMInvocationParams(TypedDict):
    """Session configuration derived from a universal ``LLMContext``.

    Parameters:
        instructions: System instructions for the live model, or ``None``.
        input: Prior text-only messages to seed the session with.
        tools: Function tools in Responses API format, for the backend model.
        tool_choice: Tool choice in Responses API format, or ``None``.
    """

    instructions: str | None
    input: list[events.InputItem]
    tools: list[dict[str, Any]]
    tool_choice: Any | None


class OpenAILiveLLMAdapter(BaseLLMAdapter[OpenAILiveLLMInvocationParams]):
    """Converts a universal ``LLMContext`` into OpenAI Live session configuration.

    The live model takes a system instruction and a text-only conversation
    history at session start; tools belong to the backend (Responses) model
    and use the Responses API tool format.
    """

    def __init__(self):
        """Initialize the adapter."""
        super().__init__()
        self._responses_adapter = OpenAIResponsesLLMAdapter()
        self._warned_input_truncated = False

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for OpenAI Live."""
        return "openai-live"

    def get_llm_invocation_params(
        self, context: LLMContext, *, system_instruction: str | None = None
    ) -> OpenAILiveLLMInvocationParams:
        """Derive session configuration from a universal LLM context.

        A leading ``system`` message becomes the live model's instructions
        (``system_instruction`` from the service settings wins if both are
        set). The remaining messages become the startup ``input`` history.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: System instruction from the service settings.

        Returns:
            The session configuration values.
        """
        # LLMSpecificMessages are opaque provider payloads with no place in a
        # text-only history.
        messages = [
            cast(dict[str, Any], m) for m in self.get_messages(context) if isinstance(m, dict)
        ]

        system_from_context = None
        if messages and messages[0].get("role") == "system":
            system_from_context = self._text_content(messages.pop(0))
        instructions = self._resolve_system_instruction(
            system_from_context, system_instruction, discard_context_system=True
        )

        return {
            "instructions": instructions,
            "input": self._to_input_items(messages),
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools) or [],
            "tool_choice": self._to_tool_choice(context.tool_choice),
        }

    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging.

        Binary data (images, audio) is replaced with short placeholders.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging.
        """
        return cast(list[dict[str, Any]], self.get_messages(context, truncate_large_values=True))

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[dict[str, Any]]:
        """Convert tool schemas to the Responses API tool format used by the backend model.

        The Live session schema rejects the Responses-only ``strict`` field, so
        it is dropped from function tools.

        Args:
            tools_schema: The tools schema containing functions to convert.

        Returns:
            List of tool definitions.
        """
        tools = []
        for tool in self._responses_adapter.to_provider_tools_format(tools_schema):
            tool = dict(tool)
            if tool.get("type") == "function":
                tool.pop("strict", None)
            tools.append(tool)
        return tools

    def _to_input_items(self, messages: list[dict[str, Any]]) -> list[events.InputItem]:
        """Convert standard messages to the text-only startup ``input`` history.

        Tool calls, tool results and non-text content have no representation
        there and are skipped. System messages become developer messages, the
        role the startup history accepts. At most
        :data:`events.MAX_INPUT_ITEMS` items are kept, dropping the oldest.
        """
        items: list[events.InputItem] = []
        for message in messages:
            role = message.get("role")
            text = self._text_content(message)
            if role in ("system", "developer") and text:
                items.append(
                    events.InputItem(role="developer", content=[events.InputTextContent(text=text)])
                )
            elif role == "user" and text:
                items.append(
                    events.InputItem(role="user", content=[events.InputTextContent(text=text)])
                )
            elif role == "assistant" and text and not message.get("tool_calls"):
                items.append(
                    events.InputItem(
                        role="assistant", content=[events.OutputTextContent(text=text)]
                    )
                )
            else:
                logger.debug(
                    f"Skipping context message with no startup-history representation: role={role!r}"
                )

        if len(items) > events.MAX_INPUT_ITEMS:
            if not self._warned_input_truncated:
                self._warned_input_truncated = True
                logger.warning(
                    f"Context has {len(items)} text messages but the OpenAI Live API accepts "
                    f"at most {events.MAX_INPUT_ITEMS} startup messages; keeping the most recent."
                )
            items = items[-events.MAX_INPUT_ITEMS :]
        return items

    @staticmethod
    def _text_content(message: dict[str, Any]) -> str:
        """Return a message's text content, joining text parts of list content."""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
            )
        return ""

    @staticmethod
    def _to_tool_choice(tool_choice: Any | NotGiven) -> Any | None:
        """Convert a context tool choice to the Responses API shape.

        The context stores the chat-completions form, whose named-function
        variant nests the name (``{"type": "function", "function": {"name": …}}``);
        the Responses API takes it flat (``{"type": "function", "name": …}``).
        """
        if not is_given(tool_choice) or tool_choice is None:
            return None
        if isinstance(tool_choice, dict) and isinstance(tool_choice.get("function"), dict):
            return {"type": "function", "name": tool_choice["function"].get("name")}
        return tool_choice
