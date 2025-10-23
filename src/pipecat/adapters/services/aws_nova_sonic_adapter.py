#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Nova Sonic LLM adapter for Pipecat."""

import copy
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage


class Role(Enum):
    """Roles supported in AWS Nova Sonic conversations.

    Parameters:
        SYSTEM: System-level messages (not used in conversation history).
        USER: Messages sent by the user.
        ASSISTANT: Messages sent by the assistant.
        TOOL: Messages sent by tools (not used in conversation history).
    """

    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"


@dataclass
class AWSNovaSonicConversationHistoryMessage:
    """A single message in AWS Nova Sonic conversation history.

    Parameters:
        role: The role of the message sender (USER or ASSISTANT only).
        text: The text content of the message.
    """

    role: Role  # only USER and ASSISTANT
    text: str


class AWSNovaSonicLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking AWS Nova Sonic LLM API.

    This is a placeholder until support for universal LLMContext machinery is added for AWS Nova Sonic.
    """

    system_instruction: Optional[str]
    messages: List[AWSNovaSonicConversationHistoryMessage]
    tools: List[Dict[str, Any]]


class AWSNovaSonicLLMAdapter(BaseLLMAdapter[AWSNovaSonicLLMInvocationParams]):
    """Adapter for AWS Nova Sonic language models.

    Converts Pipecat's standard function schemas into AWS Nova Sonic's
    specific function-calling format, enabling tool use with Nova Sonic models.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for AWS Nova Sonic."""
        return "aws-nova-sonic"

    def get_llm_invocation_params(self, context: LLMContext) -> AWSNovaSonicLLMInvocationParams:
        """Get AWS Nova Sonic-specific LLM invocation parameters from a universal LLM context.

        This is a placeholder until support for universal LLMContext machinery is added for AWS Nova Sonic.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for invoking AWS Nova Sonic's LLM API.
        """
        messages = self._from_universal_context_messages(self.get_messages(context))
        return {
            "system_instruction": messages.system_instruction,
            "messages": messages.messages,
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools) or [],
        }

    def get_messages_for_logging(self, context) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about AWS Nova Sonic.

        Removes or truncates sensitive data like image content for safe logging.

        This is a placeholder until support for universal LLMContext machinery is added for AWS Nova Sonic.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about AWS Nova Sonic.
        """
        return self._from_universal_context_messages(self.get_messages(context)).messages

    @dataclass
    class ConvertedMessages:
        """Container for Google-formatted messages converted from universal context."""

        messages: List[AWSNovaSonicConversationHistoryMessage]
        system_instruction: Optional[str] = None

    def _from_universal_context_messages(
        self, universal_context_messages: List[LLMContextMessage]
    ) -> ConvertedMessages:
        system_instruction = None
        messages = []

        # Bail if there are no messages
        if not universal_context_messages:
            return self.ConvertedMessages()

        universal_context_messages = copy.deepcopy(universal_context_messages)

        # If we have a "system" message as our first message, let's pull that out into "instruction"
        if universal_context_messages[0].get("role") == "system":
            system = universal_context_messages.pop(0)
            content = system.get("content")
            if isinstance(content, str):
                system_instruction = content
            elif isinstance(content, list):
                system_instruction = content[0].get("text")
            if system_instruction:
                self._system_instruction = system_instruction

        # Process remaining messages to fill out conversation history.
        # Nova Sonic supports "user" and "assistant" messages in history.
        for universal_context_message in universal_context_messages:
            message = self._from_universal_context_message(universal_context_message)
            if message:
                messages.append(message)

        return self.ConvertedMessages(messages=messages, system_instruction=system_instruction)

    def _from_universal_context_message(self, message) -> AWSNovaSonicConversationHistoryMessage:
        """Convert standard message format to Nova Sonic format.

        Args:
            message: Standard message dictionary to convert.

        Returns:
            Nova Sonic conversation history message, or None if not convertible.
        """
        role = message.get("role")
        if message.get("role") == "user" or message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(message.get("content"), list):
                content = ""
                for c in message.get("content"):
                    if c.get("type") == "text":
                        content += " " + c.get("text")
                    else:
                        logger.error(
                            f"Unhandled content type in context message: {c.get('type')} - {message}"
                        )
            # There won't be content if this is an assistant tool call entry.
            # We're ignoring those since they can't be loaded into AWS Nova Sonic conversation
            # history
            if content:
                return AWSNovaSonicConversationHistoryMessage(role=Role[role.upper()], text=content)
        # NOTE: we're ignoring messages with role "tool" since they can't be loaded into AWS Nova
        # Sonic conversation history

    @staticmethod
    def _to_aws_nova_sonic_function_format(function: FunctionSchema) -> Dict[str, Any]:
        """Convert a function schema to AWS Nova Sonic format.

        Args:
            function: The function schema to convert.

        Returns:
            Dictionary in AWS Nova Sonic function format with toolSpec structure.
        """
        return {
            "toolSpec": {
                "name": function.name,
                "description": function.description,
                "inputSchema": {
                    "json": json.dumps(
                        {
                            "type": "object",
                            "properties": function.properties,
                            "required": function.required,
                        }
                    )
                },
            }
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert tools schema to AWS Nova Sonic function-calling format.

        Args:
            tools_schema: The tools schema containing function definitions to convert.

        Returns:
            List of dictionaries in AWS Nova Sonic function format.
        """
        functions_schema = tools_schema.standard_tools
        standard_tools = [
            self._to_aws_nova_sonic_function_format(func) for func in functions_schema
        ]

        # For backward compatibility, AWS Nova Sonic can still be used with
        # tools in dict format, even though it always uses `LLMContext` under
        # the hood (via `LLMContext.from_openai_context()`).
        # To support this behavior, we use "shimmed" custom tools here.
        # (We maintain this backward compatibility because users aren't
        # *knowingly* opting into the new `LLMContext`.)
        shimmed_tools = []
        if tools_schema.custom_tools:
            shimmed_tools = tools_schema.custom_tools.get(AdapterType.SHIM, [])

        return standard_tools + shimmed_tools
