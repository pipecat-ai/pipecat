#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini LLM adapter for Pipecat."""

import base64
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from loguru import logger
from openai import NotGiven

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMContextMessage,
    LLMSpecificMessage,
    LLMStandardMessage,
)

try:
    from google.genai.types import (
        Blob,
        Content,
        FunctionCall,
        FunctionResponse,
        Part,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


class GeminiLLMInvocationParams(TypedDict):
    """Context-based parameters for invoking Gemini LLM."""

    system_instruction: Optional[str]
    messages: List[Content]
    tools: List[Any] | NotGiven


class GeminiLLMAdapter(BaseLLMAdapter[GeminiLLMInvocationParams]):
    """Gemini-specific adapter for Pipecat.

    Handles:
    - Extracting parameters for Gemini's API from a universal LLM context
    - Converting Pipecat's standardized tools schema to Gemini's function-calling format.
    - Extracting and sanitizing messages from the LLM context for logging with Gemini.
    """

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Get the identifier used in LLMSpecificMessage instances for Google."""
        return "google"

    def get_llm_invocation_params(self, context: LLMContext) -> GeminiLLMInvocationParams:
        """Get Gemini-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.

        Returns:
            Dictionary of parameters for Gemini's API.
        """
        messages = self._from_universal_context_messages(self.get_messages(context))
        return {
            "system_instruction": messages.system_instruction,
            "messages": messages.messages,
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools),
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> List[Dict[str, Any]]:
        """Convert tool schemas to Gemini's function-calling format.

        Args:
            tools_schema: The tools schema containing standard and custom tool definitions.

        Returns:
            List of tool definitions formatted for Gemini's function-calling API.
            Includes both converted standard tools and any custom Gemini-specific tools.
        """
        functions_schema = tools_schema.standard_tools
        formatted_standard_tools = (
            [{"function_declarations": [func.to_default_dict() for func in functions_schema]}]
            if functions_schema
            else []
        )
        custom_gemini_tools = []
        if tools_schema.custom_tools:
            custom_gemini_tools = tools_schema.custom_tools.get(AdapterType.GEMINI, [])

        return formatted_standard_tools + custom_gemini_tools

    def get_messages_for_logging(self, context: LLMContext) -> List[Dict[str, Any]]:
        """Get messages from a universal LLM context in a format ready for logging about Gemini.

        Removes or truncates sensitive data like image content for safe logging.

        Args:
            context: The LLM context containing messages.

        Returns:
            List of messages in a format ready for logging about Gemini.
        """
        # Get messages in Gemini's format
        messages = self._from_universal_context_messages(self.get_messages(context)).messages

        # Sanitize messages for logging
        messages_for_logging = []
        for message in messages:
            obj = message.to_json_dict()
            try:
                if "parts" in obj:
                    for part in obj["parts"]:
                        if "inline_data" in part:
                            part["inline_data"]["data"] = "..."
            except Exception as e:
                logger.debug(f"Error: {e}")
            messages_for_logging.append(obj)
        return messages_for_logging

    @dataclass
    class ConvertedMessages:
        """Container for Google-formatted messages converted from universal context."""

        messages: List[Content]
        system_instruction: Optional[str] = None

    def _from_universal_context_messages(
        self, universal_context_messages: List[LLMContextMessage]
    ) -> ConvertedMessages:
        """Restructures messages to ensure proper Google format and message ordering.

        This method handles conversion of OpenAI-formatted messages to Google format,
        with special handling for function calls, function responses, and system messages.
        System messages are added back to the context as user messages when needed.

        The final message order is preserved as:

        1. Function calls (from model)
        2. Function responses (from user)
        3. Text messages (converted from system messages)

        Note::

            System messages are only added back when there are no regular text
            messages in the context, ensuring proper conversation continuity
            after function calls.
        """
        system_instruction = None
        messages = []

        # Process each message, preserving Google-formatted messages and converting others
        for message in universal_context_messages:
            if isinstance(message, LLMSpecificMessage):
                # Assume that LLMSpecificMessage wraps a message in Google format
                messages.append(message.message)
                continue

            # Convert standard format to Google format
            converted = self._from_standard_message(
                message, already_have_system_instruction=bool(system_instruction)
            )
            if isinstance(converted, Content):
                # Regular (non-system) message
                messages.append(converted)
            else:
                # System instruction
                system_instruction = converted

        # Check if we only have function-related messages (no regular text)
        has_regular_messages = any(
            len(msg.parts) == 1
            and getattr(msg.parts[0], "text", None)
            and not getattr(msg.parts[0], "function_call", None)
            and not getattr(msg.parts[0], "function_response", None)
            for msg in messages
        )

        # Add system instruction back as a user message if we only have function messages
        if system_instruction and not has_regular_messages:
            messages.append(Content(role="user", parts=[Part(text=system_instruction)]))

        # Remove any empty messages
        messages = [m for m in messages if m.parts]

        return self.ConvertedMessages(messages=messages, system_instruction=system_instruction)

    def _from_standard_message(
        self, message: LLMStandardMessage, already_have_system_instruction: bool
    ) -> Content | str:
        """Convert standard universal context message to Google Content object.

        Handles conversion of text, images, and function calls to Google's
        format.
        System instructions are returned as a plain string.

        Args:
            message: Message in standard universal context format.
            already_have_system_instruction: Whether we already have a system instruction

        Returns:
            Content object with role and parts, or a plain string for system
            messages.

        Examples:
            Standard text message::

                {
                    "role": "user",
                    "content": "Hello there"
                }

            Converts to Google Content with::

                Content(
                    role="user",
                    parts=[Part(text="Hello there")]
                )

            Standard function call message::

                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "test"}'
                            }
                        }
                    ]
                }

            Converts to Google Content with::

                Content(
                    role="model",
                    parts=[Part(function_call=FunctionCall(name="search", args={"query": "test"}))]
                )
        """
        role = message["role"]
        content = message.get("content", [])
        if role == "system":
            if already_have_system_instruction:
                role = "user"  # Convert system message to user role if we already have a system instruction
            else:
                # System instructions are returned as plain text
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # If content is a list, we assume it's a list of text parts, per the standard
                    return " ".join(part["text"] for part in content if part.get("type") == "text")
        elif role == "assistant":
            role = "model"

        parts = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                parts.append(
                    Part(
                        function_call=FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                    )
                )
        elif role == "tool":
            role = "model"
            try:
                response = json.loads(message["content"])
                if isinstance(response, dict):
                    response_dict = response
                else:
                    response_dict = {"value": response}
            except Exception as e:
                # Response might not be JSON-deserializable.
                # This occurs with a UserImageFrame, for example, where we get a plain "COMPLETED" string.
                response_dict = {"value": message["content"]}
            parts.append(
                Part(
                    function_response=FunctionResponse(
                        name="tool_call_result",  # seems to work to hard-code the same name every time
                        response=response_dict,
                    )
                )
            )
        elif isinstance(content, str):
            parts.append(Part(text=content))
        elif isinstance(content, list):
            for c in content:
                if c["type"] == "text":
                    parts.append(Part(text=c["text"]))
                elif c["type"] == "image_url":
                    parts.append(
                        Part(
                            inline_data=Blob(
                                mime_type="image/jpeg",
                                data=base64.b64decode(c["image_url"]["url"].split(",")[1]),
                            )
                        )
                    )
                elif c["type"] == "input_audio":
                    input_audio = c["input_audio"]
                    audio_bytes = base64.b64decode(input_audio["data"])
                    parts.append(Part(inline_data=Blob(mime_type="audio/wav", data=audio_bytes)))

        return Content(role=role, parts=parts)
