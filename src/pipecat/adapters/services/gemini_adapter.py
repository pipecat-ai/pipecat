#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini LLM adapter for Pipecat."""

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

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
    from google.genai.types import Blob, Content, FileData, FunctionCall, FunctionResponse, Part
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

        def _strip_additional_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively remove "additionalProperties" fields from JSON schema, as they're not supported by Gemini.

            Args:
                schema: The JSON schema dict to process.

            Returns:
                JSON schema dict with "additionalProperties" stripped out.
            """
            if not isinstance(schema, dict):
                return schema

            result = {}

            for key, value in schema.items():
                if key == "additionalProperties":
                    continue
                elif isinstance(value, dict):
                    result[key] = _strip_additional_properties(value)
                elif isinstance(value, list):
                    result[key] = [
                        _strip_additional_properties(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    result[key] = value

            return result

        functions_schema = tools_schema.standard_tools
        if functions_schema:
            formatted_functions = []
            for func in functions_schema:
                func_dict = func.to_default_dict()
                func_dict["parameters"]["properties"] = _strip_additional_properties(
                    func_dict["parameters"]["properties"]
                )
                formatted_functions.append(func_dict)
            formatted_standard_tools = [{"function_declarations": formatted_functions}]
        else:
            formatted_standard_tools = []
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

    @dataclass
    class MessageConversionResult:
        """Result of converting a single universal context message to Google format.

        Either content (a Google Content object) or a system instruction string
        is guaranteed to be set.

        Also returns a tool call ID to name mapping for any tool calls
        discovered in the message.
        """

        content: Optional[Content] = None
        system_instruction: Optional[str] = None
        tool_call_id_to_name_mapping: Dict[str, str] = field(default_factory=dict)

    @dataclass
    class MessageConversionParams:
        """Parameters for converting a single universal context message to Google format."""

        already_have_system_instruction: bool
        tool_call_id_to_name_mapping: Dict[str, str]

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
        tool_call_id_to_name_mapping = {}

        # Process each message, preserving Google-formatted messages and converting others
        for message in universal_context_messages:
            result = self._from_universal_context_message(
                message,
                params=self.MessageConversionParams(
                    already_have_system_instruction=bool(system_instruction),
                    tool_call_id_to_name_mapping=tool_call_id_to_name_mapping,
                ),
            )
            # Each result is either a Content or a system instruction
            if result.content:
                messages.append(result.content)
            elif result.system_instruction:
                system_instruction = result.system_instruction

            # Merge tool call ID to name mapping
            if result.tool_call_id_to_name_mapping:
                tool_call_id_to_name_mapping.update(result.tool_call_id_to_name_mapping)

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

    def _from_universal_context_message(
        self, message: LLMContextMessage, *, params: MessageConversionParams
    ) -> MessageConversionResult:
        if isinstance(message, LLMSpecificMessage):
            return self.MessageConversionResult(content=message.message)
        return self._from_standard_message(message, params=params)

    def _from_standard_message(
        self, message: LLMStandardMessage, *, params: MessageConversionParams
    ) -> MessageConversionResult:
        """Convert standard universal context message to Google Content object.

        Handles conversion of text, images, and function calls to Google's
        format.
        System instructions are returned as a plain string.

        Args:
            message: Message in standard universal context format.
            already_have_system_instruction: Whether we already have a system instruction
            params: Parameters for conversion.

        Returns:
            MessageConversionResult containing either a Content object or a
            system instruction string.

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
                    role="user",
                    parts=[Part(function_call=FunctionCall(name="search", args={"query": "test"}))]
                )
        """
        role = message["role"]
        content = message.get("content", [])

        if role == "system":
            if params.already_have_system_instruction:
                role = "user"  # Convert system message to user role if we already have a system instruction
            else:
                system_instruction: str = None
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, list):
                    # If content is a list, we assume it's a list of text parts, per the standard
                    system_instruction = " ".join(
                        part["text"] for part in content if part.get("type") == "text"
                    )
                if system_instruction:
                    return self.MessageConversionResult(system_instruction=system_instruction)
        elif role == "assistant":
            role = "model"

        parts = []
        tool_call_id_to_name_mapping = {}

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                id = tc["id"]
                name = tc["function"]["name"]
                tool_call_id_to_name_mapping[id] = name
                parts.append(
                    Part(
                        function_call=FunctionCall(
                            id=id,
                            name=name,
                            args=json.loads(tc["function"]["arguments"]),
                        )
                    )
                )
        elif role == "tool":
            role = "user"
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

            # Get function name from mapping using tool_call_id, or fallback
            tool_call_id = message.get("tool_call_id")
            function_name = "tool_call_result"  # Default fallback
            if tool_call_id and tool_call_id in params.tool_call_id_to_name_mapping:
                function_name = params.tool_call_id_to_name_mapping[tool_call_id]

            parts.append(
                Part(
                    function_response=FunctionResponse(
                        id=tool_call_id,
                        name=function_name,
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
                elif c["type"] == "image_url" and c["image_url"]["url"].startswith("data:"):
                    parts.append(
                        Part(
                            inline_data=Blob(
                                mime_type="image/jpeg",
                                data=base64.b64decode(c["image_url"]["url"].split(",")[1]),
                            )
                        )
                    )
                elif c["type"] == "image_url":
                    url = c["image_url"]["url"]
                    logger.warning(f"Unsupported 'image_url': {url}")
                elif c["type"] == "input_audio":
                    input_audio = c["input_audio"]
                    audio_bytes = base64.b64decode(input_audio["data"])
                    parts.append(Part(inline_data=Blob(mime_type="audio/wav", data=audio_bytes)))
                elif c["type"] == "file_data":
                    file_data = c["file_data"]
                    parts.append(
                        Part(
                            file_data=FileData(
                                mime_type=file_data.get("mime_type"),
                                file_uri=file_data.get("file_uri"),
                            )
                        )
                    )

        return self.MessageConversionResult(
            content=Content(role=role, parts=parts),
            tool_call_id_to_name_mapping=tool_call_id_to_name_mapping,
        )
