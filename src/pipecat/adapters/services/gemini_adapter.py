#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini LLM adapter for Pipecat."""

import base64
import json
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

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

    system_instruction: str | None
    messages: list[Content]
    tools: list[Any] | NotGiven


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

    def get_llm_invocation_params(
        self, context: LLMContext, *, system_instruction: str | None = None
    ) -> GeminiLLMInvocationParams:
        """Get Gemini-specific LLM invocation parameters from a universal LLM context.

        Args:
            context: The LLM context containing messages, tools, etc.
            system_instruction: Optional system instruction from service settings
                or ``run_inference``.

        Returns:
            Dictionary of parameters for Gemini's API.
        """
        converted = self._from_universal_context_messages(
            self.get_messages(context), system_instruction=system_instruction
        )
        effective_system = self._resolve_system_instruction(
            converted.system_instruction,
            system_instruction,
            discard_context_system=True,
        )
        return {
            "system_instruction": effective_system,
            "messages": converted.messages,
            # NOTE: LLMContext's tools are guaranteed to be a ToolsSchema (or NOT_GIVEN)
            "tools": self.from_standard_tools(context.tools),
        }

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[dict[str, Any]]:
        """Convert tool schemas to Gemini's function-calling format.

        Args:
            tools_schema: The tools schema containing standard and custom tool definitions.

        Returns:
            List of tool definitions formatted for Gemini's function-calling API.
            Includes both converted standard tools and any custom Gemini-specific tools.
        """

        def _strip_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
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

    @staticmethod
    def to_function_response_dict(content: Any) -> dict[str, Any]:
        """Convert a tool-result content value to Gemini's FunctionResponse.response shape.

        Gemini's ``FunctionResponse.response`` field requires a dict, so
        non-dict values (e.g. plain strings, JSON-encoded scalars, or
        sentinel strings like ``"COMPLETED"`` used when a function returned
        no value) are wrapped as ``{"value": <value>}``. JSON strings that
        decode to a dict are passed through as-is.

        Args:
            content: The tool-result content. Typically the JSON-encoded
                return value of a function, but can also be a plain string
                (e.g. ``"COMPLETED"``) or already-parsed dict.

        Returns:
            A dict suitable for ``FunctionResponse.response``.
        """
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return {"value": content}
        try:
            decoded = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return {"value": content}
        if isinstance(decoded, dict):
            return decoded
        return {"value": decoded}

    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
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
        messages_for_logging: list[dict[str, Any]] = []
        for message in messages:
            # `to_json_dict()` returns `dict[str, object]`; treat as a plain
            # dict for the value indexing/mutation below. The broad `except`
            # below is the safety net if any item isn't shaped as expected.
            obj: dict[str, Any] = cast(dict[str, Any], message.to_json_dict())
            try:
                if "parts" in obj:
                    for part in obj["parts"]:
                        if "inline_data" in part:
                            part["inline_data"]["data"] = "..."
                        if "thought_signature" in part:
                            part["thought_signature"] = "..."
            except Exception as e:
                logger.debug(f"Error: {e}")
            messages_for_logging.append(obj)
        return messages_for_logging

    @dataclass
    class ConvertedMessages:
        """Container for Google-formatted messages converted from universal context."""

        messages: list[Content]
        system_instruction: str | None = None

    @dataclass
    class MessageConversionResult:
        """Result of converting a single universal context message to Google format.

        Contains a Google Content object and a tool call ID to name mapping
        for any tool calls discovered in the message.
        """

        content: Content | None = None
        tool_call_id_to_name_mapping: dict[str, str] = field(default_factory=dict)

    @dataclass
    class MessageConversionParams:
        """Parameters for converting a single universal context message to Google format."""

        tool_call_id_to_name_mapping: dict[str, str]

    def _from_universal_context_messages(
        self,
        universal_context_messages: list[LLMContextMessage],
        *,
        system_instruction: str | None = None,
    ) -> ConvertedMessages:
        """Restructures messages to ensure proper Google format and message ordering.

        This method handles conversion of OpenAI-formatted messages to Google format,
        with special handling for function calls, function responses, and system/developer
        messages.

        Initial system/developer messages are extracted as the system instruction
        (only from ``messages[0]``). Subsequent system/developer messages are
        converted to user role.

        Args:
            universal_context_messages: Messages from the LLM context.
            system_instruction: Optional system instruction from service settings,
                used to decide whether to extract an initial "developer" message.
        """
        # Extract initial system/developer message from universal messages before conversion.
        # We work on a mutable copy so we can pop messages[0] if needed.
        remaining_messages = list(universal_context_messages)
        extracted_system = None

        # Extract initial system message from universal messages BEFORE conversion,
        # so the helper works with standard message format.
        if remaining_messages and not isinstance(remaining_messages[0], LLMSpecificMessage):
            extracted_system = self._extract_initial_system(
                remaining_messages, system_instruction=system_instruction
            )

        messages = []
        tool_call_id_to_name_mapping = {}
        thought_signature_dicts = []

        # Process each message, converting to Google format as needed
        for message in remaining_messages:
            # We have a Google-specific message; this may either be a
            # thought-signature-containing message that we need to handle in a
            # special way, or a message already in Google format that we can
            # use directly
            if isinstance(message, LLMSpecificMessage):
                if (
                    isinstance(message.message, dict)
                    and message.message.get("type") == "thought_signature"
                ):
                    thought_signature_dicts.append(message.message)
                    continue

                # Fall back to assuming that the message is already in Google
                # format
                messages.append(message.message)
                continue

            # We have a standard universal context message; convert it to
            # Google format
            result = self._from_standard_message(
                message,
                params=self.MessageConversionParams(
                    tool_call_id_to_name_mapping=tool_call_id_to_name_mapping,
                ),
            )

            if result.content:
                messages.append(result.content)

            # Merge tool call ID to name mapping
            if result.tool_call_id_to_name_mapping:
                tool_call_id_to_name_mapping.update(result.tool_call_id_to_name_mapping)

        # Apply thought signatures to the corresponding messages
        self._apply_thought_signatures_to_messages(thought_signature_dicts, messages)

        # When thinking is enabled, merge parallel tool calls into single messages
        messages = self._merge_parallel_tool_calls_for_thinking(thought_signature_dicts, messages)

        # Check if we only have function-related messages (no regular text)
        effective_system = extracted_system or system_instruction
        has_regular_messages = any(
            msg.parts is not None
            and len(msg.parts) == 1
            and getattr(msg.parts[0], "text", None)
            and not getattr(msg.parts[0], "function_call", None)
            and not getattr(msg.parts[0], "function_response", None)
            for msg in messages
        )

        # Add system instruction back as a user message if we only have function messages
        if effective_system and not has_regular_messages:
            messages.append(Content(role="user", parts=[Part(text=effective_system)]))

        # Remove any empty messages
        messages = [m for m in messages if m.parts]

        return self.ConvertedMessages(
            messages=messages,
            system_instruction=extracted_system,
        )

    def _from_standard_message(
        self, message: LLMStandardMessage, *, params: MessageConversionParams
    ) -> MessageConversionResult:
        """Convert standard universal context message to Google Content object.

        Handles conversion of text, images, and function calls to Google's
        format. System and developer messages at this stage (i.e. non-initial
        ones, since the initial one is already extracted) are converted to
        user role.

        Args:
            message: Message in standard universal context format.
            params: Parameters for conversion.

        Returns:
            MessageConversionResult containing a Content object.

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
        # ChatCompletionMessageParam (a union of TypedDicts) doesn't allow
        # the dict-style key access used below; treat it as a plain dict.
        msg = cast(dict[str, Any], message)
        role = msg["role"]
        content = msg.get("content", [])

        # Convert non-initial system/developer messages to user role,
        # as Gemini doesn't support these as input messages.
        if role in ("system", "developer"):
            role = "user"
        elif role == "assistant":
            role = "model"

        parts = []
        tool_call_id_to_name_mapping = {}

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
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
            response_dict = self.to_function_response_dict(msg["content"])

            # Get function name from mapping using tool_call_id, or fallback
            tool_call_id = msg.get("tool_call_id")
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
                    # Extract MIME type from data URL (format: "data:image/jpeg;base64,...")
                    url = c["image_url"]["url"]
                    mime_type = url.split(":")[1].split(";")[0]
                    parts.append(
                        Part(
                            inline_data=Blob(
                                mime_type=mime_type,
                                data=base64.b64decode(url.split(",")[1]),
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

    def _merge_parallel_tool_calls_for_thinking(
        self, thought_signature_dicts: list[dict], messages: list[Content]
    ) -> list[Content]:
        """Merge parallel tool calls into single Content objects when thinking is enabled.

        Gemini expects parallel tool calls (multiple function calls made
        simultaneously) to be in a single Content with multiple function_call
        Parts. This method takes a list of Content messages, where parallel
        tool calls may be split across multiple messages, and merges them into
        single messages.

        This only has an effect when thought_signatures are present (i.e., when
        thinking is enabled). When thinking is disabled, merging doesn't matter.
        When thinking is enabled, there is a guarantee that the first tool call
        (and only the first) in any batch of parallel tool calls will have a
        thought_signature. This allows us to distinguish:

        - Parallel tool calls: share a single thought_signature (on the first call)
        - Sequential tool calls: each have their own thought_signature

        Algorithm: A tool call message with a thought_signature starts a new
        parallel group. Any tool call messages after it without a
        thought_signature get merged into that group, regardless of what
        messages appear in between.

        Args:
            thought_signature_dicts: A list of thought signature dicts, used
                to determine if the work of merging is necessary.
            messages: List of Content messages to process.

        Returns:
            List of Content messages with parallel tool calls merged when
            thought_signatures are present, otherwise unchanged.
        """
        if not messages:
            return messages

        # Fast-exit if no function-call-related thought signatures
        # This is a shortcut for determining both:
        # - whether thinking is enabled, and
        # - whether there are function calls in the messages
        has_function_call_signatures = any(
            ts.get("bookmark", {}).get("function_call") for ts in thought_signature_dicts
        )
        if not has_function_call_signatures:
            return messages

        def is_tool_call_message(msg: Content) -> bool:
            """Check if message contains only function_call parts."""
            return bool(
                msg.role == "model"
                and msg.parts
                and all(getattr(part, "function_call", None) for part in msg.parts)
            )

        def message_has_thought_signature(msg: Content) -> bool:
            """Check if any part in the message has a thought_signature."""
            if msg.parts is None:
                return False
            return any(getattr(part, "thought_signature", None) for part in msg.parts)

        merged_messages = []
        i = 0

        while i < len(messages):
            current = messages[i]

            # If this is a tool call message with a thought signature, start merging
            if is_tool_call_message(current) and message_has_thought_signature(current):
                merged_parts = list(current.parts)
                other_messages = []
                j = i + 1

                # Scan forward, merging tool calls without signatures, collecting others
                while j < len(messages):
                    next_msg = messages[j]
                    if is_tool_call_message(next_msg):
                        if message_has_thought_signature(next_msg):
                            # New parallel group starts, stop here
                            break
                        else:
                            # Merge this call into the current group
                            merged_parts.extend(next_msg.parts)
                            j += 1
                    else:
                        # Collect non-tool-call message, keep scanning
                        other_messages.append(next_msg)
                        j += 1

                # Output merged calls, then collected other messages
                merged_messages.append(Content(role="model", parts=merged_parts))
                merged_messages.extend(other_messages)
                i = j
            else:
                merged_messages.append(current)
                i += 1

        return merged_messages

    def _apply_thought_signatures_to_messages(
        self, thought_signature_dicts: list[dict], messages: list[Content]
    ) -> None:
        """Apply thought signatures to corresponding assistant messages.

        See GoogleLLMService for more details about thought signatures.

        Args:
            thought_signature_dicts: A list of dicts containing:
                - "signature": a thought signature
                - "bookmark": a bookmark to identify the message part to apply the signature to.
                  The bookmark may contain one of:
                    - "function_call" (a function call ID string)
                    - "text" (a text string)
                    - "inline_data" (a Blob)
                The list of thought signature dicts is in order.
            messages: List of messages to apply the thought signatures to.
        """
        if not thought_signature_dicts:
            return

        # For debugging, print out thought signatures and their bookmarks
        logger.debug(f"Thought signatures to apply: {len(thought_signature_dicts)}")
        for ts in thought_signature_dicts:
            bookmark = ts.get("bookmark")
            if bookmark is None:
                continue
            if bookmark.get("function_call"):
                logger.trace(f" - To function call: {bookmark['function_call']}")
            elif bookmark.get("text"):
                text = bookmark["text"]
                log_display_text = f"{text[:50]}..." if len(text) > 50 else text
                logger.trace(f" - To text: {log_display_text}")
            elif bookmark.get("inline_data"):
                logger.trace(f" - To inline data")

        # Get all assistant messages
        assistant_messages = [
            message
            for message in messages
            if isinstance(message, Content) and message.role == "model"
        ]

        # Apply thought signatures to the corresponding assistant messages.
        # Thought signatures are already in message order.
        thought_signatures_applied = 0
        message_start_index = 0  # Track where to start searching for the next matching message.
        for thought_signature_dict in thought_signature_dicts:
            signature = thought_signature_dict.get("signature")
            bookmark = thought_signature_dict.get("bookmark")
            if not signature or not bookmark:
                continue

            # Search through remaining assistant messages for a match
            for i in range(message_start_index, len(assistant_messages)):
                message = assistant_messages[i]
                if not message.parts:
                    continue

                # We're assuming that the thought signature always applies to the last part
                last_part = message.parts[-1]

                # If the bookmark matches the part...
                if self._thought_signature_bookmark_matches_part(bookmark, last_part):
                    # Apply the thought signature
                    last_part.thought_signature = signature
                    thought_signatures_applied += 1

                    # Update the start index and stop searching for a match
                    message_start_index = i + 1
                    break

        # For debugging, print out how many thought signatures were applied
        logger.debug(f"Applied {thought_signatures_applied} thought signatures.")

    def _thought_signature_bookmark_matches_part(self, bookmark: dict, part: Part) -> bool:
        if function_call_bookmark := bookmark.get("function_call"):
            return self._thought_signature_function_call_bookmark_matches_part(
                function_call_bookmark, part
            )
        elif text_bookmark := bookmark.get("text"):
            return self._thought_signature_text_bookmark_matches_part(text_bookmark, part)
        elif inline_data := bookmark.get("inline_data"):
            return self._thought_signature_inline_data_bookmark_matches_part(inline_data, part)
        else:
            logger.warning(f"Unknown thought signature bookmark type: {bookmark}")

        return False

    def _thought_signature_function_call_bookmark_matches_part(
        self, bookmark_function_call_id: str, part: Part
    ) -> bool:
        if (
            hasattr(part, "function_call")
            and part.function_call
            and part.function_call.id == bookmark_function_call_id
        ):
            logger.trace(f"Thought signature function call match: {bookmark_function_call_id}")
            return True

        return False

    def _thought_signature_text_bookmark_matches_part(self, bookmark_text: str, part: Part) -> bool:
        if hasattr(part, "text") and part.text:
            # Normalize whitespace for comparison
            bookmark_text = " ".join(bookmark_text.split())
            part_text = " ".join(part.text.split())
            # Check that either:
            # - the part text is the same as the bookmark text
            # - a prefix of the bookmark text (in case the part text was truncated due to interruption)
            # - the bookmark text is a prefix of the part text (in case the bookmark represents just first chunk of multi-chunk text)
            if (
                part_text == bookmark_text
                or bookmark_text.startswith(part_text)
                or part_text.startswith(bookmark_text)
            ):
                log_display_text = f"{part.text[:50]}..." if len(part.text) > 50 else part.text
                logger.trace(f"Thought signature text match: {log_display_text}")
                return True

        return False

    def _thought_signature_inline_data_bookmark_matches_part(
        self, bookmark_inline_data: Blob, part: Part
    ) -> bool:
        if (
            hasattr(part, "inline_data")
            and part.inline_data
            and part.inline_data.data is not None
            and bookmark_inline_data.data is not None
            # Comparing length should be good enough for matching inline data,
            # especially since we're already matching thought signatures in
            # strict message order. Comparing actual data is expensive.
            and len(part.inline_data.data) == len(bookmark_inline_data.data)
        ):
            logger.trace(f"Thought signature inline data match")
            return True

        return False
