#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AWS Bedrock integration for Large Language Model services.

This module provides AWS Bedrock LLM service implementation with support for
Amazon Nova and Anthropic Claude models, including vision capabilities and
function calling.
"""

import asyncio
import base64
import copy
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.utils.tracing.service_decorators import traced_llm

try:
    import boto3
    import httpx
    from botocore.config import Config
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AWS services, you need to `pip install pipecat-ai[aws]`. Also, remember to set `AWS_SECRET_ACCESS_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_REGION` environment variable."
    )
    raise Exception(f"Missing module: {e}")


@dataclass
class AWSBedrockContextAggregatorPair:
    """Container for AWS Bedrock context aggregators.

    Provides convenient access to both user and assistant context aggregators
    for AWS Bedrock LLM operations.

    Parameters:
        _user: The user context aggregator instance.
        _assistant: The assistant context aggregator instance.
    """

    _user: "AWSBedrockUserContextAggregator"
    _assistant: "AWSBedrockAssistantContextAggregator"

    def user(self) -> "AWSBedrockUserContextAggregator":
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> "AWSBedrockAssistantContextAggregator":
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class AWSBedrockLLMContext(OpenAILLMContext):
    """AWS Bedrock-specific LLM context implementation.

    Extends OpenAI LLM context to handle AWS Bedrock's specific message format
    and system message handling. Manages conversion between OpenAI and Bedrock
    message formats.

    Args:
        messages: List of conversation messages in OpenAI format.
        tools: List of available function calling tools.
        tool_choice: Tool selection strategy or specific tool choice.
        system: System message content for AWS Bedrock.
    """

    def __init__(
        self,
        messages: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[dict] = None,
        *,
        system: Optional[str] = None,
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self.system = system

    @staticmethod
    def upgrade_to_bedrock(obj: OpenAILLMContext) -> "AWSBedrockLLMContext":
        """Upgrade an OpenAI LLM context to AWS Bedrock format.

        Args:
            obj: The OpenAI LLM context to upgrade.

        Returns:
            The upgraded AWS Bedrock LLM context.
        """
        logger.debug(f"Upgrading to AWS Bedrock: {obj}")
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, AWSBedrockLLMContext):
            obj.__class__ = AWSBedrockLLMContext
            obj._restructure_from_openai_messages()
        else:
            obj._restructure_from_bedrock_messages()
        return obj

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        """Create AWS Bedrock context from OpenAI context.

        Args:
            openai_context: The OpenAI LLM context to convert.

        Returns:
            New AWS Bedrock LLM context instance.
        """
        self = cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )
        self.set_llm_adapter(openai_context.get_llm_adapter())
        self._restructure_from_openai_messages()
        return self

    @classmethod
    def from_messages(cls, messages: List[dict]) -> "AWSBedrockLLMContext":
        """Create AWS Bedrock context from message list.

        Args:
            messages: List of messages in OpenAI format.

        Returns:
            New AWS Bedrock LLM context instance.
        """
        self = cls(messages=messages)
        self._restructure_from_openai_messages()
        return self

    @classmethod
    def from_image_frame(cls, frame: VisionImageRawFrame) -> "AWSBedrockLLMContext":
        """Create AWS Bedrock context from vision image frame.

        Args:
            frame: The vision image frame to convert.

        Returns:
            New AWS Bedrock LLM context instance.
        """
        context = cls()
        context.add_image_frame_message(
            format=frame.format, size=frame.size, image=frame.image, text=frame.text
        )
        return context

    def set_messages(self, messages: List):
        """Set the messages list and restructure for Bedrock format.

        Args:
            messages: List of messages to set.
        """
        self._messages[:] = messages
        self._restructure_from_openai_messages()

    def to_standard_messages(self, obj):
        """Convert AWS Bedrock message format to standard structured format.

        Handles text content and function calls for both user and assistant messages.

        Args:
            obj: Message in AWS Bedrock format:
                {
                    "role": "user/assistant",
                    "content": [{"text": str} | {"toolUse": {...}} | {"toolResult": {...}}]
                }

        Returns:
            List of messages in standard format:
            [
                {
                    "role": "user/assistant/tool",
                    "content": [{"type": "text", "text": str}]
                }
            ]
        """
        role = obj.get("role")
        content = obj.get("content")

        if role == "assistant":
            if isinstance(content, str):
                return [{"role": role, "content": [{"type": "text", "text": content}]}]
            elif isinstance(content, list):
                text_items = []
                tool_items = []
                for item in content:
                    if "text" in item:
                        text_items.append({"type": "text", "text": item["text"]})
                    elif "toolUse" in item:
                        tool_use = item["toolUse"]
                        tool_items.append(
                            {
                                "type": "function",
                                "id": tool_use["toolUseId"],
                                "function": {
                                    "name": tool_use["name"],
                                    "arguments": json.dumps(tool_use["input"]),
                                },
                            }
                        )
                messages = []
                if text_items:
                    messages.append({"role": role, "content": text_items})
                if tool_items:
                    messages.append({"role": role, "tool_calls": tool_items})
                return messages
        elif role == "user":
            if isinstance(content, str):
                return [{"role": role, "content": [{"type": "text", "text": content}]}]
            elif isinstance(content, list):
                text_items = []
                tool_items = []
                for item in content:
                    if "text" in item:
                        text_items.append({"type": "text", "text": item["text"]})
                    elif "toolResult" in item:
                        tool_result = item["toolResult"]
                        # Extract content from toolResult
                        result_content = ""
                        if isinstance(tool_result["content"], list):
                            for content_item in tool_result["content"]:
                                if "text" in content_item:
                                    result_content = content_item["text"]
                                elif "json" in content_item:
                                    result_content = json.dumps(content_item["json"])
                        else:
                            result_content = tool_result["content"]

                        tool_items.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_result["toolUseId"],
                                "content": result_content,
                            }
                        )
                messages = []
                if text_items:
                    messages.append({"role": role, "content": text_items})
                messages.extend(tool_items)
                return messages

    def from_standard_message(self, message):
        """Convert standard format message to AWS Bedrock format.

        Handles conversion of text content, tool calls, and tool results.
        Empty text content is converted to "(empty)".

        Args:
            message: Message in standard format:
                {
                    "role": "user/assistant/tool",
                    "content": str | [{"type": "text", ...}],
                    "tool_calls": [{"id": str, "function": {"name": str, "arguments": str}}]
                }

        Returns:
            Message in AWS Bedrock format:
            {
                "role": "user/assistant",
                "content": [
                    {"text": str} |
                    {"toolUse": {"toolUseId": str, "name": str, "input": dict}} |
                    {"toolResult": {"toolUseId": str, "content": [...], "status": str}}
                ]
            }
        """
        if message["role"] == "tool":
            # Try to parse the content as JSON if it looks like JSON
            try:
                if message["content"].strip().startswith("{") and message[
                    "content"
                ].strip().endswith("}"):
                    content_json = json.loads(message["content"])
                    tool_result_content = [{"json": content_json}]
                else:
                    tool_result_content = [{"text": message["content"]}]
            except:
                tool_result_content = [{"text": message["content"]}]

            return {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": message["tool_call_id"],
                            "content": tool_result_content,
                        },
                    },
                ],
            }

        if message.get("tool_calls"):
            tc = message["tool_calls"]
            ret = {"role": "assistant", "content": []}
            for tool_call in tc:
                function = tool_call["function"]
                arguments = json.loads(function["arguments"])
                new_tool_use = {
                    "toolUse": {
                        "toolUseId": tool_call["id"],
                        "name": function["name"],
                        "input": arguments,
                    }
                }
                ret["content"].append(new_tool_use)
            return ret

        # Handle text content
        content = message.get("content")
        if isinstance(content, str):
            if content == "":
                return {"role": message["role"], "content": [{"text": "(empty)"}]}
            else:
                return {"role": message["role"], "content": [{"text": content}]}
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type", "") == "text":
                    text_content = item["text"] if item["text"] != "" else "(empty)"
                    new_content.append({"text": text_content})
            return {"role": message["role"], "content": new_content}

        return message

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        """Add an image message to the context.

        Args:
            format: The image format (e.g., 'RGB', 'RGBA').
            size: The image dimensions as (width, height).
            image: The raw image data as bytes.
            text: Optional text to accompany the image.
        """
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Image should be the first content block in the message
        content = [{"type": "image", "format": "jpeg", "source": {"bytes": encoded_image}}]
        if text:
            content.append({"text": text})
        self.add_message({"role": "user", "content": content})

    def add_message(self, message):
        """Add a message to the context, merging with previous message if same role.

        AWS Bedrock requires alternating roles, so consecutive messages from the
        same role are merged together.

        Args:
            message: The message to add to the context.
        """
        try:
            if self.messages:
                # AWS Bedrock requires that roles alternate. If this message's
                # role is the same as the last message, we should add this
                # message's content to the last message.
                if self.messages[-1]["role"] == message["role"]:
                    # if the last message has just a content string, convert it to a list
                    # in the proper format
                    if isinstance(self.messages[-1]["content"], str):
                        self.messages[-1]["content"] = [{"text": self.messages[-1]["content"]}]
                    # if this message has just a content string, convert it to a list
                    # in the proper format
                    if isinstance(message["content"], str):
                        message["content"] = [{"text": message["content"]}]
                    # append the content of this message to the last message
                    self.messages[-1]["content"].extend(message["content"])
                else:
                    self.messages.append(message)
            else:
                self.messages.append(message)
        except Exception as e:
            logger.error(f"Error adding message: {e}")

    def _restructure_from_bedrock_messages(self):
        """Restructure messages in AWS Bedrock format.

        Handles system messages, merging consecutive messages with the same role,
        and ensuring proper content formatting.
        """
        # Handle system message if present at the beginning
        if self.messages and self.messages[0]["role"] == "system":
            if len(self.messages) == 1:
                self.messages[0]["role"] = "user"
            else:
                system_content = self.messages.pop(0)["content"]
                if isinstance(system_content, str):
                    system_content = [{"text": system_content}]

                if self.system:
                    if isinstance(self.system, str):
                        self.system = [{"text": self.system}]
                    self.system.extend(system_content)
                else:
                    self.system = system_content

        # Ensure content is properly formatted
        for msg in self.messages:
            if isinstance(msg["content"], str):
                msg["content"] = [{"text": msg["content"]}]
            elif not msg["content"]:
                msg["content"] = [{"text": "(empty)"}]
            elif isinstance(msg["content"], list):
                for idx, item in enumerate(msg["content"]):
                    if isinstance(item, dict) and "text" in item and item["text"] == "":
                        item["text"] = "(empty)"
                    elif isinstance(item, str) and item == "":
                        msg["content"][idx] = {"text": "(empty)"}

        # Merge consecutive messages with the same role
        merged_messages = []
        for msg in self.messages:
            if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                merged_messages[-1]["content"].extend(msg["content"])
            else:
                merged_messages.append(msg)

        self.messages.clear()
        self.messages.extend(merged_messages)

    def _restructure_from_openai_messages(self):
        # first, map across self._messages calling self.from_standard_message(m) to modify messages in place
        try:
            self._messages[:] = [self.from_standard_message(m) for m in self._messages]
        except Exception as e:
            logger.error(f"Error mapping messages: {e}")

        # See if we should pull the system message out of our context.messages list. (For
        # compatibility with Open AI messages format.)
        if self.messages and self.messages[0]["role"] == "system":
            self.system = self.messages[0]["content"]
            self.messages.pop(0)

        # Merge consecutive messages with the same role.
        i = 0
        while i < len(self.messages) - 1:
            current_message = self.messages[i]
            next_message = self.messages[i + 1]
            if current_message["role"] == next_message["role"]:
                # Convert content to list of dictionaries if it's a string
                if isinstance(current_message["content"], str):
                    current_message["content"] = [
                        {"type": "text", "text": current_message["content"]}
                    ]
                if isinstance(next_message["content"], str):
                    next_message["content"] = [{"type": "text", "text": next_message["content"]}]
                # Concatenate the content
                current_message["content"].extend(next_message["content"])
                # Remove the next message from the list
                self.messages.pop(i + 1)
            else:
                i += 1

        # Avoid empty content in messages
        for message in self.messages:
            if isinstance(message["content"], str) and message["content"] == "":
                message["content"] = "(empty)"
            elif isinstance(message["content"], list) and len(message["content"]) == 0:
                message["content"] = [{"type": "text", "text": "(empty)"}]

    def get_messages_for_persistent_storage(self):
        """Get messages formatted for persistent storage.

        Returns:
            List of messages including system message if present.
        """
        messages = super().get_messages_for_persistent_storage()
        if self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        return messages

    def get_messages_for_logging(self) -> str:
        """Get messages formatted for logging with sensitive data redacted.

        Returns:
            JSON string representation of messages with image data redacted.
        """
        msgs = []
        for message in self.messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("image"):
                            item["source"]["bytes"] = "..."
            msgs.append(msg)
        return json.dumps(msgs)


class AWSBedrockUserContextAggregator(LLMUserContextAggregator):
    """User context aggregator for AWS Bedrock LLM service.

    Handles aggregation of user messages and frames for AWS Bedrock format.
    Inherits all functionality from the base LLM user context aggregator.

    Args:
        context: The LLM context to aggregate messages into.
        params: Configuration parameters for the aggregator.
    """

    pass


class AWSBedrockAssistantContextAggregator(LLMAssistantContextAggregator):
    """Assistant context aggregator for AWS Bedrock LLM service.

    Handles aggregation of assistant responses and function calls for AWS Bedrock
    format, including tool use and tool result handling.

    Args:
        context: The LLM context to aggregate messages into.
        params: Configuration parameters for the aggregator.
    """

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle function call in progress frame.

        Args:
            frame: The function call in progress frame to handle.
        """
        # Format tool use according to AWS Bedrock API
        self._context.add_message(
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": frame.tool_call_id,
                            "name": frame.function_name,
                            "input": frame.arguments if frame.arguments else {},
                        }
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": frame.tool_call_id,
                            "content": [{"text": "IN_PROGRESS"}],
                        }
                    }
                ],
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle function call result frame.

        Args:
            frame: The function call result frame to handle.
        """
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle function call cancel frame.

        Args:
            frame: The function call cancel frame to handle.
        """
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if (
                        isinstance(content, dict)
                        and content.get("toolResult")
                        and content["toolResult"]["toolUseId"] == tool_call_id
                    ):
                        content["toolResult"]["content"] = [{"text": result}]

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle user image frame.

        Args:
            frame: The user image frame to handle.
        """
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )


class AWSBedrockLLMService(LLMService):
    """AWS Bedrock Large Language Model service implementation.

    Provides inference capabilities for AWS Bedrock models including Amazon Nova
    and Anthropic Claude. Supports streaming responses, function calling, and
    vision capabilities.

    Args:
        model: The AWS Bedrock model identifier to use.
        aws_access_key: AWS access key ID. If None, uses default credentials.
        aws_secret_key: AWS secret access key. If None, uses default credentials.
        aws_session_token: AWS session token for temporary credentials.
        aws_region: AWS region for the Bedrock service.
        params: Model parameters and configuration.
        client_config: Custom boto3 client configuration.
        **kwargs: Additional arguments passed to parent LLMService.
    """

    # Overriding the default adapter to use the Anthropic one.
    adapter_class = AWSBedrockLLMAdapter

    class InputParams(BaseModel):
        """Input parameters for AWS Bedrock LLM service.

        Parameters:
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature between 0.0 and 1.0.
            top_p: Nucleus sampling parameter between 0.0 and 1.0.
            stop_sequences: List of strings that stop generation.
            latency: Performance mode - "standard" or "optimized".
            additional_model_request_fields: Additional model-specific parameters.
        """

        max_tokens: Optional[int] = Field(default_factory=lambda: 4096, ge=1)
        temperature: Optional[float] = Field(default_factory=lambda: 0.7, ge=0.0, le=1.0)
        top_p: Optional[float] = Field(default_factory=lambda: 0.999, ge=0.0, le=1.0)
        stop_sequences: Optional[List[str]] = Field(default_factory=lambda: [])
        latency: Optional[str] = Field(default_factory=lambda: "standard")
        additional_model_request_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: str = "us-east-1",
        params: Optional[InputParams] = None,
        client_config: Optional[Config] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        params = params or AWSBedrockLLMService.InputParams()

        # Initialize the AWS Bedrock client
        if not client_config:
            client_config = Config(
                connect_timeout=300,  # 5 minutes
                read_timeout=300,  # 5 minutes
                retries={"max_attempts": 3},
            )
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=aws_region,
        )
        self._client = session.client(service_name="bedrock-runtime", config=client_config)

        self.set_model_name(model)
        self._settings = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "latency": params.latency,
            "additional_model_request_fields": params.additional_model_request_fields
            if isinstance(params.additional_model_request_fields, dict)
            else {},
        }

        logger.info(f"Using AWS Bedrock model: {model}")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> AWSBedrockContextAggregatorPair:
        """Create AWS Bedrock-specific context aggregators.

        Creates a pair of context aggregators optimized for AWS Bedrocks's message
        format, including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            AWSBedrockContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            AWSBedrockContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext):
            context = AWSBedrockLLMContext.from_openai_context(context)

        user = AWSBedrockUserContextAggregator(context, params=user_params)
        assistant = AWSBedrockAssistantContextAggregator(context, params=assistant_params)
        return AWSBedrockContextAggregatorPair(_user=user, _assistant=assistant)

    def _create_no_op_tool(self):
        """Create a no-operation tool for AWS Bedrock when tool content exists but no tools are defined.

        This is required because AWS Bedrock doesn't allow empty tool configurations after tools were
        previously set. Other LLM vendors allow NOT_GIVEN or empty tool configurations,
        but AWS Bedrock requires at least one tool to be defined.
        """
        return {
            "toolSpec": {
                "name": "no_operation",
                "description": "Internal placeholder function. Do not call this function.",
                "inputSchema": {"json": {"type": "object", "properties": {}, "required": []}},
            }
        }

    @traced_llm
    async def _process_context(self, context: AWSBedrockLLMContext):
        # Usage tracking
        prompt_tokens = 0
        completion_tokens = 0
        completion_tokens_estimate = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        use_completion_tokens_estimate = False

        using_noop_tool = False

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            await self.start_ttfb_metrics()

            # Set up inference config
            inference_config = {
                "maxTokens": self._settings["max_tokens"],
                "temperature": self._settings["temperature"],
                "topP": self._settings["top_p"],
            }

            # Prepare request parameters
            request_params = {
                "modelId": self.model_name,
                "messages": context.messages,
                "inferenceConfig": inference_config,
                "additionalModelRequestFields": self._settings["additional_model_request_fields"],
            }

            # Add system message
            request_params["system"] = context.system

            # Check if messages contain tool use or tool result content blocks
            has_tool_content = False
            for message in context.messages:
                if isinstance(message.get("content"), list):
                    for content_item in message["content"]:
                        if "toolUse" in content_item or "toolResult" in content_item:
                            has_tool_content = True
                            break
                if has_tool_content:
                    break

            # Handle tools: use current tools, or no-op if tool content exists but no current tools
            tools = context.tools or []
            if has_tool_content and not tools:
                tools = [self._create_no_op_tool()]
                using_noop_tool = True

            if tools:
                tool_config = {"tools": tools}

                # Only add tool_choice if we have real tools (not just no-op)
                if not using_noop_tool and context.tool_choice:
                    if context.tool_choice == "auto":
                        tool_config["toolChoice"] = {"auto": {}}
                    elif context.tool_choice == "none":
                        # Skip adding toolChoice for "none"
                        pass
                    elif (
                        isinstance(context.tool_choice, dict) and "function" in context.tool_choice
                    ):
                        tool_config["toolChoice"] = {
                            "tool": {"name": context.tool_choice["function"]["name"]}
                        }

                request_params["toolConfig"] = tool_config

            # Add performance config if latency is specified
            if self._settings["latency"] in ["standard", "optimized"]:
                request_params["performanceConfig"] = {"latency": self._settings["latency"]}

            logger.debug(f"Calling AWS Bedrock model with: {request_params}")

            # Call AWS Bedrock with streaming
            response = self._client.converse_stream(**request_params)

            await self.stop_ttfb_metrics()

            # Process the streaming response
            tool_use_block = None
            json_accumulator = ""

            function_calls = []
            for event in response["stream"]:
                self.reset_watchdog()

                # Handle text content
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        await self.push_frame(LLMTextFrame(delta["text"]))
                        completion_tokens_estimate += self._estimate_tokens(delta["text"])
                    elif "toolUse" in delta and "input" in delta["toolUse"]:
                        # Handle partial JSON for tool use
                        json_accumulator += delta["toolUse"]["input"]
                        completion_tokens_estimate += self._estimate_tokens(
                            delta["toolUse"]["input"]
                        )

                # Handle tool use start
                elif "contentBlockStart" in event:
                    content_block_start = event["contentBlockStart"]["start"]
                    if "toolUse" in content_block_start:
                        tool_use_block = {
                            "id": content_block_start["toolUse"].get("toolUseId", ""),
                            "name": content_block_start["toolUse"].get("name", ""),
                        }
                        json_accumulator = ""

                # Handle message completion with tool use
                elif "messageStop" in event and "stopReason" in event["messageStop"]:
                    if event["messageStop"]["stopReason"] == "tool_use" and tool_use_block:
                        try:
                            arguments = json.loads(json_accumulator) if json_accumulator else {}

                            # Only call function if it's not the no_operation tool
                            if not using_noop_tool:
                                function_calls.append(
                                    FunctionCallFromLLM(
                                        context=context,
                                        tool_call_id=tool_use_block["id"],
                                        function_name=tool_use_block["name"],
                                        arguments=arguments,
                                    )
                                )
                            else:
                                logger.debug("Ignoring no_operation tool call")
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {json_accumulator}")

                # Handle usage metrics if available
                if "metadata" in event and "usage" in event["metadata"]:
                    usage = event["metadata"]["usage"]
                    prompt_tokens += usage.get("inputTokens", 0)
                    completion_tokens += usage.get("outputTokens", 0)
                    cache_read_input_tokens += usage.get("cacheReadInputTokens", 0)
                    cache_creation_input_tokens += usage.get("cacheWriteInputTokens", 0)

            await self.run_function_calls(function_calls)
        except asyncio.CancelledError:
            # If we're interrupted, we won't get a complete usage report. So set our flag to use the
            # token estimate. The reraise the exception so all the processors running in this task
            # also get cancelled.
            use_completion_tokens_estimate = True
            raise
        except httpx.TimeoutException:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            comp_tokens = (
                completion_tokens
                if not use_completion_tokens_estimate
                else completion_tokens_estimate
            )
            await self._report_usage_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle LLM-specific frame types.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context = AWSBedrockLLMContext.upgrade_to_bedrock(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            context = AWSBedrockLLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            # This is only useful in very simple pipelines because it creates
            # a new context. Generally we want a context manager to catch
            # UserImageRawFrames coming through the pipeline and add them
            # to the context.
            context = AWSBedrockLLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(re.split(r"[^\w]+", text)) * 1.3)

    async def _report_usage_metrics(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_input_tokens: int,
        cache_creation_input_tokens: int,
    ):
        if prompt_tokens or completion_tokens:
            tokens = LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
            )
            await self.start_llm_usage_metrics(tokens)
