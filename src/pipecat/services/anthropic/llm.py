#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import copy
import io
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMEnablePromptCachingFrame,
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

try:
    from anthropic import NOT_GIVEN, AsyncAnthropic, NotGiven
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Anthropic, you need to `pip install pipecat-ai[anthropic]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AnthropicContextAggregatorPair:
    _user: "AnthropicUserContextAggregator"
    _assistant: "AnthropicAssistantContextAggregator"

    def user(self) -> "AnthropicUserContextAggregator":
        return self._user

    def assistant(self) -> "AnthropicAssistantContextAggregator":
        return self._assistant


class AnthropicLLMService(LLMService):
    """This class implements inference with Anthropic's AI models.

    Can provide a custom client via the `client` kwarg, allowing you to
    use `AsyncAnthropicBedrock` and `AsyncAnthropicVertex` clients
    """

    # Overriding the default adapter to use the Anthropic one.
    adapter_class = AnthropicLLMAdapter

    class InputParams(BaseModel):
        enable_prompt_caching_beta: Optional[bool] = False
        max_tokens: Optional[int] = Field(default_factory=lambda: 4096, ge=1)
        temperature: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        top_k: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        top_p: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-3-7-sonnet-20250219",
        params: InputParams = InputParams(),
        client=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client or AsyncAnthropic(
            api_key=api_key
        )  # if the client is provided, use it and remove it, otherwise create a new one
        self.set_model_name(model)
        self._settings = {
            "max_tokens": params.max_tokens,
            "enable_prompt_caching_beta": params.enable_prompt_caching_beta or False,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def enable_prompt_caching_beta(self) -> bool:
        return self._enable_prompt_caching_beta

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> AnthropicContextAggregatorPair:
        """Create an instance of AnthropicContextAggregatorPair from an
        OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_params (LLMUserAggregatorParams, optional): User aggregator
                parameters.
            assistant_params (LLMAssistantAggregatorParams, optional): User
                aggregator parameters.

        Returns:
            AnthropicContextAggregatorPair: A pair of context aggregators, one
            for the user and one for the assistant, encapsulated in an
            AnthropicContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext):
            context = AnthropicLLMContext.from_openai_context(context)
        user = AnthropicUserContextAggregator(context, params=user_params)
        assistant = AnthropicAssistantContextAggregator(context, params=assistant_params)
        return AnthropicContextAggregatorPair(_user=user, _assistant=assistant)

    async def _process_context(self, context: OpenAILLMContext):
        # Usage tracking. We track the usage reported by Anthropic in prompt_tokens and
        # completion_tokens. We also estimate the completion tokens from output text
        # and use that estimate if we are interrupted, because we almost certainly won't
        # get a complete usage report if the task we're running in is cancelled.
        prompt_tokens = 0
        completion_tokens = 0
        completion_tokens_estimate = 0
        use_completion_tokens_estimate = False
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            logger.debug(
                f"{self}: Generating chat [{context.system}] | [{context.get_messages_for_logging()}]"
            )

            messages = context.messages
            if self._settings["enable_prompt_caching_beta"]:
                messages = context.get_messages_with_cache_control_markers()

            api_call = self._client.messages.create
            if self._settings["enable_prompt_caching_beta"]:
                api_call = self._client.beta.prompt_caching.messages.create

            await self.start_ttfb_metrics()

            params = {
                "tools": context.tools or [],
                "system": context.system,
                "messages": messages,
                "model": self.model_name,
                "max_tokens": self._settings["max_tokens"],
                "stream": True,
                "temperature": self._settings["temperature"],
                "top_k": self._settings["top_k"],
                "top_p": self._settings["top_p"],
            }

            params.update(self._settings["extra"])

            response = await api_call(**params)

            await self.stop_ttfb_metrics()

            # Function calling
            tool_use_block = None
            json_accumulator = ""

            async for event in response:
                # logger.debug(f"Anthropic LLM event: {event}")

                # Aggregate streaming content, create frames, trigger events

                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        await self.push_frame(LLMTextFrame(event.delta.text))
                        completion_tokens_estimate += self._estimate_tokens(event.delta.text)
                    elif hasattr(event.delta, "partial_json") and tool_use_block:
                        json_accumulator += event.delta.partial_json
                        completion_tokens_estimate += self._estimate_tokens(
                            event.delta.partial_json
                        )
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        tool_use_block = event.content_block
                        json_accumulator = ""
                elif (
                    event.type == "message_delta"
                    and hasattr(event.delta, "stop_reason")
                    and event.delta.stop_reason == "tool_use"
                ):
                    if tool_use_block:
                        await self.call_function(
                            context=context,
                            tool_call_id=tool_use_block.id,
                            function_name=tool_use_block.name,
                            arguments=json.loads(json_accumulator) if json_accumulator else dict(),
                        )

                # Calculate usage. Do this here in its own if statement, because there may be usage
                # data embedded in messages that we do other processing for, above.
                if hasattr(event, "usage"):
                    prompt_tokens += (
                        event.usage.input_tokens if hasattr(event.usage, "input_tokens") else 0
                    )
                    completion_tokens += (
                        event.usage.output_tokens if hasattr(event.usage, "output_tokens") else 0
                    )
                elif hasattr(event, "message") and hasattr(event.message, "usage"):
                    prompt_tokens += (
                        event.message.usage.input_tokens
                        if hasattr(event.message.usage, "input_tokens")
                        else 0
                    )
                    completion_tokens += (
                        event.message.usage.output_tokens
                        if hasattr(event.message.usage, "output_tokens")
                        else 0
                    )
                    cache_creation_input_tokens += (
                        event.message.usage.cache_creation_input_tokens
                        if (
                            hasattr(event.message.usage, "cache_creation_input_tokens")
                            and event.message.usage.cache_creation_input_tokens is not None
                        )
                        else 0
                    )
                    logger.debug(f"Cache creation input tokens: {cache_creation_input_tokens}")
                    cache_read_input_tokens += (
                        event.message.usage.cache_read_input_tokens
                        if (
                            hasattr(event.message.usage, "cache_read_input_tokens")
                            and event.message.usage.cache_read_input_tokens is not None
                        )
                        else 0
                    )
                    logger.debug(f"Cache read input tokens: {cache_read_input_tokens}")
                    total_input_tokens = (
                        prompt_tokens + cache_creation_input_tokens + cache_read_input_tokens
                    )
                    if total_input_tokens >= 1024:
                        context.turns_above_cache_threshold += 1

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
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: "AnthropicLLMContext" = AnthropicLLMContext.upgrade_to_anthropic(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            context = AnthropicLLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            # This is only useful in very simple pipelines because it creates
            # a new context. Generally we want a context manager to catch
            # UserImageRawFrames coming through the pipeline and add them
            # to the context.
            context = AnthropicLLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, LLMEnablePromptCachingFrame):
            logger.debug(f"Setting enable prompt caching to: [{frame.enable}]")
            self._settings["enable_prompt_caching_beta"] = frame.enable
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
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
    ):
        if (
            prompt_tokens
            or completion_tokens
            or cache_creation_input_tokens
            or cache_read_input_tokens
        ):
            tokens = LLMTokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            await self.start_llm_usage_metrics(tokens)


class AnthropicLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[dict] = None,
        *,
        system: Union[str, NotGiven] = NOT_GIVEN,
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)

        # For beta prompt caching. This is a counter that tracks the number of turns
        # we've seen above the cache threshold. We reset this when we reset the
        # messages list. We only care about this number being 0, 1, or 2. But
        # it's easiest just to treat it as a counter.
        self.turns_above_cache_threshold = 0

        self.system = system

    @staticmethod
    def upgrade_to_anthropic(obj: OpenAILLMContext) -> "AnthropicLLMContext":
        logger.debug(f"Upgrading to Anthropic: {obj}")
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, AnthropicLLMContext):
            obj.__class__ = AnthropicLLMContext
            obj._restructure_from_openai_messages()
        return obj

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        self = cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )
        self.set_llm_adapter(openai_context.get_llm_adapter())
        self._restructure_from_openai_messages()
        return self

    @classmethod
    def from_messages(cls, messages: List[dict]) -> "AnthropicLLMContext":
        self = cls(messages=messages)
        self._restructure_from_openai_messages()
        return self

    @classmethod
    def from_image_frame(cls, frame: VisionImageRawFrame) -> "AnthropicLLMContext":
        context = cls()
        context.add_image_frame_message(
            format=frame.format, size=frame.size, image=frame.image, text=frame.text
        )
        return context

    def set_messages(self, messages: List):
        self.turns_above_cache_threshold = 0
        self._messages[:] = messages
        self._restructure_from_openai_messages()

    # convert a message in Anthropic format into one or more messages in OpenAI format
    def to_standard_messages(self, obj):
        """Convert Anthropic message format to standard structured format.

        Handles text content and function calls for both user and assistant messages.

        Args:
            obj: Message in Anthropic format:
                {
                    "role": "user/assistant",
                    "content": str | [{"type": "text/tool_use/tool_result", ...}]
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
        # todo: image format (?)
        # tool_use
        role = obj.get("role")
        content = obj.get("content")
        if role == "assistant":
            if isinstance(content, str):
                return [{"role": role, "content": [{"type": "text", "text": content}]}]
            elif isinstance(content, list):
                text_items = []
                tool_items = []
                for item in content:
                    if item["type"] == "text":
                        text_items.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "tool_use":
                        tool_items.append(
                            {
                                "type": "function",
                                "id": item["id"],
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(item["input"]),
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
                    if item["type"] == "text":
                        text_items.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "tool_result":
                        tool_items.append(
                            {
                                "role": "tool",
                                "tool_call_id": item["tool_use_id"],
                                "content": item["content"],
                            }
                        )
                messages = []
                if text_items:
                    messages.append({"role": role, "content": text_items})
                messages.extend(tool_items)
                return messages

    def from_standard_message(self, message):
        """Convert standard format message to Anthropic format.

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
            Message in Anthropic format:
            {
                "role": "user/assistant",
                "content": str | [
                    {"type": "text", "text": str} |
                    {"type": "tool_use", "id": str, "name": str, "input": dict} |
                    {"type": "tool_result", "tool_use_id": str, "content": str}
                ]
            }
        """
        # todo: image messages (?)
        if message["role"] == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
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
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": function["name"],
                    "input": arguments,
                }
                ret["content"].append(new_tool_use)
            return ret
        # check for empty text strings
        content = message.get("content")
        if isinstance(content, str):
            if content == "":
                content = "(empty)"
        elif isinstance(content, list):
            for item in content:
                if item["type"] == "text" and item["text"] == "":
                    item["text"] = "(empty)"

        return message

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Anthropic docs say that the image should be the first content block in the message.
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded_image,
                },
            }
        ]
        if text:
            content.append({"type": "text", "text": text})
        self.add_message({"role": "user", "content": content})

    def add_message(self, message):
        try:
            if self.messages:
                # Anthropic requires that roles alternate. If this message's role is the same as the
                # last message, we should add this message's content to the last message.
                if self.messages[-1]["role"] == message["role"]:
                    # if the last message has just a content string, convert it to a list
                    # in the proper format
                    if isinstance(self.messages[-1]["content"], str):
                        self.messages[-1]["content"] = [
                            {"type": "text", "text": self.messages[-1]["content"]}
                        ]
                    # if this message has just a content string, convert it to a list
                    # in the proper format
                    if isinstance(message["content"], str):
                        message["content"] = [{"type": "text", "text": message["content"]}]
                    # append the content of this message to the last message
                    self.messages[-1]["content"].extend(message["content"])
                else:
                    self.messages.append(message)
            else:
                self.messages.append(message)
        except Exception as e:
            logger.error(f"Error adding message: {e}")

    def get_messages_with_cache_control_markers(self) -> List[dict]:
        try:
            messages = copy.deepcopy(self.messages)
            if self.turns_above_cache_threshold >= 1 and messages[-1]["role"] == "user":
                if isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
                messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            if (
                self.turns_above_cache_threshold >= 2
                and len(messages) > 2
                and messages[-3]["role"] == "user"
            ):
                if isinstance(messages[-3]["content"], str):
                    messages[-3]["content"] = [{"type": "text", "text": messages[-3]["content"]}]
                messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            return messages
        except Exception as e:
            logger.error(f"Error adding cache control marker: {e}")
            return self.messages

    def _restructure_from_openai_messages(self):
        # first, map across self._messages calling self.from_standard_message(m) to modify messages in place
        try:
            self._messages[:] = [self.from_standard_message(m) for m in self._messages]
        except Exception as e:
            logger.error(f"Error mapping messages: {e}")

        # See if we should pull the system message out of our context.messages list. (For
        # compatibility with Open AI messages format.)
        if self.messages and self.messages[0]["role"] == "system":
            if len(self.messages) == 1:
                # If we have only have a system message in the list, all we can really do
                # without introducing too much magic is change the role to "user".
                self.messages[0]["role"] = "user"
            else:
                # If we have more than one message, we'll pull the system message out of the
                # list.
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
        messages = super().get_messages_for_persistent_storage()
        if self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        return messages

    def get_messages_for_logging(self) -> str:
        msgs = []
        for message in self.messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "image":
                            item["source"]["data"] = "..."
            msgs.append(msg)
        return json.dumps(msgs)


class AnthropicUserContextAggregator(LLMUserContextAggregator):
    pass


#
# Claude returns a text content block along with a tool use content block. This works quite nicely
# with streaming. We get the text first, so we can start streaming it right away. Then we get the
# tool_use block. While the text is streaming to TTS and the transport, we can run the tool call.
#
# But Claude is verbose. It would be nice to come up with prompt language that suppresses Claude's
# chattiness about it's tool thinking.
#


class AnthropicAssistantContextAggregator(LLMAssistantContextAggregator):
    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        assistant_message = {"role": "assistant", "content": []}
        assistant_message["content"].append(
            {
                "type": "tool_use",
                "id": frame.tool_call_id,
                "name": frame.function_name,
                "input": frame.arguments,
            }
        )
        self._context.add_message(assistant_message)
        self._context.add_message(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": frame.tool_call_id,
                        "content": "IN_PROGRESS",
                    }
                ],
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
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
                        and content["type"] == "tool_result"
                        and content["tool_use_id"] == tool_call_id
                    ):
                        content["content"] = result

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )
