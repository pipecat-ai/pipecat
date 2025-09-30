#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Anthropic AI service integration for Pipecat.

This module provides LLM services and context management for Anthropic's Claude models,
including support for function calling, vision, and prompt caching features.
"""

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

from pipecat.adapters.services.anthropic_adapter import (
    AnthropicLLMAdapter,
    AnthropicLLMInvocationParams,
)
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMEnablePromptCachingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
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
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.utils.tracing.service_decorators import traced_llm

try:
    from anthropic import NOT_GIVEN, APITimeoutError, AsyncAnthropic, NotGiven
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Anthropic, you need to `pip install pipecat-ai[anthropic]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AnthropicContextAggregatorPair:
    """Pair of context aggregators for Anthropic conversations.

    Encapsulates both user and assistant context aggregators
    to manage conversation flow and message formatting.

    Parameters:
        _user: The user context aggregator.
        _assistant: The assistant context aggregator.
    """

    _user: "AnthropicUserContextAggregator"
    _assistant: "AnthropicAssistantContextAggregator"

    def user(self) -> "AnthropicUserContextAggregator":
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> "AnthropicAssistantContextAggregator":
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class AnthropicLLMService(LLMService):
    """LLM service for Anthropic's Claude models.

    Provides inference capabilities with Claude models including support for
    function calling, vision processing, streaming responses, and prompt caching.
    Can use custom clients like AsyncAnthropicBedrock and AsyncAnthropicVertex.
    """

    # Overriding the default adapter to use the Anthropic one.
    adapter_class = AnthropicLLMAdapter

    class InputParams(BaseModel):
        """Input parameters for Anthropic model inference.

        Parameters:
            enable_prompt_caching: Whether to enable the prompt caching feature.
            enable_prompt_caching_beta (deprecated): Whether to enable the beta prompt caching feature.

                .. deprecated:: 0.0.84
                    Use the `enable_prompt_caching` parameter instead.

            max_tokens: Maximum tokens to generate. Must be at least 1.
            temperature: Sampling temperature between 0.0 and 1.0.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter between 0.0 and 1.0.
            extra: Additional parameters to pass to the API.
        """

        enable_prompt_caching: Optional[bool] = None
        enable_prompt_caching_beta: Optional[bool] = None
        max_tokens: Optional[int] = Field(default_factory=lambda: 4096, ge=1)
        temperature: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        top_k: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        top_p: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

        def model_post_init(self, __context):
            """Post-initialization to handle deprecated parameters."""
            if self.enable_prompt_caching_beta is not None:
                import warnings

                warnings.simplefilter("always")
                warnings.warn(
                    "enable_prompt_caching_beta is deprecated. Use enable_prompt_caching instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        params: Optional[InputParams] = None,
        client=None,
        retry_timeout_secs: Optional[float] = 5.0,
        retry_on_timeout: Optional[bool] = False,
        **kwargs,
    ):
        """Initialize the Anthropic LLM service.

        Args:
            api_key: Anthropic API key for authentication.
            model: Model name to use. Defaults to "claude-sonnet-4-5-20250929".
            params: Optional model parameters for inference.
            client: Optional custom Anthropic client instance.
            retry_timeout_secs: Request timeout in seconds for retry logic.
            retry_on_timeout: Whether to retry the request once if it times out.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        super().__init__(**kwargs)
        params = params or AnthropicLLMService.InputParams()
        self._client = client or AsyncAnthropic(
            api_key=api_key
        )  # if the client is provided, use it and remove it, otherwise create a new one
        self.set_model_name(model)
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
        self._settings = {
            "max_tokens": params.max_tokens,
            "enable_prompt_caching": (
                params.enable_prompt_caching
                if params.enable_prompt_caching is not None
                else (
                    params.enable_prompt_caching_beta
                    if params.enable_prompt_caching_beta is not None
                    else False
                )
            ),
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate usage metrics.

        Returns:
            True, as Anthropic provides detailed token usage metrics.
        """
        return True

    async def _create_message_stream(self, api_call, params):
        """Create message stream with optional timeout and retry.

        Args:
            api_call: The Anthropic API method to call.
            params: Parameters for the API call.

        Returns:
            Async stream of message events.
        """
        if self._retry_on_timeout:
            try:
                response = await asyncio.wait_for(
                    api_call(**params), timeout=self._retry_timeout_secs
                )
                return response
            except (APITimeoutError, asyncio.TimeoutError):
                # Retry, this time without a timeout so we get a response
                logger.debug(f"{self}: Retrying message creation due to timeout")
                response = await api_call(**params)
                return response
        else:
            response = await api_call(**params)
            return response

    async def run_inference(self, context: LLMContext | OpenAILLMContext) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        messages = []
        system = NOT_GIVEN
        if isinstance(context, LLMContext):
            adapter: AnthropicLLMAdapter = self.get_llm_adapter()
            params = adapter.get_llm_invocation_params(
                context, enable_prompt_caching=self._settings["enable_prompt_caching"]
            )
            messages = params["messages"]
            system = params["system"]
        else:
            context = AnthropicLLMContext.upgrade_to_anthropic(context)
            messages = context.messages
            system = getattr(context, "system", NOT_GIVEN)

        # LLM completion
        response = await self._client.messages.create(
            model=self.model_name,
            messages=messages,
            system=system,
            max_tokens=8192,
            stream=False,
        )

        return response.content[0].text

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> AnthropicContextAggregatorPair:
        """Create Anthropic-specific context aggregators.

        Creates a pair of context aggregators optimized for Anthropic's message format,
        including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context.
            user_params: User aggregator parameters.
            assistant_params: Assistant aggregator parameters.

        Returns:
            A pair of context aggregators, one for the user and one for the assistant,
            encapsulated in an AnthropicContextAggregatorPair.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext):
            context = AnthropicLLMContext.from_openai_context(context)
        user = AnthropicUserContextAggregator(context, params=user_params)
        assistant = AnthropicAssistantContextAggregator(context, params=assistant_params)
        return AnthropicContextAggregatorPair(_user=user, _assistant=assistant)

    def _get_llm_invocation_params(
        self, context: OpenAILLMContext | LLMContext
    ) -> AnthropicLLMInvocationParams:
        # Universal LLMContext
        if isinstance(context, LLMContext):
            adapter: AnthropicLLMAdapter = self.get_llm_adapter()
            params = adapter.get_llm_invocation_params(
                context, enable_prompt_caching=self._settings["enable_prompt_caching"]
            )
            return params

        # Anthropic-specific context
        messages = (
            context.get_messages_with_cache_control_markers()
            if self._settings["enable_prompt_caching"]
            else context.messages
        )
        return AnthropicLLMInvocationParams(
            system=context.system,
            messages=messages,
            tools=context.tools or [],
        )

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext | LLMContext):
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

            params_from_context = self._get_llm_invocation_params(context)

            if isinstance(context, LLMContext):
                adapter = self.get_llm_adapter()
                context_type_for_logging = "universal"
                messages_for_logging = adapter.get_messages_for_logging(context)
            else:
                context_type_for_logging = "LLM-specific"
                messages_for_logging = context.get_messages_for_logging()
            logger.debug(
                f"{self}: Generating chat from {context_type_for_logging} context [{params_from_context['system']}] | {messages_for_logging}"
            )

            await self.start_ttfb_metrics()

            params = {
                "model": self.model_name,
                "max_tokens": self._settings["max_tokens"],
                "stream": True,
                "temperature": self._settings["temperature"],
                "top_k": self._settings["top_k"],
                "top_p": self._settings["top_p"],
            }

            # Messages, system, tools
            params.update(params_from_context)

            params.update(self._settings["extra"])

            response = await self._create_message_stream(self._client.messages.create, params)

            await self.stop_ttfb_metrics()

            # Function calling
            tool_use_block = None
            json_accumulator = ""

            function_calls = []
            async for event in response:
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
                        args = json.loads(json_accumulator) if json_accumulator else {}
                        function_calls.append(
                            FunctionCallFromLLM(
                                context=context,
                                tool_call_id=tool_use_block.id,
                                function_name=tool_use_block.name,
                                arguments=args,
                            )
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
                        if hasattr(
                            context, "turns_above_cache_threshold"
                        ):  # LLMContext doesn't have this attribute
                            context.turns_above_cache_threshold += 1

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
            await self.push_error(ErrorFrame(f"{e}"))
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
        """Process incoming frames and route them appropriately.

        Handles various frame types including context frames, message frames,
        vision frames, and settings updates.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: "AnthropicLLMContext" = AnthropicLLMContext.upgrade_to_anthropic(frame.context)
        elif isinstance(frame, LLMContextFrame):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = AnthropicLLMContext.from_messages(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, LLMEnablePromptCachingFrame):
            logger.debug(f"Setting enable prompt caching to: [{frame.enable}]")
            self._settings["enable_prompt_caching"] = frame.enable
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
    """LLM context specialized for Anthropic's message format and features.

    Extends OpenAILLMContext to handle Anthropic-specific features like
    system messages, prompt caching, and message format conversions.
    Manages conversation state and message history formatting.
    """

    def __init__(
        self,
        messages: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[dict] = None,
        *,
        system: Union[str, NotGiven] = NOT_GIVEN,
    ):
        """Initialize the Anthropic LLM context.

        Args:
            messages: Initial list of conversation messages.
            tools: Available function calling tools.
            tool_choice: Tool selection preference.
            system: System message content.
        """
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self.__setup_local()
        self.system = system

    def __setup_local(self):
        # For beta prompt caching. This is a counter that tracks the number of turns
        # we've seen above the cache threshold. We reset this when we reset the
        # messages list. We only care about this number being 0, 1, or 2. But
        # it's easiest just to treat it as a counter.
        self.turns_above_cache_threshold = 0
        return

    @staticmethod
    def upgrade_to_anthropic(obj: OpenAILLMContext) -> "AnthropicLLMContext":
        """Upgrade an OpenAI context to Anthropic format.

        Converts message format and restructures content for Anthropic compatibility.

        Args:
            obj: The OpenAI context to upgrade.

        Returns:
            The upgraded Anthropic context.
        """
        logger.debug(f"Upgrading to Anthropic: {obj}")
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, AnthropicLLMContext):
            obj.__class__ = AnthropicLLMContext
            obj.__setup_local()
            obj._restructure_from_openai_messages()
        return obj

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        """Create Anthropic context from OpenAI context.

        Args:
            openai_context: The OpenAI context to convert.

        Returns:
            New Anthropic context with converted messages.
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
    def from_messages(cls, messages: List[dict]) -> "AnthropicLLMContext":
        """Create context from a list of messages.

        Args:
            messages: List of conversation messages.

        Returns:
            New Anthropic context with the provided messages.
        """
        self = cls(messages=messages)
        self._restructure_from_openai_messages()
        return self

    def set_messages(self, messages: List):
        """Set the messages list and reset cache tracking.

        Args:
            messages: New list of messages to set.
        """
        self.turns_above_cache_threshold = 0
        self._messages[:] = messages
        self._restructure_from_openai_messages()

    def to_standard_messages(self, obj):
        """Convert Anthropic message format to standard structured format.

        Handles text content and function calls for both user and assistant messages.

        Args:
            obj: Message in Anthropic format.

        Returns:
            List of messages in standard format.

        Examples:
            Input Anthropic format::

                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "tool_use", "id": "123", "name": "search", "input": {"q": "test"}}
                    ]
                }

            Output standard format::

                [
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "123",
                                "function": {"name": "search", "arguments": '{"q": "test"}'}
                            }
                        ]
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
            message: Message in standard format.

        Returns:
            Message in Anthropic format.

        Examples:
            Input standard format::

                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "123",
                            "function": {"name": "search", "arguments": '{"q": "test"}'}
                        }
                    ]
                }

            Output Anthropic format::

                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "123",
                            "name": "search",
                            "input": {"q": "test"}
                        }
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
        """Add an image message to the context.

        Converts the image to base64 JPEG format and adds it as a user message
        with optional accompanying text.

        Args:
            format: The image format (e.g., 'RGB', 'RGBA').
            size: Image dimensions as (width, height).
            image: Raw image bytes.
            text: Optional text to accompany the image.
        """
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
        """Add a message to the context, merging with previous message if same role.

        Anthropic requires alternating roles, so consecutive messages from the same
        role are merged together.

        Args:
            message: The message to add to the context.
        """
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
        """Get messages with prompt caching markers applied.

        Adds cache control markers to appropriate messages based on the
        number of turns above the cache threshold.

        Returns:
            List of messages with cache control markers added.
        """
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
        """Get messages formatted for persistent storage.

        Includes system message at the beginning if present.

        Returns:
            List of messages suitable for storage.
        """
        messages = super().get_messages_for_persistent_storage()
        if self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        return messages

    def get_messages_for_logging(self) -> List[Dict[str, Any]]:
        """Get messages formatted for logging with sensitive data redacted.

        Replaces image data with placeholder text for cleaner logs.

        Returns:
            List of messages in a format ready for logging.
        """
        msgs = []
        for message in self.messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "image":
                            item["source"]["data"] = "..."
            msgs.append(msg)
        return msgs


class AnthropicUserContextAggregator(LLMUserContextAggregator):
    """Anthropic-specific user context aggregator.

    Handles aggregation of user messages for Anthropic LLM services.
    Inherits all functionality from the base LLMUserContextAggregator.
    """

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
    """Context aggregator for assistant messages in Anthropic conversations.

    Handles function call lifecycle management including in-progress tracking,
    result handling, and cancellation for Anthropic's tool use format.
    """

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle a function call that is starting.

        Creates tool use message and placeholder tool result for tracking.

        Args:
            frame: Frame containing function call details.
        """
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
        """Handle the result of a completed function call.

        Updates the tool result with actual return value or completion status.

        Args:
            frame: Frame containing function call result.
        """
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle cancellation of a function call.

        Updates the tool result to indicate cancellation.

        Args:
            frame: Frame containing function call cancellation details.
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
                        and content["type"] == "tool_result"
                        and content["tool_use_id"] == tool_call_id
                    ):
                        content["content"] = result

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle a user image frame with function call context.

        Marks the associated function call as completed and adds the image
        to the conversation context.

        Args:
            frame: User image frame with request context.
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
