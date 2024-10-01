#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import copy
import io
import json
import re
from asyncio import CancelledError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMEnablePromptCachingFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    StartInterruptionFrame,
    TextFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService

try:
    from anthropic import NOT_GIVEN, AsyncAnthropic, NotGiven
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Anthropic, you need to `pip install pipecat-ai[anthropic]`. "
        + "Also, set `ANTHROPIC_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


# internal use only -- todo: refactor
@dataclass
class AnthropicImageMessageFrame(Frame):
    user_image_raw_frame: UserImageRawFrame
    text: Optional[str] = None


@dataclass
class AnthropicContextAggregatorPair:
    _user: "AnthropicUserContextAggregator"
    _assistant: "AnthropicAssistantContextAggregator"

    def user(self) -> "AnthropicUserContextAggregator":
        return self._user

    def assistant(self) -> "AnthropicAssistantContextAggregator":
        return self._assistant


class AnthropicLLMService(LLMService):
    """This class implements inference with Anthropic's AI models"""

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
        model: str = "claude-3-5-sonnet-20240620",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = AsyncAnthropic(api_key=api_key)
        self.set_model_name(model)
        self._max_tokens = params.max_tokens
        self._enable_prompt_caching_beta: bool = params.enable_prompt_caching_beta or False
        self._temperature = params.temperature
        self._top_k = params.top_k
        self._top_p = params.top_p
        self._extra = params.extra if isinstance(params.extra, dict) else {}

    def can_generate_metrics(self) -> bool:
        return True

    @property
    def enable_prompt_caching_beta(self) -> bool:
        return self._enable_prompt_caching_beta

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> AnthropicContextAggregatorPair:
        user = AnthropicUserContextAggregator(context)
        assistant = AnthropicAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return AnthropicContextAggregatorPair(_user=user, _assistant=assistant)

    async def set_enable_prompt_caching_beta(self, enable_prompt_caching_beta: bool):
        logger.debug(f"Switching LLM enable_prompt_caching_beta to: [{enable_prompt_caching_beta}]")
        self._enable_prompt_caching_beta = enable_prompt_caching_beta

    async def set_max_tokens(self, max_tokens: int):
        logger.debug(f"Switching LLM max_tokens to: [{max_tokens}]")
        self._max_tokens = max_tokens

    async def set_temperature(self, temperature: float):
        logger.debug(f"Switching LLM temperature to: [{temperature}]")
        self._temperature = temperature

    async def set_top_k(self, top_k: float):
        logger.debug(f"Switching LLM top_k to: [{top_k}]")
        self._top_k = top_k

    async def set_top_p(self, top_p: float):
        logger.debug(f"Switching LLM top_p to: [{top_p}]")
        self._top_p = top_p

    async def set_extra(self, extra: Dict[str, Any]):
        logger.debug(f"Switching LLM extra to: [{extra}]")
        self._extra = extra

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
                f"Generating chat: {context.system} | {context.get_messages_for_logging()}"
            )

            messages = context.messages
            if self._enable_prompt_caching_beta:
                messages = context.get_messages_with_cache_control_markers()

            api_call = self._client.messages.create
            if self._enable_prompt_caching_beta:
                api_call = self._client.beta.prompt_caching.messages.create

            await self.start_ttfb_metrics()

            params = {
                "tools": context.tools or [],
                "system": context.system,
                "messages": messages,
                "model": self.model_name,
                "max_tokens": self._max_tokens,
                "stream": True,
                "temperature": self._temperature,
                "top_k": self._top_k,
                "top_p": self._top_p,
            }

            params.update(self._extra)

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
                        await self.push_frame(TextFrame(event.delta.text))
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
                    if hasattr(event.message.usage, "cache_creation_input_tokens"):
                        cache_creation_input_tokens += (
                            event.message.usage.cache_creation_input_tokens
                        )
                        logger.debug(f"Cache creation input tokens: {cache_creation_input_tokens}")
                    if hasattr(event.message.usage, "cache_read_input_tokens"):
                        cache_read_input_tokens += event.message.usage.cache_read_input_tokens
                        logger.debug(f"Cache read input tokens: {cache_read_input_tokens}")
                    total_input_tokens = (
                        prompt_tokens + cache_creation_input_tokens + cache_read_input_tokens
                    )
                    if total_input_tokens >= 1024:
                        context.turns_above_cache_threshold += 1

        except CancelledError:
            # If we're interrupted, we won't get a complete usage report. So set our flag to use the
            # token estimate. The reraise the exception so all the processors running in this task
            # also get cancelled.
            use_completion_tokens_estimate = True
            raise
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

    async def _update_settings(self, frame: LLMUpdateSettingsFrame):
        if frame.model is not None:
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self.set_model_name(frame.model)
        if frame.max_tokens is not None:
            await self.set_max_tokens(frame.max_tokens)
        if frame.temperature is not None:
            await self.set_temperature(frame.temperature)
        if frame.top_k is not None:
            await self.set_top_k(frame.top_k)
        if frame.top_p is not None:
            await self.set_top_p(frame.top_p)
        if frame.extra:
            await self.set_extra(frame.extra)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = AnthropicLLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            # This is only useful in very simple pipelines because it creates
            # a new context. Generally we want a context manager to catch
            # UserImageRawFrames coming through the pipeline and add them
            # to the context.
            context = AnthropicLLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame)
        elif isinstance(frame, LLMEnablePromptCachingFrame):
            logger.debug(f"Setting enable prompt caching to: [{frame.enable}]")
            self._enable_prompt_caching_beta = frame.enable
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
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        *,
        system: str | NotGiven = NOT_GIVEN,
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)

        # For beta prompt caching. This is a counter that tracks the number of turns
        # we've seen above the cache threshold. We reset this when we reset the
        # messages list. We only care about this number being 0, 1, or 2. But
        # it's easiest just to treat it as a counter.
        self.turns_above_cache_threshold = 0

        self.system = system

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        self = cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )
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
    def __init__(self, context: OpenAILLMContext | AnthropicLLMContext):
        super().__init__(context=context)

        if isinstance(context, OpenAILLMContext):
            self._context = AnthropicLLMContext.from_openai_context(context)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # Our parent method has already called push_frame(). So we can't interrupt the
        # flow here and we don't need to call push_frame() ourselves. Possibly something
        # to talk through (tagging @aleix). At some point we might need to refactor these
        # context aggregators.
        try:
            if isinstance(frame, UserImageRequestFrame):
                # The LLM sends a UserImageRequestFrame upstream. Cache any context provided with
                # that frame so we can use it when we assemble the image message in the assistant
                # context aggregator.
                if frame.context:
                    if isinstance(frame.context, str):
                        self._context._user_image_request_context[frame.user_id] = frame.context
                    else:
                        logger.error(
                            f"Unexpected UserImageRequestFrame context type: {type(frame.context)}"
                        )
                        del self._context._user_image_request_context[frame.user_id]
                else:
                    if frame.user_id in self._context._user_image_request_context:
                        del self._context._user_image_request_context[frame.user_id]
            elif isinstance(frame, UserImageRawFrame):
                # Push a new AnthropicImageMessageFrame with the text context we cached
                # downstream to be handled by our assistant context aggregator. This is
                # necessary so that we add the message to the context in the right order.
                text = self._context._user_image_request_context.get(frame.user_id) or ""
                if text:
                    del self._context._user_image_request_context[frame.user_id]
                frame = AnthropicImageMessageFrame(user_image_raw_frame=frame, text=text)
                await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")


#
# Claude returns a text content block along with a tool use content block. This works quite nicely
# with streaming. We get the text first, so we can start streaming it right away. Then we get the
# tool_use block. While the text is streaming to TTS and the transport, we can run the tool call.
#
# But Claude is verbose. It would be nice to come up with prompt language that suppresses Claude's
# chattiness about it's tool thinking.
#


class AnthropicAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: AnthropicUserContextAggregator, **kwargs):
        super().__init__(context=user_context_aggregator._context, **kwargs)
        self._user_context_aggregator = user_context_aggregator
        self._function_call_in_progress = None
        self._function_call_result = None
        self._pending_image_frame_message = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # See note above about not calling push_frame() here.
        if isinstance(frame, StartInterruptionFrame):
            self._function_call_in_progress = None
            self._function_call_finished = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._function_call_in_progress = frame
        elif isinstance(frame, FunctionCallResultFrame):
            if (
                self._function_call_in_progress
                and self._function_call_in_progress.tool_call_id == frame.tool_call_id
            ):
                self._function_call_in_progress = None
                self._function_call_result = frame
            else:
                logger.warning(
                    "FunctionCallResultFrame tool_call_id != InProgressFrame tool_call_id"
                )
                self._function_call_in_progress = None
                self._function_call_result = None
        elif isinstance(frame, AnthropicImageMessageFrame):
            self._pending_image_frame_message = frame

    async def _push_aggregation(self):
        if not self._aggregation:
            return

        run_llm = False

        aggregation = self._aggregation
        self._reset()

        try:
            if self._function_call_result:
                frame = self._function_call_result
                self._function_call_result = None
                if frame.result:
                    self._context.add_message(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": aggregation},
                                {
                                    "type": "tool_use",
                                    "id": frame.tool_call_id,
                                    "name": frame.function_name,
                                    "input": frame.arguments,
                                },
                            ],
                        }
                    )
                    self._context.add_message(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": frame.tool_call_id,
                                    "content": json.dumps(frame.result),
                                }
                            ],
                        }
                    )
                    run_llm = True
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if self._pending_image_frame_message:
                frame = self._pending_image_frame_message
                self._pending_image_frame_message = None
                self._context.add_image_frame_message(
                    format=frame.user_image_raw_frame.format,
                    size=frame.user_image_raw_frame.size,
                    image=frame.user_image_raw_frame.image,
                    text=frame.text,
                )
                run_llm = True

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
