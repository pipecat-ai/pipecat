#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import io
import copy
from typing import List
from dataclasses import dataclass
from PIL import Image

from pipecat.frames.frames import (
    Frame,
    LLMModelUpdateFrame,
    TextFrame,
    VisionImageRawFrame,
    LLMMessagesFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator

from loguru import logger

try:
    from anthropic import AsyncAnthropic
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Anthropic, you need to `pip install pipecat-ai[anthropic]`. Also, set `ANTHROPIC_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AnthropicToolUseFrame(Frame):
    tool_id: str
    tool_name: str
    tool_input: dict
    result_content: str | list
    llm: 'AnthropicLLMService'


@dataclass
class AnthropicContextAggregatorPair:
    _user: 'AnthropicUserContextAggregator'
    _assistant: 'AnthropicAssistantContextAggregator'

    def user(self) -> str:
        return self._user

    def assistant(self) -> str:
        return self._assistant


class AnthropicLLMService(LLMService):
    """This class implements inference with Anthropic's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Anthropic Python SDK. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    def __init__(
            self,
            *,
            api_key: str,
            model: str = "claude-3-5-sonnet-20240620",
            max_tokens: int = 4096):
        super().__init__()
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def can_generate_metrics(self) -> bool:
        return True

    @ staticmethod
    def create_context_aggregator(context: OpenAILLMContext) -> AnthropicContextAggregatorPair:
        user = AnthropicUserContextAggregator(context)
        assistant = AnthropicAssistantContextAggregator(user._context)
        return AnthropicContextAggregatorPair(
            _user=user,
            _assistant=assistant
        )

    async def _handle_function_call(
        self, context, tool_call_id, function_name, arguments
    ):
        arguments_obj = json.loads(arguments)
        result = await self.call_function(function_name, arguments_obj)

        if isinstance(result, type(None)):
            pass
        elif hasattr(result, 'is_error'):
            await self.push_frame(AnthropicToolUseFrame(
                tool_id=tool_call_id,
                tool_name=function_name,
                tool_input=arguments_obj,
                result_content=result.content,
                is_error=result.is_error,
                llm=self
            ))
        elif isinstance(result, str) or isinstance(result, list):
            await self.push_frame(AnthropicToolUseFrame(
                tool_id=tool_call_id,
                tool_name=function_name,
                tool_input=arguments_obj,
                result_content=result,
                llm=self
            ))
        else:
            raise TypeError(f"Unknown return type from function callback: {type(result)}")

    async def _process_context(self, context: OpenAILLMContext):
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()

            logger.debug(f"Generating chat: {context.get_messages_for_logging()}")

            messages = context.messages

            await self.start_ttfb_metrics()

            response = await self._client.messages.create(
                messages=messages,
                tools=context.tools or [],
                model=self._model,
                max_tokens=self._max_tokens,
                stream=True)

            await self.stop_ttfb_metrics()

            # for tool use
            tool_use_block = None
            json_accumulator = ''

            async for event in response:
                # logger.debug(f"Anthropic LLM event: {event}")
                if (event.type == "content_block_delta"):
                    if hasattr(event.delta, 'text'):
                        await self.push_frame(TextFrame(event.delta.text))
                    elif hasattr(event.delta, 'partial_json') and tool_use_block:
                        json_accumulator += event.delta.partial_json
                elif (event.type == "content_block_start"):
                    if event.content_block.type == "tool_use":
                        tool_use_block = event.content_block
                        json_accumulator = ''
                elif (event.type == "message_delta" and
                      hasattr(event.delta, 'stop_reason') and event.delta.stop_reason == 'tool_use'):
                    if tool_use_block:
                        await self._handle_function_call(
                            context, tool_use_block.id, tool_use_block.name, json_accumulator)

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            logger.debug(f"LLM context from: {frame.context}")
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            logger.debug(f"LLM context from messages: {frame.messages}")
            context = AnthropicLLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            logger.debug(f"LLM context from image frame")
            context = AnthropicLLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMModelUpdateFrame):
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self._model = frame.model
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)


class AnthropicLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        *,
        system: str | None = None
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self.system_message = system

    @ classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        self = cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )
        # See if we should pull the system message out of our context.messages list. (For
        # compatibility with Open AI messages format.)
        if self.messages and self.messages[0]["role"] == "system":
            logger.debug(f"Pulling system message from context.")
            if len(self.messages) == 1:
                logger.debug(f"Only system message in context.")
                # If we have only have a system message in the list, all we can really do
                # without introducing too much magic is change the role to "user".
                self.messages[0]["role"] = "user"
            else:
                # If we have more than one message, we'll pull the system message out of the
                # list.
                self.system_message = self.messages[0]["content"]
                self.messages.pop(0)
                logger.debug(f"Messages: {self.messages}")
        return self

    @ classmethod
    def from_messages(cls, messages: List[dict]) -> "AnthropicLLMContext":
        return cls(messages=messages)

    @ classmethod
    def from_image_frame(cls, frame: VisionImageRawFrame) -> "AnthropicLLMContext":
        context = cls()
        context.add_image_frame_message(frame)
        return context

    def add_image_frame_message(self, frame: VisionImageRawFrame):
        buffer = io.BytesIO()
        Image.frombytes(
            frame.format,
            frame.size,
            frame.image
        ).save(
            buffer,
            format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug(f"Encoded image length: {type(encoded_image)} {len(encoded_image)}")
        # Anthropic docs say that the image should be the first content block in the message.
        content = [{"type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded_image,
                    }}]
        if frame.text:
            content.append({"type": "text", "text": frame.text})
        self.messages.append({"role": "user", "content": content})

    def get_messages_for_logging(self) -> str:
        msgs = []
        for message in self.messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    if len(msg["content"]) > 1 and msg["content"][0]["type"] == "image":
                        msg["content"][0]["source"]["data"] = "..."
            msgs.append(msg)
        return json.dumps(msgs)


class AnthropicUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext | AnthropicLLMContext):
        logger.debug(f"AnthropicUserContextAggregator init")
        super().__init__(context=context)

        if isinstance(context, OpenAILLMContext):
            logger.debug("upcycling OpenAILLMContext to AnthropicLLMContext")
            self._context = AnthropicLLMContext.from_openai_context(context)


#
# Claude returns a text content block along with a tool use content block. This works quite nicely
# with streaming. We get the text first, so we can start streaming it right away. Then we get the
# tool_use block. While the text is streaming to TTS and the transport, we can run the tool call.
#
# The tricky thing is that we need to append the tool use result to the context, after the initial
# text content is entirely streamed. Then we need to re-run LLM inference with the updated context.
#
# We manage that flow by sending an AnthropicToolUseFrame from the AnthropicLLMService when the LLM
# emits a tool_use block. This AnthropicAssistantContextAggregator catches that frame and caches it.
# Then, when we see an LLMFullResponseEndFrame, the AnthropicAssistantContextAggregator:
#   1. appends the tool_use content block to the last message in the context. For this to happen
#      properly, we're subclassing LLMAssistantContextAggregator and running our logic here
#      immediately after our super class appends the standard text content message.
#   2. appends the tool_result as a new "user" message.
#   3. runs the LLM inference again with the updated context.
#


class AnthropicAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, context: AnthropicLLMContext):
        super().__init__(context=context)
        self.tool_use_frame = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, AnthropicToolUseFrame):
            self.tool_use_frame = frame

    async def _push_aggregation(self):
        await super()._push_aggregation()
        try:
            if self.tool_use_frame:
                tuf = self.tool_use_frame
                self.tool_use_frame = None

                if self._context.messages[-1]["role"] == "assistant" and "content" in self._context.messages[-1]:
                    if isinstance(self._context.messages[-1]["content"], str):
                        self._context.messages[-1]["content"] = [
                            {"type": "text", "text": self._context.messages[-1]["content"]}]
                    else:
                        raise Exception(
                            f"Last message content type is not str. Need to implement for this case. {self._context.messages}")
                    self._context.messages[-1]["content"].append({
                        "type": "tool_use",
                        "id": tuf.tool_id,
                        "name": tuf.tool_name,
                        "input": tuf.tool_input,
                    })
                    self._context.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tuf.tool_id,
                                "content": tuf.result_content,
                            }
                        ]
                    })
                    await tuf.llm._process_context(self._context)
                else:
                    logger.error(
                        "Expected last message to be an assistant message with content block, but it wasn't.")
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
