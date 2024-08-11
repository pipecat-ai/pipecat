#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import io
import copy
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image

from pipecat.frames.frames import (
    Frame,
    LLMModelUpdateFrame,
    TextFrame,
    VisionImageRawFrame,
    UserImageRequestFrame,
    UserImageRawFrame,
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
class AnthropicImageMessageFrame(Frame):
    user_image_raw_frame: UserImageRawFrame
    text: Optional[str] = None


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
        assistant = AnthropicAssistantContextAggregator(user)
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
            # This is only useful in very simple pipelines because it creates
            # a new context. Generally we want a context manager to catch
            # UserImageRawFrames coming through the pipeline and add them
            # to the context.
            logger.debug(f"LLM context from image frame")
            context = AnthropicLLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMModelUpdateFrame):
            logger.debug(f"Switching LLM model to: [{frame.model}]")
            self._model = frame.model
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    async def request_image_frame(self, user_id: str, *, text_content: str = None):
        await self.push_frame(UserImageRequestFrame(user_id=user_id, context=text_content), FrameDirection.UPSTREAM)


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
        self._user_image_request_context = {}

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
        context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.text)
        return context

    def add_image_frame_message(
            self, *, format: str, size: tuple[int, int], image: bytes, text: str = None):
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        logger.debug(f"Encoded image length: {type(encoded_image)} {len(encoded_image)}")
        # Anthropic docs say that the image should be the first content block in the message.
        content = [{"type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded_image,
                    }}]
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
                        self.messages[-1]["content"] = [{"type": "text",
                                                         "text": self.messages[-1]["content"]}]
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
            logger.debug("upcycling OpenAILLMContext to AnthropicLLMContext")
            self._context = AnthropicLLMContext.from_openai_context(context)

    async def push_messages_frame(self):
        frame = OpenAILLMContextFrame(self._context)
        await self.push_frame(frame)

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
                if (frame.context):
                    if isinstance(frame.context, str):
                        self._context._user_image_request_context[frame.user_id] = frame.context
                    else:
                        logger.error(
                            f"Unexpected UserImageRequestFrame context type: {type(frame.context)}")
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
    def __init__(self, user_context_aggregator: AnthropicUserContextAggregator):
        super().__init__(context=user_context_aggregator._context)
        self._tool_use_frame = None
        self._user_context_aggregator = user_context_aggregator

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # See note above about not calling push_frame() here.
        if isinstance(frame, AnthropicToolUseFrame):
            self._tool_use_frame = frame
        elif isinstance(frame, AnthropicImageMessageFrame):
            try:
                self._context.add_image_frame_message(
                    format=frame.user_image_raw_frame.format,
                    size=frame.user_image_raw_frame.size,
                    image=frame.user_image_raw_frame.image,
                    text=frame.text)
                await self._user_context_aggregator.push_messages_frame()
            except Exception as e:
                logger.error(f"Error processing UserImageRawFrame: {e}")

    def add_message(self, message):
        self._user_context_aggregator.add_message(message)

    async def _push_aggregation(self):
        await super()._push_aggregation()
        try:
            if self._tool_use_frame:
                tuf = self._tool_use_frame
                self._tool_use_frame = None

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
                    await self._user_context_aggregator.push_messages_frame()
                else:
                    logger.error(
                        "Expected last message to be an assistant message with content block, but it wasn't.")
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
