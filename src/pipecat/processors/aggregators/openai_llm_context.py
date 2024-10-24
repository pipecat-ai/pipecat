#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import copy
import io
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List

from loguru import logger
from PIL import Image

from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    VisionImageRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

try:
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolChoiceOptionParam,
        ChatCompletionToolParam,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

# JSON custom encoder to handle bytes arrays so that we can log contexts
# with images to the console.


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, io.BytesIO):
            # Convert the first 8 bytes to an ASCII hex string
            return f"{obj.getbuffer()[0:8].hex()}..."
        return super().default(obj)


class OpenAILLMContext:
    def __init__(
        self,
        messages: List[ChatCompletionMessageParam] | None = None,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    ):
        self._messages: List[ChatCompletionMessageParam] = messages if messages else []
        self._tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self._tools: List[ChatCompletionToolParam] | NotGiven = tools
        self._user_image_request_context = {}

    @staticmethod
    def from_messages(messages: List[dict]) -> "OpenAILLMContext":
        context = OpenAILLMContext()

        for message in messages:
            if "name" not in message:
                message["name"] = message["role"]
            context.add_message(message)
        return context

    # todo: deprecate from_image_frame. It's only used to create a single-use
    # context, which isn't useful for most real-world applications.
    @staticmethod
    def from_image_frame(frame: VisionImageRawFrame) -> "OpenAILLMContext":
        """
        For images, we are deviating from the OpenAI messages shape. OpenAI
        expects images to be base64 encoded, but other vision models may not.
        So we'll store the image as bytes and do the base64 encoding as needed
        in the LLM service.

        NOTE: the above only applies to the deprecated use of this method. The
        add_image_frame_message() below does the base64 encoding as expected
        in the OpenAI format.
        """
        context = OpenAILLMContext()
        buffer = io.BytesIO()
        Image.frombytes(frame.format, frame.size, frame.image).save(buffer, format="JPEG")
        context.add_message(
            {"content": frame.text, "role": "user", "data": buffer, "mime_type": "image/jpeg"}
        )
        return context

    @property
    def messages(self) -> List[ChatCompletionMessageParam]:
        return self._messages

    @property
    def tools(self) -> List[ChatCompletionToolParam] | NotGiven:
        return self._tools

    @property
    def tool_choice(self) -> ChatCompletionToolChoiceOptionParam | NotGiven:
        return self._tool_choice

    def add_message(self, message: ChatCompletionMessageParam):
        self._messages.append(message)

    def add_messages(self, messages: List[ChatCompletionMessageParam]):
        self._messages.extend(messages)

    def set_messages(self, messages: List[ChatCompletionMessageParam]):
        self._messages[:] = messages

    def get_messages(self) -> List[ChatCompletionMessageParam]:
        return self._messages

    def get_messages_json(self) -> str:
        return json.dumps(self._messages, cls=CustomEncoder, ensure_ascii=False, indent=2)

    def get_messages_for_logging(self) -> str:
        msgs = []
        for message in self.messages:
            msg = copy.deepcopy(message)
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "image_url":
                            if item["image_url"]["url"].startswith("data:image/"):
                                item["image_url"]["url"] = "data:image/..."
            if "mime_type" in msg and msg["mime_type"].startswith("image/"):
                msg["data"] = "..."
            msgs.append(msg)
        return json.dumps(msgs)

    def from_standard_message(self, message):
        return message

    # convert a message in this LLM's format to one or more messages in OpenAI format
    def to_standard_messages(self, obj) -> list:
        return [obj]

    def get_messages_for_initializing_history(self):
        return self._messages

    def get_messages_for_persistent_storage(self):
        messages = []
        for m in self._messages:
            standard_messages = self.to_standard_messages(m)
            messages.extend(standard_messages)
        return messages

    def set_tool_choice(self, tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven):
        self._tool_choice = tool_choice

    def set_tools(self, tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN):
        if tools != NOT_GIVEN and len(tools) == 0:
            tools = NOT_GIVEN
        self._tools = tools

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        ]
        if text:
            content.append({"type": "text", "text": text})
        self.add_message({"role": "user", "content": content})

    async def call_function(
        self,
        f: Callable[
            [str, str, Any, FrameProcessor, "OpenAILLMContext", Callable[[Any], Awaitable[None]]],
            Awaitable[None],
        ],
        *,
        function_name: str,
        tool_call_id: str,
        arguments: str,
        llm: FrameProcessor,
        run_llm: bool = True,
    ) -> None:
        logger.info(f"Calling function {function_name} with arguments {arguments}")
        # Push a SystemFrame downstream. This frame will let our assistant context aggregator
        # know that we are in the middle of a function call. Some contexts/aggregators may
        # not need this. But some definitely do (Anthropic, for example).
        await llm.push_frame(
            FunctionCallInProgressFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
            )
        )

        # Define a callback function that pushes a FunctionCallResultFrame downstream.
        async def function_call_result_callback(result):
            await llm.push_frame(
                FunctionCallResultFrame(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    result=result,
                    run_llm=run_llm,
                )
            )

        await f(function_name, tool_call_id, arguments, llm, self, function_call_result_callback)


@dataclass
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """

    context: OpenAILLMContext
