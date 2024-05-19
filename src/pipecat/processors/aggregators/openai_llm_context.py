#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass
import io
import json

from typing import List

from PIL import Image

from pipecat.frames.frames import Frame, VisionImageRawFrame

from openai._types import NOT_GIVEN, NotGiven

from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessageParam
)

# JSON custom encoder to handle bytes arrays so that we can log contexts
# with images to the console.


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, io.BytesIO):
            # Convert the first 8 bytes to an ASCII hex string
            return (f"{obj.getbuffer()[0:8].hex()}...")
        return super().default(obj)


class OpenAILLMContext:

    def __init__(
        self,
        messages: List[ChatCompletionMessageParam] | None = None,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    ):
        self.messages: List[ChatCompletionMessageParam] = messages if messages else [
        ]
        self.tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self.tools: List[ChatCompletionToolParam] | NotGiven = tools

    @staticmethod
    def from_messages(messages: List[dict]) -> "OpenAILLMContext":
        context = OpenAILLMContext()
        for message in messages:
            context.add_message({
                "content": message["content"],
                "role": message["role"],
                "name": message["name"] if "name" in message else message["role"]
            })
        return context

    @staticmethod
    def from_image_frame(frame: VisionImageRawFrame) -> "OpenAILLMContext":
        """
        For images, we are deviating from the OpenAI messages shape. OpenAI
        expects images to be base64 encoded, but other vision models may not.
        So we'll store the image as bytes and do the base64 encoding as needed
        in the LLM service.
        """
        context = OpenAILLMContext()
        buffer = io.BytesIO()
        Image.frombytes(
            frame.format,
            frame.size,
            frame.image
        ).save(
            buffer,
            format="JPEG")
        context.add_message({
            "content": frame.text,
            "role": "user",
            "data": buffer,
            "mime_type": "image/jpeg"
        })
        return context

    def add_message(self, message: ChatCompletionMessageParam):
        self.messages.append(message)

    def get_messages(self) -> List[ChatCompletionMessageParam]:
        return self.messages

    def get_messages_json(self) -> str:
        return json.dumps(self.messages, cls=CustomEncoder)

    def set_tool_choice(
        self, tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    ):
        self.tool_choice = tool_choice

    def set_tools(self, tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN):
        if tools != NOT_GIVEN and len(tools) == 0:
            tools = NOT_GIVEN

        self.tools = tools


@dataclass
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """
    context: OpenAILLMContext
