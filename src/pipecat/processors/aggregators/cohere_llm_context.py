#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from collections.abc import Sequence
import io
import json

from dataclasses import dataclass

from typing import Any, Awaitable, Callable, List

from PIL import Image

from pipecat.frames.frames import Frame, VisionImageRawFrame, FunctionCallInProgressFrame, FunctionCallResultFrame
from pipecat.processors.frame_processor import FrameProcessor

from loguru import logger

try:
    from cohere import Message, Tool
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Cohere, you need to `pip install pipecat-ai[cohere]`. Also, set `COHERE_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")

# JSON custom encoder to handle bytes arrays so that we can log contexts
# with images to the console.


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, io.BytesIO):
            # Convert the first 8 bytes to an ASCII hex string
            return (f"{obj.getbuffer()[0:8].hex()}...")
        return super().default(obj)


class CohereLLMContext:

    def __init__(
        self,
        messages: Sequence[Message] | None = None,
        tools: Sequence[Tool] | None = None,
        # tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN # Maybe not needed
    ):
        self._chat_history: Sequence[Message] = messages if messages else [
        ]
        self._tools: Sequence[Tool] | None = tools

    @staticmethod
    def from_messages(messages: List[dict]) -> "CohereLLMContext":
        context = CohereLLMContext()

        for message in messages:
            if "name" not in message:
                message["name"] = message["role"]
            context.add_message(message)
        return context

    @staticmethod
    def from_image_frame(frame: VisionImageRawFrame) -> "CohereLLMContext":
        """
        For images, we are deviating from the Cohere messages shape. Cohere
        expects images to be base64 encoded, but other vision models may not.
        So we'll store the image as bytes and do the base64 encoding as needed
        in the LLM service.
        """
        context = CohereLLMContext()
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

    @property
    def messages(self) -> Sequence[Message]:
        return self._chat_history

    @property
    def tools(self) -> Sequence[Tool] | None:
        return self._tools

    def add_message(self, message: Message):
        self._chat_history.append(message)

    def add_messages(self, messages: Sequence[Message]):
        self._chat_history.extend(messages)

    def set_messages(self, messages: Sequence[Message]):
        self._chat_history[:] = messages

    def get_messages(self) -> Sequence[Message]:
        return self._chat_history

    def get_messages_json(self) -> str:
        return json.dumps(self._chat_history, cls=CustomEncoder)

    def set_tools(self, tools: Sequence[Tool] | None):
        if tools is not None and len(tools) == 0:
            tools = None
        self._tools = tools

    async def call_function(self,
                            f: Callable[[str,
                                         str,
                                         Any,
                                         FrameProcessor,
                                         'CohereLLMContext',
                                         Callable[[Any],
                                                  Awaitable[None]]],
                                        Awaitable[None]],
                            *,
                            function_name: str,
                            tool_call_id: str,
                            arguments: str,
                            llm: FrameProcessor) -> None:

        # Push a SystemFrame downstream. This frame will let our assistant context aggregator
        # know that we are in the middle of a function call. Some contexts/aggregators may
        # not need this. But some definitely do (Anthropic, for example).
        await llm.push_frame(FunctionCallInProgressFrame(
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        ))

        # Define a callback function that pushes a FunctionCallResultFrame downstream.
        async def function_call_result_callback(result):
            await llm.push_frame(FunctionCallResultFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=result))
        await f(function_name, tool_call_id, arguments, llm, self, function_call_result_callback)


@dataclass
class CohereLLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the Cohere
    API. The context in this message is also mutable, and will be changed by the
    CohereContextAggregator frame processor.

    """
    context: CohereLLMContext
