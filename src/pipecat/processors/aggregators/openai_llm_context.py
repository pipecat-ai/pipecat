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
    AudioRawFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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
        """Convert from OpenAI message format to OpenAI message format (passthrough).

        OpenAI's format allows both simple string content and structured content:
        - Simple: {"role": "user", "content": "Hello"}
        - Structured: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

        Since OpenAI is our standard format, this is a passthrough function.

        Args:
            message (dict): Message in OpenAI format

        Returns:
            dict: Same message, unchanged
        """
        return message

    def to_standard_messages(self, obj) -> list:
        """Convert from OpenAI message format to OpenAI message format (passthrough).

        OpenAI's format is our standard format throughout Pipecat. This function
        returns a list containing the original message to maintain consistency with
        other LLM services that may need to return multiple messages.

        Args:
            obj (dict): Message in OpenAI format with either:
                - Simple content: {"role": "user", "content": "Hello"}
                - List content: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

        Returns:
            list: List containing the original messages, preserving whether
                the content was in simple string or structured list format
        """
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

        content = []
        if text:
            content.append({"type": "text", "text": text})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        )
        self.add_message({"role": "user", "content": content})

    def add_audio_frames_message(self, *, audio_frames: list[AudioRawFrame], text: str = None):
        # todo: implement for OpenAI models and others
        pass

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
        # Also push a SystemFrame upstream for use by other processors, like STTMuteFilter.
        progress_frame_downstream = FunctionCallInProgressFrame(
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )
        progress_frame_upstream = FunctionCallInProgressFrame(
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
        )

        # Push frame both downstream and upstream
        await llm.push_frame(progress_frame_downstream, FrameDirection.DOWNSTREAM)
        await llm.push_frame(progress_frame_upstream, FrameDirection.UPSTREAM)

        # Define a callback function that pushes a FunctionCallResultFrame upstream & downstream.
        async def function_call_result_callback(result):
            result_frame_downstream = FunctionCallResultFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=result,
                run_llm=run_llm,
            )
            result_frame_upstream = FunctionCallResultFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=result,
                run_llm=run_llm,
            )

            # Push frame both downstream and upstream
            await llm.push_frame(result_frame_downstream, FrameDirection.DOWNSTREAM)
            await llm.push_frame(result_frame_upstream, FrameDirection.UPSTREAM)

        await f(function_name, tool_call_id, arguments, llm, self, function_call_result_callback)

    def create_wav_header(self, sample_rate, num_channels, bits_per_sample, data_size):
        # RIFF chunk descriptor
        header = bytearray()
        header.extend(b"RIFF")  # ChunkID
        header.extend((data_size + 36).to_bytes(4, "little"))  # ChunkSize: total size - 8
        header.extend(b"WAVE")  # Format
        # "fmt " sub-chunk
        header.extend(b"fmt ")  # Subchunk1ID
        header.extend((16).to_bytes(4, "little"))  # Subchunk1Size (16 for PCM)
        header.extend((1).to_bytes(2, "little"))  # AudioFormat (1 for PCM)
        header.extend(num_channels.to_bytes(2, "little"))  # NumChannels
        header.extend(sample_rate.to_bytes(4, "little"))  # SampleRate
        # Calculate byte rate and block align
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)
        header.extend(byte_rate.to_bytes(4, "little"))  # ByteRate
        header.extend(block_align.to_bytes(2, "little"))  # BlockAlign
        header.extend(bits_per_sample.to_bytes(2, "little"))  # BitsPerSample
        # "data" sub-chunk
        header.extend(b"data")  # Subchunk2ID
        header.extend(data_size.to_bytes(4, "little"))  # Subchunk2Size
        return header


@dataclass
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """

    context: OpenAILLMContext
