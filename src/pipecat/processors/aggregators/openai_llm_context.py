#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import copy
import io
import json
from dataclasses import dataclass
from typing import Any, List, Optional

from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from PIL import Image

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    ):
        self._messages: List[ChatCompletionMessageParam] = messages if messages else []
        self._tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self._tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = tools
        self._llm_adapter: Optional[BaseLLMAdapter] = None

    def get_llm_adapter(self) -> Optional[BaseLLMAdapter]:
        return self._llm_adapter

    def set_llm_adapter(self, llm_adapter: BaseLLMAdapter):
        self._llm_adapter = llm_adapter

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
    def tools(self) -> List[ChatCompletionToolParam] | NotGiven | List[Any]:
        if self._llm_adapter:
            return self._llm_adapter.from_standard_tools(self._tools)
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
        return json.dumps(msgs, ensure_ascii=False)

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

    def set_tools(self, tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = NOT_GIVEN):
        if tools != NOT_GIVEN and isinstance(tools, list) and len(tools) == 0:
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
