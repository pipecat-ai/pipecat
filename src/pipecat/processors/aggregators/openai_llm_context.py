#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI LLM context management for Pipecat.

This module provides classes for managing OpenAI-specific conversation contexts,
including message handling, tool management, and image/audio processing capabilities.
"""

import base64
import copy
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    """Custom JSON encoder for handling special data types in logging.

    Provides specialized encoding for io.BytesIO objects to display
    readable representations in log output instead of raw binary data.
    """

    def default(self, obj):
        """Encode special objects for JSON serialization.

        Args:
            obj: The object to encode.

        Returns:
            Encoded representation of the object.
        """
        if isinstance(obj, io.BytesIO):
            # Convert the first 8 bytes to an ASCII hex string
            return f"{obj.getbuffer()[0:8].hex()}..."
        return super().default(obj)


class OpenAILLMContext:
    """Manages conversation context for OpenAI LLM interactions.

    Handles message history, tool definitions, tool choices, and multimedia content
    for OpenAI API conversations. Provides methods for message manipulation,
    content formatting, and integration with various LLM adapters.
    """

    def __init__(
        self,
        messages: Optional[List[ChatCompletionMessageParam]] = None,
        tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
    ):
        """Initialize the OpenAI LLM context.

        Args:
            messages: Initial list of conversation messages.
            tools: Available tools for the LLM to use.
            tool_choice: Tool selection strategy for the LLM.
        """
        self._messages: List[ChatCompletionMessageParam] = messages if messages else []
        self._tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self._tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = tools
        self._llm_adapter: Optional[BaseLLMAdapter] = None

    def get_llm_adapter(self) -> Optional[BaseLLMAdapter]:
        """Get the current LLM adapter.

        Returns:
            The currently set LLM adapter, or None if not set.
        """
        return self._llm_adapter

    def set_llm_adapter(self, llm_adapter: BaseLLMAdapter):
        """Set the LLM adapter for context processing.

        Args:
            llm_adapter: The LLM adapter to use for tool conversion.
        """
        self._llm_adapter = llm_adapter

    @staticmethod
    def from_messages(messages: List[dict]) -> "OpenAILLMContext":
        """Create a context from a list of message dictionaries.

        Args:
            messages: List of message dictionaries to convert to context.

        Returns:
            New OpenAILLMContext instance with the provided messages.
        """
        context = OpenAILLMContext()

        for message in messages:
            context.add_message(message)
        return context

    @property
    def messages(self) -> List[ChatCompletionMessageParam]:
        """Get the current messages list.

        Returns:
            List of conversation messages.
        """
        return self._messages

    @property
    def tools(self) -> List[ChatCompletionToolParam] | NotGiven | List[Any]:
        """Get the tools list, converting through adapter if available.

        Returns:
            Tools list, potentially converted by the LLM adapter.
        """
        if self._llm_adapter:
            return self._llm_adapter.from_standard_tools(self._tools)
        return self._tools

    @property
    def tool_choice(self) -> ChatCompletionToolChoiceOptionParam | NotGiven:
        """Get the current tool choice setting.

        Returns:
            The tool choice configuration.
        """
        return self._tool_choice

    def add_message(self, message: ChatCompletionMessageParam):
        """Add a single message to the context.

        Args:
            message: The message to add to the conversation history.
        """
        self._messages.append(message)

    def add_messages(self, messages: List[ChatCompletionMessageParam]):
        """Add multiple messages to the context.

        Args:
            messages: List of messages to add to the conversation history.
        """
        self._messages.extend(messages)

    def set_messages(self, messages: List[ChatCompletionMessageParam]):
        """Replace all messages in the context.

        Args:
            messages: New list of messages to replace the current history.
        """
        self._messages[:] = messages

    def get_messages(self) -> List[ChatCompletionMessageParam]:
        """Get a copy of the current messages list.

        Returns:
            List of all messages in the conversation history.
        """
        return self._messages

    def get_messages_json(self) -> str:
        """Get messages as a formatted JSON string.

        Returns:
            JSON string representation of all messages with custom encoding.
        """
        return json.dumps(self._messages, cls=CustomEncoder, ensure_ascii=False, indent=2)

    def get_messages_for_logging(self) -> List[Dict[str, Any]]:
        """Get sanitized messages suitable for logging.

        Removes or truncates sensitive data like image content for safe logging.

        Returns:
            List of messages in a format ready for logging.
        """
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
        return msgs

    def from_standard_message(self, message):
        """Convert from OpenAI message format to OpenAI message format (passthrough).

        OpenAI's format allows both simple string content and structured content::

            Simple: {"role": "user", "content": "Hello"}
            Structured: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

        Since OpenAI is our standard format, this is a passthrough function.

        Args:
            message: Message in OpenAI format.

        Returns:
            Same message, unchanged.
        """
        return message

    def to_standard_messages(self, obj) -> list:
        """Convert from OpenAI message format to OpenAI message format (passthrough).

        OpenAI's format is our standard format throughout Pipecat. This function
        returns a list containing the original message to maintain consistency with
        other LLM services that may need to return multiple messages.

        Args:
            obj: Message in OpenAI format with either simple string content
                or structured list content.

        Returns:
            List containing the original messages, preserving the content format.
        """
        return [obj]

    def get_messages_for_initializing_history(self):
        """Get messages for initializing conversation history.

        Returns:
            List of messages suitable for history initialization.
        """
        return self._messages

    def get_messages_for_persistent_storage(self):
        """Get messages formatted for persistent storage.

        Returns:
            List of messages converted to standard format for storage.
        """
        messages = []
        for m in self._messages:
            standard_messages = self.to_standard_messages(m)
            messages.extend(standard_messages)
        return messages

    def set_tool_choice(self, tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven):
        """Set the tool choice configuration.

        Args:
            tool_choice: Tool selection strategy for the LLM.
        """
        self._tool_choice = tool_choice

    def set_tools(self, tools: List[ChatCompletionToolParam] | NotGiven | ToolsSchema = NOT_GIVEN):
        """Set the available tools for the LLM.

        Args:
            tools: List of tools available to the LLM, or NOT_GIVEN to disable tools.
        """
        if tools != NOT_GIVEN and isinstance(tools, list) and len(tools) == 0:
            tools = NOT_GIVEN
        self._tools = tools

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        """Add a message containing an image frame.

        Args:
            format: Image format (e.g., 'RGB', 'RGBA').
            size: Image dimensions as (width, height) tuple.
            image: Raw image bytes.
            text: Optional text to include with the image.
        """
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
        """Add a message containing audio frames.

        Args:
            audio_frames: List of audio frame objects to include.
            text: Optional text to include with the audio.

        Note:
            This method is currently a placeholder for future implementation.
        """
        # todo: implement for OpenAI models and others
        pass

    def create_wav_header(self, sample_rate, num_channels, bits_per_sample, data_size):
        """Create a WAV file header for audio data.

        Args:
            sample_rate: Audio sample rate in Hz.
            num_channels: Number of audio channels.
            bits_per_sample: Bits per audio sample.
            data_size: Size of audio data in bytes.

        Returns:
            WAV header as a bytearray.
        """
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
    """Frame containing OpenAI-specific LLM context.

    Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    Parameters:
        context: The OpenAI LLM context containing messages, tools, and configuration.
    """

    context: OpenAILLMContext
