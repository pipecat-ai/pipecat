#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Universal LLM context management for LLM services in Pipecat.

Context contents are represented in a universal format (based on OpenAI)
that supports a union of known Pipecat LLM service functionality.

Whenever an LLM service needs to access context, it does a just-in-time
translation from this universal context into whatever format it needs, using a
service-specific adapter.
"""

import base64
import io
from dataclasses import dataclass
from typing import Any, List, Optional, TypeAlias, Union

from loguru import logger
from openai._types import NOT_GIVEN as OPEN_AI_NOT_GIVEN
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)
from PIL import Image

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import AudioRawFrame

# "Re-export" types from OpenAI that we're using as universal context types.
# NOTE: if universal message types need to someday diverge from OpenAI's, we
# should consider managing our own definitions. But we should do so carefully,
# as the OpenAI messages are somewhat of a standard and we want to continue
# supporting them.
LLMStandardMessage = ChatCompletionMessageParam
LLMContextToolChoice = ChatCompletionToolChoiceOptionParam
NOT_GIVEN = OPEN_AI_NOT_GIVEN
NotGiven = OpenAINotGiven


@dataclass
class LLMSpecificMessage:
    """A container for a context message that is specific to a particular LLM service.

    Enables the use of service-specific message types while maintaining
    compatibility with the universal LLM context format.
    """

    llm: str
    message: Any


LLMContextMessage: TypeAlias = Union[LLMStandardMessage, LLMSpecificMessage]


class LLMContext:
    """Manages conversation context for LLM interactions.

    Handles message history, tool definitions, tool choices, and multimedia
    content for LLM conversations. Provides methods for message manipulation,
    and content formatting.
    """

    def __init__(
        self,
        messages: Optional[List[LLMContextMessage]] = None,
        tools: ToolsSchema | NotGiven = NOT_GIVEN,
        tool_choice: LLMContextToolChoice | NotGiven = NOT_GIVEN,
    ):
        """Initialize the LLM context.

        Args:
            messages: Initial list of conversation messages.
            tools: Available tools for the LLM to use.
            tool_choice: Tool selection strategy for the LLM.
        """
        self._messages: List[LLMContextMessage] = messages if messages else []
        self._tools: ToolsSchema | NotGiven = LLMContext._normalize_and_validate_tools(tools)
        self._tool_choice: LLMContextToolChoice | NotGiven = tool_choice

    def get_messages(self, llm_specific_filter: Optional[str] = None) -> List[LLMContextMessage]:
        """Get the current messages list.

        Args:
            llm_specific_filter: Optional filter to return LLM-specific
                messages for the given LLM, in addition to the standard
                messages. If messages end up being filtered, an error will be
                logged.

        Returns:
            List of conversation messages.
        """
        if llm_specific_filter is None:
            return self._messages
        filtered_messages = [
            msg
            for msg in self._messages
            if not isinstance(msg, LLMSpecificMessage) or msg.llm == llm_specific_filter
        ]
        if len(filtered_messages) < len(self._messages):
            logger.error(
                f"Attempted to use incompatible LLMSpecificMessages with LLM '{llm_specific_filter}'."
            )
        return filtered_messages

    @property
    def tools(self) -> ToolsSchema | NotGiven:
        """Get the tools list.

        Returns:
            Tools list.
        """
        return self._tools

    @property
    def tool_choice(self) -> LLMContextToolChoice | NotGiven:
        """Get the current tool choice setting.

        Returns:
            The tool choice configuration.
        """
        return self._tool_choice

    def add_message(self, message: LLMContextMessage):
        """Add a single message to the context.

        Args:
            message: The message to add to the conversation history.
        """
        self._messages.append(message)

    def add_messages(self, messages: List[LLMContextMessage]):
        """Add multiple messages to the context.

        Args:
            messages: List of messages to add to the conversation history.
        """
        self._messages.extend(messages)

    def set_messages(self, messages: List[LLMContextMessage]):
        """Replace all messages in the context.

        Args:
            messages: New list of messages to replace the current history.
        """
        self._messages[:] = messages

    def set_tools(self, tools: ToolsSchema | NotGiven = NOT_GIVEN):
        """Set the available tools for the LLM.

        Args:
            tools: A ToolsSchema or NOT_GIVEN to disable tools.
        """
        self._tools = LLMContext._normalize_and_validate_tools(tools)

    def set_tool_choice(self, tool_choice: LLMContextToolChoice | NotGiven):
        """Set the tool choice configuration.

        Args:
            tool_choice: Tool selection strategy for the LLM.
        """
        self._tool_choice = tool_choice

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

    def add_audio_frames_message(
        self, *, audio_frames: list[AudioRawFrame], text: str = "Audio follows"
    ):
        """Add a message containing audio frames.

        Args:
            audio_frames: List of audio frame objects to include.
            text: Optional text to include with the audio.
        """
        if not audio_frames:
            return

        sample_rate = audio_frames[0].sample_rate
        num_channels = audio_frames[0].num_channels

        content = []
        content.append({"type": "text", "text": text})
        data = b"".join(frame.audio for frame in audio_frames)
        data = bytes(
            self._create_wav_header(
                sample_rate,
                num_channels,
                16,
                len(data),
            )
            + data
        )
        encoded_audio = base64.b64encode(data).decode("utf-8")
        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": encoded_audio, "format": "wav"},
            }
        )
        self.add_message({"role": "user", "content": content})

    def _create_wav_header(self, sample_rate, num_channels, bits_per_sample, data_size):
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

    @staticmethod
    def _normalize_and_validate_tools(tools: ToolsSchema | NotGiven) -> ToolsSchema | NotGiven:
        """Normalize and validate the given tools.

        Raises:
            TypeError: If tools are not a ToolsSchema or NotGiven.
        """
        if isinstance(tools, ToolsSchema):
            if not tools.standard_tools and not tools.custom_tools:
                return NOT_GIVEN
            return tools
        elif tools is NOT_GIVEN:
            return NOT_GIVEN
        else:
            raise TypeError(
                f"In LLMContext, tools must be a ToolsSchema object or NOT_GIVEN. Got type: {type(tools)}",
            )
