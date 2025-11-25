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

import asyncio
import base64
import io
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, TypeAlias, Union

from loguru import logger
from openai._types import NOT_GIVEN as OPEN_AI_NOT_GIVEN
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)
from PIL import Image

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.frames.frames import AudioRawFrame

if TYPE_CHECKING:
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

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

    @staticmethod
    def from_openai_context(openai_context: "OpenAILLMContext") -> "LLMContext":
        """Create a universal LLM context from an OpenAI-specific context.

        NOTE: this should only be used internally, for facilitating migration
        from OpenAILLMContext to LLMContext. New user code should use
        LLMContext directly.

        Args:
            openai_context: The OpenAI LLM context to convert.

        Returns:
            New LLMContext instance with converted messages and settings.
        """
        # Convert tools to ToolsSchema if needed.
        # If the tools are already a ToolsSchema, this is a no-op.
        # Otherwise, we wrap them in a shim ToolsSchema.
        converted_tools = openai_context.tools
        if isinstance(converted_tools, list):
            converted_tools = ToolsSchema(
                standard_tools=[], custom_tools={AdapterType.SHIM: converted_tools}
            )
        return LLMContext(
            messages=openai_context.get_messages(),
            tools=converted_tools,
            tool_choice=openai_context.tool_choice,
        )

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

    @staticmethod
    def create_image_url_message(
        *,
        role: str = "user",
        url: str,
        text: Optional[str] = None,
    ) -> LLMContextMessage:
        """Create a context message containing an image URL.

        Args:
            role: The role of this message (defaults to "user").
            url: The URL of the image.
            text: Optional text to include with the image.
        """
        content = []
        if text:
            content.append({"type": "text", "text": text})

        content.append({"type": "image_url", "image_url": {"url": url}})

        return {"role": role, "content": content}

    @staticmethod
    async def create_image_message(
        *,
        role: str = "user",
        format: str,
        size: tuple[int, int],
        image: bytes,
        text: Optional[str] = None,
    ) -> LLMContextMessage:
        """Create a context message containing an image.

        Args:
            role: The role of this message (defaults to "user").
            format: Image format (e.g., 'RGB', 'RGBA').
            size: Image dimensions as (width, height) tuple.
            image: Raw image bytes.
            text: Optional text to include with the image.
        """

        def encode_image():
            buffer = io.BytesIO()
            Image.frombytes(format, size, image).save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded_image

        encoded_image = await asyncio.to_thread(encode_image)

        url = f"data:image/jpeg;base64,{encoded_image}"

        return LLMContext.create_image_url_message(role=role, url=url, text=text)

    @staticmethod
    async def create_audio_message(
        *, role: str = "user", audio_frames: list[AudioRawFrame], text: str = "Audio follows"
    ) -> LLMContextMessage:
        """Create a context message containing audio.

        Args:
            role: The role of this message (defaults to "user").
            audio_frames: List of audio frame objects to include.
            text: Optional text to include with the audio.
        """

        async def encode_audio():
            sample_rate = audio_frames[0].sample_rate
            num_channels = audio_frames[0].num_channels

            content = []
            content.append({"type": "text", "text": text})
            data = b"".join(frame.audio for frame in audio_frames)

            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(num_channels)
                    wf.setframerate(sample_rate)
                    wf.writeframes(data)

            encoded_audio = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded_audio

        encoded_audio = await asyncio.to_thread(encode_audio)

        content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": encoded_audio, "format": "wav"},
            }
        )

        return {"role": role, "content": content}

    @property
    def messages(self) -> List[LLMContextMessage]:
        """Get the current messages list.

        NOTE: This is equivalent to calling `get_messages()` with no filter. If
        you want to filter out LLM-specific messages that don't pertain to your
        LLM, use `get_messages()` directly.

        Returns:
            List of conversation messages.
        """
        return self.get_messages()

    def get_messages_for_persistent_storage(self) -> List[LLMContextMessage]:
        """Get messages suitable for persistent storage.

        NOTE: the only reason this method exists is because we're "silently"
        switching from OpenAILLMContext to LLMContext under the hood in some
        services and don't want to trip up users who may have been relying on
        this method, which is part of the public API of OpenAILLMContext but
        doesn't need to be for LLMContext.

        .. deprecated::
            Use `get_messages()` instead.

        Returns:
            List of conversation messages.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "get_messages_for_persistent_storage() is deprecated, use get_messages() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        return self.get_messages()

    def get_messages(self, llm_specific_filter: Optional[str] = None) -> List[LLMContextMessage]:
        """Get the current messages list.

        Args:
            llm_specific_filter: Optional filter to return LLM-specific
                messages for the given LLM, in addition to the standard
                messages. If messages end up being filtered, an error will be
                logged; this is intended to catch accidental use of
                incompatible LLM-specific messages.

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

    async def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: Optional[str] = None
    ):
        """Add a message containing an image frame.

        Args:
            format: Image format (e.g., 'RGB', 'RGBA').
            size: Image dimensions as (width, height) tuple.
            image: Raw image bytes.
            text: Optional text to include with the image.
        """
        message = await LLMContext.create_image_message(
            format=format, size=size, image=image, text=text
        )
        self.add_message(message)

    async def add_audio_frames_message(
        self, *, audio_frames: list[AudioRawFrame], text: str = "Audio follows"
    ):
        """Add a message containing audio frames.

        Args:
            audio_frames: List of audio frame objects to include.
            text: Optional text to include with the audio.
        """
        message = await LLMContext.create_audio_message(audio_frames=audio_frames, text=text)
        self.add_message(message)

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
