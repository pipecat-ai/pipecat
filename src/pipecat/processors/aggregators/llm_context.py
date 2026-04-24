#
# Copyright (c) 2024-2026, Daily
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
import copy
import io
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeGuard, TypeVar

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
# NOTE: these are aliased to OpenAI's today, but callers should treat them as
# LLMContext's own types — independent definitions that happen to coincide
# with OpenAI's as an implementation detail. If universal context types need
# to someday diverge from OpenAI's, we should consider managing our own
# definitions (but with care, since OpenAI's types are somewhat of a standard
# and we want to continue supporting them). In the meantime, code at the
# LLMContext/OpenAI boundary should use explicit casts rather than rely on
# the aliasing.
LLMStandardMessage = ChatCompletionMessageParam
LLMContextToolChoice = ChatCompletionToolChoiceOptionParam
NOT_GIVEN = OPEN_AI_NOT_GIVEN
NotGiven = OpenAINotGiven


_T = TypeVar("_T")


def is_given(value: _T | NotGiven) -> TypeGuard[_T]:
    """Check whether a value was explicitly provided.

    Typically used when checking whether a ``NotGiven``-valued field or
    parameter was set::

        if is_given(context.tools):
            ...

    Also acts as a type guard: inside a true branch, the value is narrowed
    to exclude ``NotGiven`` (e.g. ``ToolsSchema | NotGiven`` becomes
    ``ToolsSchema``).

    Args:
        value: The value to check.

    Returns:
        ``True`` if *value* is anything other than ``NOT_GIVEN``.
    """
    return not isinstance(value, NotGiven)


@dataclass
class LLMSpecificMessage:
    """A container for a context message that is specific to a particular LLM service.

    Enables the use of service-specific message types while maintaining
    compatibility with the universal LLM context format.
    """

    llm: str
    message: Any


LLMContextMessage: TypeAlias = LLMStandardMessage | LLMSpecificMessage


class LLMContext:
    """Manages conversation context for LLM interactions.

    Handles message history, tool definitions, tool choices, and multimedia
    content for LLM conversations. Provides methods for message manipulation,
    and content formatting.
    """

    def __init__(
        self,
        messages: list[LLMContextMessage] | None = None,
        tools: ToolsSchema | NotGiven = NOT_GIVEN,
        tool_choice: LLMContextToolChoice | NotGiven = NOT_GIVEN,
    ):
        """Initialize the LLM context.

        Args:
            messages: Initial list of conversation messages.
            tools: Available tools for the LLM to use.
            tool_choice: Tool selection strategy for the LLM.
        """
        self._messages: list[LLMContextMessage] = messages if messages else []
        self._tools: ToolsSchema | NotGiven = LLMContext._normalize_and_validate_tools(tools)
        self._tool_choice: LLMContextToolChoice | NotGiven = tool_choice

    @staticmethod
    def create_image_url_message(
        *,
        role: str = "user",
        url: str,
        text: str | None = None,
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
        text: str | None = None,
    ) -> LLMContextMessage:
        """Create a context message containing an image.

        Args:
            role: The role of this message (defaults to "user").
            format: Image format (e.g., 'RGB', 'RGBA', or, if already encoded,
                the MIME type like 'image/jpeg').
            size: Image dimensions as (width, height) tuple.
            image: Raw image bytes.
            text: Optional text to include with the image.
        """
        # Format is a mime type: image is already encoded
        image_already_encoded = format.startswith("image/")

        def encode_image():
            if image_already_encoded:
                bytes = image
            else:
                # Encode to JPEG
                buffer = io.BytesIO()
                Image.frombytes(format, size, image).save(buffer, format="JPEG")
                bytes = buffer.getvalue()
            encoded_image = base64.b64encode(bytes).decode("utf-8")
            return encoded_image

        encoded_image = await asyncio.to_thread(encode_image)

        url = f"data:{format if image_already_encoded else 'image/jpeg'};base64,{encoded_image}"

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
        content = [{"type": "text", "text": text}]

        def encode_audio():
            sample_rate = audio_frames[0].sample_rate
            num_channels = audio_frames[0].num_channels

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
    def messages(self) -> list[LLMContextMessage]:
        """Get the current messages list.

        NOTE: This is equivalent to calling `get_messages()` with no filter. If
        you want to filter out LLM-specific messages that don't pertain to your
        LLM, use `get_messages()` directly.

        Returns:
            List of conversation messages.
        """
        return self.get_messages()

    def get_messages(
        self,
        llm_specific_filter: str | None = None,
        *,
        truncate_large_values: bool = False,
    ) -> list[LLMContextMessage]:
        """Get the current messages list.

        Args:
            llm_specific_filter: Optional filter to return LLM-specific
                messages for the given LLM, in addition to the standard
                messages. If messages end up being filtered, an error will be
                logged; this is intended to catch accidental use of
                incompatible LLM-specific messages.
            truncate_large_values: If True, return deep copies of messages with
                large values shortened. For standard messages, known binary
                data (base64-encoded images, audio) is replaced with short
                placeholders. For LLM-specific messages, long string values
                are truncated.

        Returns:
            List of conversation messages.
        """
        if llm_specific_filter is None:
            messages = self._messages
        else:
            messages = [
                msg
                for msg in self._messages
                if not isinstance(msg, LLMSpecificMessage) or msg.llm == llm_specific_filter
            ]
            if len(messages) < len(self._messages):
                logger.error(
                    f"Attempted to use incompatible LLMSpecificMessages with LLM '{llm_specific_filter}'."
                )

        if truncate_large_values:
            messages = LLMContext._truncate_large_values_from_messages(messages)

        return messages

    @staticmethod
    def _truncate_large_values_from_messages(
        messages: list[LLMContextMessage],
    ) -> list[LLMContextMessage]:
        """Return deep copies of messages with large values replaced by placeholders.

        For standard (universal-format) messages, the following known binary
        patterns are replaced with short placeholders:

        - ``image_url`` items with ``data:image/...`` base64 URLs
        - ``input_audio`` items with ``input_audio.data`` or ``audio`` fields
        - ``audio`` items with an ``audio`` field
        - Top-level messages with a ``mime_type`` starting with ``image/``

        For ``LLMSpecificMessage`` instances, long string values are truncated
        since the internal structure is provider-specific.
        """
        result = []
        for message in messages:
            if isinstance(message, LLMSpecificMessage):
                msg_copy = copy.deepcopy(message)
                msg_copy.message = LLMContext._truncate_long_strings(msg_copy.message)
                result.append(msg_copy)
                continue

            msg = copy.deepcopy(message)
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    item_type = item.get("type")
                    if item_type == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/"):
                            item["image_url"]["url"] = "data:image/..."
                    elif item_type == "input_audio":
                        if "input_audio" in item:
                            item["input_audio"]["data"] = "..."
                        if "audio" in item:
                            item["audio"] = "..."
                    elif item_type == "audio":
                        if "audio" in item:
                            item["audio"] = "..."

            if msg.get("mime_type", "").startswith("image/"):
                msg["data"] = "..."

            result.append(msg)
        return result

    @staticmethod
    def _truncate_long_strings(value: Any, *, max_length: int = 100) -> Any:
        """Recursively truncate long strings in a nested structure.

        Preserves the structure of dicts and lists while truncating any string
        values that exceed ``max_length``.

        Args:
            value: The value to process (dict, list, str, or other).
            max_length: Strings longer than this are truncated.

        Returns:
            A copy of the structure with long strings truncated.
        """
        if isinstance(value, str):
            if len(value) > max_length:
                return f"{value[:max_length]}...({len(value)} chars)"
            return value
        elif isinstance(value, dict):
            return {
                k: LLMContext._truncate_long_strings(v, max_length=max_length)
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [
                LLMContext._truncate_long_strings(item, max_length=max_length) for item in value
            ]
        return value

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

    def add_messages(self, messages: list[LLMContextMessage]):
        """Add multiple messages to the context.

        Args:
            messages: List of messages to add to the conversation history.
        """
        self._messages.extend(messages)

    def set_messages(self, messages: list[LLMContextMessage]):
        """Replace all messages in the context.

        Args:
            messages: New list of messages to replace the current history.
        """
        self._messages[:] = messages

    def transform_messages(
        self, transform: Callable[[list[LLMContextMessage]], list[LLMContextMessage]]
    ):
        """Transform the current messages using the provided function.

        Args:
            transform: A function that takes the current list of messages and returns
                a modified list of messages to set in the context.
        """
        self.set_messages(transform(self._messages))

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
        self,
        *,
        format: str,
        size: tuple[int, int],
        image: bytes,
        text: str | None = None,
        role: str = "user",
    ):
        """Add a message containing an image frame.

        Args:
            format: Image format (e.g., 'RGB', 'RGBA', or, if already encoded,
                the MIME type like 'image/jpeg').
            size: Image dimensions as (width, height) tuple.
            image: Raw image bytes.
            text: Optional text to include with the image.
            role: The role of this message (defaults to "user").
        """
        message = await LLMContext.create_image_message(
            role=role, format=format, size=size, image=image, text=text
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
