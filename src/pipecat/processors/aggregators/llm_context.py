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
import io
import wave
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeAlias, Union

from loguru import logger
from openai._types import NOT_GIVEN as OPEN_AI_NOT_GIVEN
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)
from PIL import Image

from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema, ToolsSchemaDiff
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
class LLMContextDiff:
    """Represents the differences between two LLMContext instances.

    Parameters:
        messages_appended: New messages appended at the end. Empty if history_edited is True.
        history_edited: True if earlier messages were changed, inserted, or removed.
        tool_calls_resolved: List of tool_call_ids that changed from "IN_PROGRESS" to a result.
        tools_diff: Differences in tools configuration, or None if unchanged or both NOT_GIVEN.
        tool_choice_changed: True if the tool_choice setting differs.
    """

    messages_appended: List["LLMContextMessage"] = field(default_factory=list)
    history_edited: bool = False
    tool_calls_resolved: List[str] = field(default_factory=list)
    tools_diff: ToolsSchemaDiff = field(default_factory=ToolsSchemaDiff)
    tool_choice_changed: bool = False

    def has_changes(self) -> bool:
        """Check if there are any differences.

        Returns:
            True if any field indicates a change, False otherwise.
        """
        return bool(
            self.messages_appended
            or self.history_edited
            or self.tool_calls_resolved
            or self.tools_diff.has_changes()
            or self.tool_choice_changed
        )


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

        .. deprecated:: 0.0.99
            `from_openai_context()` is deprecated and will be removed in a future version.
            Directly use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
            See `OpenAILLMContext` docstring for migration guide.

        Args:
            openai_context: The OpenAI LLM context to convert.

        Returns:
            New LLMContext instance with converted messages and settings.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "from_openai_context() (likely invoked by create_context_aggregator()) is deprecated and will be removed in a future version. "
                "Directly use the universal LLMContext and LLMContextAggregatorPair instead. "
                "See OpenAILLMContext docstring for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )

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
        self,
        *,
        format: str,
        size: tuple[int, int],
        image: bytes,
        text: Optional[str] = None,
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

    def diff(self, other: "LLMContext") -> LLMContextDiff:
        """Compare this context to another and return the differences.

        Compares self (the "before" state) to other (the "after" state) and
        identifies what has changed.

        Args:
            other: The LLMContext to compare against (the "after" state).

        Returns:
            ContextDiff containing the differences between self and other.
        """
        result = LLMContextDiff()

        # Compare messages
        self_messages = self._messages
        other_messages = other._messages
        self_len = len(self_messages)
        other_len = len(other_messages)

        # Check if history was edited (messages removed, modified, or inserted in the middle)
        if other_len < self_len:
            # Messages were removed
            result.history_edited = True
        else:
            # Check if the prefix matches (first self_len messages should be identical)
            for i in range(self_len):
                if not self._messages_equal(self_messages[i], other_messages[i]):
                    result.history_edited = True
                    break

        # If history wasn't edited, capture appended messages
        if not result.history_edited and other_len > self_len:
            result.messages_appended = other_messages[self_len:]

        # Find resolved tool calls (IN_PROGRESS -> something else)
        result.tool_calls_resolved = self._find_resolved_tool_calls(other)

        # Compare tools
        result.tools_diff = self._compute_tools_diff(other)

        # Compare tool_choice
        # print("[pk] comparing tool choices: ", self._tool_choice, other._tool_choice, type(self._tool_choice), type(other._tool_choice), self._tool_choice == other._tool_choice, self._tool_choice != other._tool_choice)
        # (For some reason if they're both NOT_GIVEN, equality check returns False?)
        if not self._tool_choice and not other._tool_choice:
            result.tool_choice_changed = False
        else:
            result.tool_choice_changed = self._tool_choice != other._tool_choice

        return result

    def _messages_equal(self, msg1: LLMContextMessage, msg2: LLMContextMessage) -> bool:
        """Compare two messages for equality.

        Args:
            msg1: First message to compare.
            msg2: Second message to compare.

        Returns:
            True if the messages are equal, False otherwise.
        """
        # Handle LLMSpecificMessage
        if isinstance(msg1, LLMSpecificMessage) and isinstance(msg2, LLMSpecificMessage):
            return msg1.llm == msg2.llm and msg1.message == msg2.message
        elif isinstance(msg1, LLMSpecificMessage) or isinstance(msg2, LLMSpecificMessage):
            return False

        # Both are standard messages (dicts)
        return msg1 == msg2

    def _find_resolved_tool_calls(self, other: "LLMContext") -> List[str]:
        """Find tool calls that changed from IN_PROGRESS to a resolved state.

        Args:
            other: The context to compare against (the "after" state).

        Returns:
            List of tool_call_ids that were resolved.
        """
        resolved: List[str] = []

        # Build a map of tool_call_id -> content for "other" context
        other_tool_contents: Dict[str, Any] = {}
        for msg in other._messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    other_tool_contents[tool_call_id] = msg.get("content")

        # Find tool messages in self that are IN_PROGRESS but resolved in other
        for msg in self._messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                if msg.get("content") == "IN_PROGRESS":
                    tool_call_id = msg.get("tool_call_id")
                    if tool_call_id and tool_call_id in other_tool_contents:
                        other_content = other_tool_contents[tool_call_id]
                        if other_content != "IN_PROGRESS":
                            resolved.append(tool_call_id)

        return resolved

    def _compute_tools_diff(self, other: "LLMContext") -> ToolsSchemaDiff:
        """Compute the difference in tools between self and other.

        Args:
            other: The context to compare against (the "after" state).

        Returns:
            ToolsSchemaDiff if there are changes, None if both are NOT_GIVEN or identical.
        """
        self_has_tools = isinstance(self._tools, ToolsSchema)
        other_has_tools = isinstance(other._tools, ToolsSchema)

        if not self_has_tools and not other_has_tools:
            # Both are NOT_GIVEN
            return ToolsSchemaDiff()

        if self_has_tools and other_has_tools:
            # Both have tools - use ToolsSchema.diff()
            diff = self._tools.diff(other._tools)
            return diff

        if not self_has_tools and other_has_tools:
            # Tools were added (self is NOT_GIVEN, other has tools)
            return ToolsSchemaDiff(
                standard_tools_added=[tool.name for tool in other._tools.standard_tools],
                custom_tools_changed=other._tools.custom_tools is not None,
            )

        # Tools were removed (self has tools, other is NOT_GIVEN)
        return ToolsSchemaDiff(
            standard_tools_removed=[tool.name for tool in self._tools.standard_tools],
            custom_tools_changed=self._tools.custom_tools is not None,
        )
