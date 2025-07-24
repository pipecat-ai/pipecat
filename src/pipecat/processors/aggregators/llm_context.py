#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Universal LLM context management for LLM services in Pipecat.

Context contents are represented in a generic format (extended from OpenAI)
that supports a union of known Pipecat LLM service functionality.

Whenever an LLM service needs to access context, it does a just-in-time
translation from this universal context into whatever format it needs, using a
service-specific adapter.
"""

import base64
import io
from dataclasses import dataclass
from typing import Any, List, Optional

from openai._types import NOT_GIVEN as OPEN_AI_NOT_GIVEN
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from PIL import Image

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import AudioRawFrame, Frame

# "Re-export" types from OpenAI that we're using as universal context types.
LLMContextMessage = ChatCompletionMessageParam
LLMContextTool = ChatCompletionToolParam
LLMContextToolChoice = ChatCompletionToolChoiceOptionParam
NOT_GIVEN = OPEN_AI_NOT_GIVEN
NotGiven = OpenAINotGiven


class LLMContext:
    """Manages conversation context for LLM interactions.

    Handles message history, tool definitions, tool choices, and multimedia
    content for LLM conversations. Provides methods for message manipulation,
    and content formatting.
    """

    def __init__(
        self,
        messages: Optional[List[LLMContextMessage]] = None,
        tools: List[LLMContextTool] | NotGiven | ToolsSchema = NOT_GIVEN,
        tool_choice: LLMContextToolChoice | NotGiven = NOT_GIVEN,
    ):
        """Initialize the LLM context.

        Args:
            messages: Initial list of conversation messages.
            tools: Available tools for the LLM to use.
            tool_choice: Tool selection strategy for the LLM.
        """
        self._messages: List[LLMContextMessage] = messages if messages else []
        self._tools: List[LLMContextTool] | NotGiven | ToolsSchema = tools
        self._tool_choice: LLMContextToolChoice | NotGiven = tool_choice

    @property
    def messages(self) -> List[LLMContextMessage]:
        """Get the current messages list.

        Returns:
            List of conversation messages.
        """
        return self._messages

    @property
    def tools(self) -> List[LLMContextTool] | NotGiven | List[Any]:
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

    def set_tools(self, tools: List[LLMContextTool] | NotGiven | ToolsSchema = NOT_GIVEN):
        """Set the available tools for the LLM.

        Args:
            tools: List of tools available to the LLM, a ToolsSchema, or NOT_GIVEN to disable tools.
        """
        # TODO: convert empty ToolsSchema to NOT_GIVEN if needed
        # TODO: maybe also convert non-ToolsSchema tools to ToolsSchema? See open_ai_adapter.py for related comment
        if isinstance(tools, list) and len(tools) == 0:
            tools = NOT_GIVEN
        self._tools = tools

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

    def add_audio_frames_message(self, *, audio_frames: list[AudioRawFrame], text: str = None):
        """Add a message containing audio frames.

        Args:
            audio_frames: List of audio frame objects to include.
            text: Optional text to include with the audio.

        Note:
            This method is currently a placeholder for future implementation.
        """
        # TODO: implement storing universal representation of audio frames in context (only used by Google for now)
        pass


@dataclass
class LLMContextFrame(Frame):
    """Frame containing LLM context.

    Used as a signal to LLM services to ingest the provided context and
    generate a response based on it.

    Parameters:
        context: The LLM context containing messages, tools, and configuration.
    """

    context: LLMContext
