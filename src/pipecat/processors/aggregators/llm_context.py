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
# NOTE: this is just for convenience, for now. As soon as the universal types
# diverge from OpenAI's, we should ditch this. In fact, audio frames already
# diverge from OpenAI's standard format...we really ought to do this.
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
        # TODO: convert empty ToolsSchema to NOT_GIVEN if needed?
        # TODO: maybe someday also convert provider-specific tools to ToolsSchema so it's always in a provider-neutral format here? See open_ai_adapter.py for related comment. Pipecat Flows is currently converting provider-specific tools to ToolsSchema...
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
        # TODO: we might not want the universal format to be base64 encoded, since encoding is not needed by all LLM services; today, te Gemini adapter has to decode from base64, which is less than ideal.
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        content = []
        if text:
            content.append({"type": "text", "text": text})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
        )
        self.add_message({"role": "user", "content": content})

    # NOTE: today we've only built support for audio frames with the Google
    # LLM, so this "universal" representation skews towards that.
    # When we add support for other LLMs, we may need to adjust this.
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
        # TODO: filter this out in OpenAI adapter, since it doesn't support audio frames
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": data,
                    "sample_rate": sample_rate,
                    "num_channels": num_channels,
                },
            }
        )
        self.add_message({"role": "user", "content": content})
