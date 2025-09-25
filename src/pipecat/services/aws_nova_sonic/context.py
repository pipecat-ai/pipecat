#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Context management for AWS Nova Sonic LLM service.

This module provides specialized context aggregators and message handling for AWS Nova Sonic,
including conversation history management and role-specific message processing.
"""

import copy
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolChoiceFrame,
    LLMSetToolsFrame,
    TextFrame,
    UserImageRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws_nova_sonic.frames import AWSNovaSonicFunctionCallResultFrame
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)


class AWSNovaSonicLLMContext(OpenAILLMContext):
    """Specialized LLM context for AWS Nova Sonic service.

    Extends OpenAI context with Nova Sonic-specific message handling,
    conversation history management, and text buffering capabilities.
    """

    def buffer_user_text(self, text):
        """Buffer user text for later flushing to context.

        Args:
            text: User text to buffer.
        """
        self._user_text += f" {text}" if self._user_text else text
        # logger.debug(f"User text buffered: {self._user_text}")

    def flush_aggregated_user_text(self) -> str:
        """Flush buffered user text to context as a complete message.

        Returns:
            The flushed user text, or empty string if no text was buffered.
        """
        if not self._user_text:
            return ""
        user_text = self._user_text
        message = {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        }
        self._user_text = ""
        self.add_message(message)
        # logger.debug(f"Context updated (user): {self.get_messages_for_logging()}")
        return user_text

    def buffer_assistant_text(self, text):
        """Buffer assistant text for later flushing to context.

        Args:
            text: Assistant text to buffer.
        """
        self._assistant_text += text
        # logger.debug(f"Assistant text buffered: {self._assistant_text}")

    def flush_aggregated_assistant_text(self):
        """Flush buffered assistant text to context as a complete message."""
        if not self._assistant_text:
            return
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": self._assistant_text}],
        }
        self._assistant_text = ""
        self.add_message(message)
        # logger.debug(f"Context updated (assistant): {self.get_messages_for_logging()}")


@dataclass
class AWSNovaSonicMessagesUpdateFrame(DataFrame):
    """Frame containing updated AWS Nova Sonic context.

    Parameters:
        context: The updated AWS Nova Sonic LLM context.
    """

    context: AWSNovaSonicLLMContext


class AWSNovaSonicUserContextAggregator(OpenAIUserContextAggregator):
    """Context aggregator for user messages in AWS Nova Sonic conversations.

    Extends the OpenAI user context aggregator to emit Nova Sonic-specific
    context update frames.
    """

    async def process_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        """Process frames and emit Nova Sonic-specific context updates.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        await super().process_frame(frame, direction)

        # Parent does not push LLMMessagesUpdateFrame
        if isinstance(frame, LLMMessagesUpdateFrame):
            await self.push_frame(AWSNovaSonicMessagesUpdateFrame(context=self._context))


class AWSNovaSonicAssistantContextAggregator(OpenAIAssistantContextAggregator):
    """Context aggregator for assistant messages in AWS Nova Sonic conversations.

    Provides specialized handling for assistant responses and function calls
    in AWS Nova Sonic context, with custom frame processing logic.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Nova Sonic-specific logic.

        Args:
            frame: The frame to process.
            direction: The direction the frame is traveling.
        """
        # HACK: For now, disable the context aggregator by making it just pass through all frames
        # that the parent handles (except the function call stuff, which we still need).
        # For an explanation of this hack, see
        # AWSNovaSonicLLMService._report_assistant_response_text_added.
        if isinstance(
            frame,
            (
                InterruptionFrame,
                LLMFullResponseStartFrame,
                LLMFullResponseEndFrame,
                TextFrame,
                LLMMessagesAppendFrame,
                LLMMessagesUpdateFrame,
                LLMSetToolsFrame,
                LLMSetToolChoiceFrame,
                UserImageRawFrame,
                BotStoppedSpeakingFrame,
            ),
        ):
            await self.push_frame(frame, direction)
        else:
            await super().process_frame(frame, direction)

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle function call results for AWS Nova Sonic.

        Args:
            frame: The function call result frame to handle.
        """
        await super().handle_function_call_result(frame)

        # The standard function callback code path pushes the FunctionCallResultFrame from the LLM
        # itself, so we didn't have a chance to add the result to the AWS Nova Sonic server-side
        # context. Let's push a special frame to do that.
        await self.push_frame(
            AWSNovaSonicFunctionCallResultFrame(result_frame=frame), FrameDirection.UPSTREAM
        )


@dataclass
class AWSNovaSonicContextAggregatorPair:
    """Pair of user and assistant context aggregators for AWS Nova Sonic.

    Parameters:
        _user: The user context aggregator.
        _assistant: The assistant context aggregator.
    """

    _user: AWSNovaSonicUserContextAggregator
    _assistant: AWSNovaSonicAssistantContextAggregator

    def user(self) -> AWSNovaSonicUserContextAggregator:
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> AWSNovaSonicAssistantContextAggregator:
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant
