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
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolChoiceFrame,
    LLMSetToolsFrame,
    StartInterruptionFrame,
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


class Role(Enum):
    """Roles supported in AWS Nova Sonic conversations."""

    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"


@dataclass
class AWSNovaSonicConversationHistoryMessage:
    """A single message in AWS Nova Sonic conversation history.

    Parameters:
        role: The role of the message sender (USER or ASSISTANT only).
        text: The text content of the message.
    """

    role: Role  # only USER and ASSISTANT
    text: str


@dataclass
class AWSNovaSonicConversationHistory:
    """Complete conversation history for AWS Nova Sonic initialization.

    Parameters:
        system_instruction: System-level instruction for the conversation.
        messages: List of conversation messages between user and assistant.
    """

    system_instruction: str = None
    messages: list[AWSNovaSonicConversationHistoryMessage] = field(default_factory=list)


class AWSNovaSonicLLMContext(OpenAILLMContext):
    """Specialized LLM context for AWS Nova Sonic service.

    Extends OpenAI context with Nova Sonic-specific message handling,
    conversation history management, and text buffering capabilities.

    Args:
        messages: Initial messages for the context.
        tools: Available tools for the context.
        **kwargs: Additional arguments passed to parent class.
    """

    def __init__(self, messages=None, tools=None, **kwargs):
        super().__init__(messages=messages, tools=tools, **kwargs)
        self.__setup_local()

    def __setup_local(self, system_instruction: str = ""):
        self._assistant_text = ""
        self._user_text = ""
        self._system_instruction = system_instruction

    @staticmethod
    def upgrade_to_nova_sonic(
        obj: OpenAILLMContext, system_instruction: str
    ) -> "AWSNovaSonicLLMContext":
        """Upgrade an OpenAI context to AWS Nova Sonic context.

        Args:
            obj: The OpenAI context to upgrade.
            system_instruction: System instruction for the context.

        Returns:
            The upgraded AWS Nova Sonic context.
        """
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, AWSNovaSonicLLMContext):
            obj.__class__ = AWSNovaSonicLLMContext
            obj.__setup_local(system_instruction)
        return obj

    # NOTE: this method has the side-effect of updating _system_instruction from messages
    def get_messages_for_initializing_history(self) -> AWSNovaSonicConversationHistory:
        """Get conversation history for initializing AWS Nova Sonic session.

        Processes stored messages and extracts system instruction and conversation
        history in the format expected by AWS Nova Sonic.

        Returns:
            Formatted conversation history with system instruction and messages.
        """
        history = AWSNovaSonicConversationHistory(system_instruction=self._system_instruction)

        # Bail if there are no messages
        if not self.messages:
            return history

        messages = copy.deepcopy(self.messages)

        # If we have a "system" message as our first message, let's pull that out into "instruction"
        if messages[0].get("role") == "system":
            system = messages.pop(0)
            content = system.get("content")
            if isinstance(content, str):
                history.system_instruction = content
            elif isinstance(content, list):
                history.system_instruction = content[0].get("text")
            if history.system_instruction:
                self._system_instruction = history.system_instruction

        # Process remaining messages to fill out conversation history.
        # Nova Sonic supports "user" and "assistant" messages in history.
        for message in messages:
            history_message = self.from_standard_message(message)
            if history_message:
                history.messages.append(history_message)

        return history

    def get_messages_for_persistent_storage(self):
        """Get messages formatted for persistent storage.

        Returns:
            List of messages including system instruction if present.
        """
        messages = super().get_messages_for_persistent_storage()
        # If we have a system instruction and messages doesn't already contain it, add it
        if self._system_instruction and not (messages and messages[0].get("role") == "system"):
            messages.insert(0, {"role": "system", "content": self._system_instruction})
        return messages

    def from_standard_message(self, message) -> AWSNovaSonicConversationHistoryMessage:
        """Convert standard message format to Nova Sonic format.

        Args:
            message: Standard message dictionary to convert.

        Returns:
            Nova Sonic conversation history message, or None if not convertible.
        """
        role = message.get("role")
        if message.get("role") == "user" or message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(message.get("content"), list):
                content = ""
                for c in message.get("content"):
                    if c.get("type") == "text":
                        content += " " + c.get("text")
                    else:
                        logger.error(
                            f"Unhandled content type in context message: {c.get('type')} - {message}"
                        )
            # There won't be content if this is an assistant tool call entry.
            # We're ignoring those since they can't be loaded into AWS Nova Sonic conversation
            # history
            if content:
                return AWSNovaSonicConversationHistoryMessage(role=Role[role.upper()], text=content)
        # NOTE: we're ignoring messages with role "tool" since they can't be loaded into AWS Nova
        # Sonic conversation history

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
                StartInterruptionFrame,
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
