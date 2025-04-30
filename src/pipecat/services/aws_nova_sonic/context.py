#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import copy
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    FunctionCallResultFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.aws_nova_sonic.frames import AWSNovaSonicFunctionCallResultFrame
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)


class Role(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    TOOL = "TOOL"


@dataclass
class AWSNovaSonicConversationHistoryMessage:
    role: Role  # only USER and ASSISTANT
    text: str


@dataclass
class AWSNovaSonicConversationHistory:
    instruction: str = None
    messages: list[AWSNovaSonicConversationHistoryMessage] = field(default_factory=list)


@dataclass
class AWSNovaSonicLLMContext(OpenAILLMContext):
    @staticmethod
    def upgrade_to_nova_sonic(obj: OpenAILLMContext) -> "AWSNovaSonicLLMContext":
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, AWSNovaSonicLLMContext):
            obj.__class__ = AWSNovaSonicLLMContext
        return obj

    def get_messages_for_initializing_history(self) -> AWSNovaSonicConversationHistory:
        history = AWSNovaSonicConversationHistory()

        # Bail if there are no messages
        if not self.messages:
            return history

        messages = copy.deepcopy(self.messages)

        # If we have a "system" message as our first message, let's pull that out into "instruction"
        if messages[0].get("role") == "system":
            system = messages.pop(0)
            content = system.get("content")
            if isinstance(content, str):
                history.instruction = content
            elif isinstance(content, list):
                history.instruction = content[0].get("text")

        # Process remaining messages to fill out conversation history.
        # Nova Sonic supports "user" and "assistant" messages in history.
        for message in messages:
            history_message = self.from_standard_message(message)
            if history_message:
                history.messages.append(history_message)

        return history

    def from_standard_message(self, message) -> AWSNovaSonicConversationHistoryMessage:
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

    def add_user_transcription_text_as_message(self, text):
        message = {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }
        self.add_message(message)


@dataclass
class AWSNovaSonicMessagesUpdateFrame(DataFrame):
    context: AWSNovaSonicLLMContext


class AWSNovaSonicUserContextAggregator(OpenAIUserContextAggregator):
    async def process_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ):
        await super().process_frame(frame, direction)

        # Parent does not push LLMMessagesUpdateFrame
        if isinstance(frame, LLMMessagesUpdateFrame):
            await self.push_frame(AWSNovaSonicMessagesUpdateFrame(context=self._context))

        # Parent also doesn't push the LLMSetToolsFrame
        # TODO: this
        # if isinstance(frame, LLMSetToolsFrame):
        #     await self.push_frame(frame, direction)


class AWSNovaSonicAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        await super().handle_function_call_result(frame)

        # The standard function callback code path pushes the FunctionCallResultFrame from the llm itself,
        # so we didn't have a chance to add the result to the openai realtime api context. Let's push a
        # special frame to do that.
        await self.push_frame(
            AWSNovaSonicFunctionCallResultFrame(result_frame=frame), FrameDirection.UPSTREAM
        )


@dataclass
class AWSNovaSonicContextAggregatorPair:
    _user: AWSNovaSonicUserContextAggregator
    _assistant: AWSNovaSonicAssistantContextAggregator

    def user(self) -> AWSNovaSonicUserContextAggregator:
        return self._user

    def assistant(self) -> AWSNovaSonicAssistantContextAggregator:
        return self._assistant
