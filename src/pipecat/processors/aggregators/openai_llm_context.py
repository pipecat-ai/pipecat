#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass

from typing import List

from pipecat.frames.frames import Frame

from openai._types import NOT_GIVEN, NotGiven

from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionMessageParam
)


class OpenAILLMContext:

    def __init__(
        self,
        messages: List[ChatCompletionMessageParam] | None = None,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN
    ):
        self.messages: List[ChatCompletionMessageParam] = messages if messages else [
        ]
        self.tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = tool_choice
        self.tools: List[ChatCompletionToolParam] | NotGiven = tools

    @staticmethod
    def from_messages(messages: List[dict]) -> "OpenAILLMContext":
        context = OpenAILLMContext()
        for message in messages:
            context.add_message({
                "content": message["content"],
                "role": message["role"],
                "name": message["name"] if "name" in message else message["role"]
            })
        return context

    def add_message(self, message: ChatCompletionMessageParam):
        self.messages.append(message)

    def get_messages(self) -> List[ChatCompletionMessageParam]:
        return self.messages

    def set_tool_choice(
        self, tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    ):
        self.tool_choice = tool_choice

    def set_tools(self, tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN):
        if tools != NOT_GIVEN and len(tools) == 0:
            tools = NOT_GIVEN

        self.tools = tools


@dataclass
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """
    context: OpenAILLMContext
