#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass

from typing import AsyncGenerator, Callable, List

from pipecat.frames.frames import (
    Frame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from openai._types import NOT_GIVEN, NotGiven

from openai.types.chat import (
    ChatCompletionRole,
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

    @ staticmethod
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


class OpenAIContextAggregator(FrameProcessor):

    def __init__(
        self,
        context: OpenAILLMContext,
        aggregator: Callable[[Frame, str | None], str | None],
        role: ChatCompletionRole,
        start_frame: type,
        end_frame: type,
        accumulator_frame: type,
        pass_through=True,
    ):
        if not (
            issubclass(start_frame, Frame)
            and issubclass(end_frame, Frame)
            and issubclass(accumulator_frame, Frame)
        ):
            raise TypeError(
                "start_frame, end_frame and accumulator_frame must be instances of Frame"
            )

        self._context: OpenAILLMContext = context
        self._aggregator: Callable[[Frame, str | None], None] = aggregator
        self._role: ChatCompletionRole = role
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._accumulator_frame = accumulator_frame
        self._pass_through = pass_through

        self._aggregating = False
        self._aggregation = None

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, self._start_frame):
            self._aggregating = True
        elif isinstance(frame, self._end_frame):
            self._aggregating = False
            if self._aggregation:
                self._context.add_message(
                    {
                        "role": self._role,
                        "content": self._aggregation,
                        "name": self._role,
                    }  # type: ignore
                )
            self._aggregation = None
            yield OpenAILLMContextFrame(self._context)
        elif isinstance(frame, self._accumulator_frame) and self._aggregating:
            self._aggregation = self._aggregator(frame, self._aggregation)
            if self._pass_through:
                yield frame
        else:
            yield frame

    def string_aggregator(
            self,
            frame: Frame,
            aggregation: str | None) -> str | None:
        if not isinstance(frame, TextFrame):
            raise TypeError(
                "Frame must be a TextFrame instance to be aggregated by a string aggregator."
            )
        if not aggregation:
            aggregation = ""
        return " ".join([aggregation, frame.text])


class OpenAIUserContextAggregator(OpenAIContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            context=context,
            aggregator=self.string_aggregator,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            pass_through=False,
        )


class OpenAIAssistantContextAggregator(OpenAIContextAggregator):

    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            context,
            aggregator=self.string_aggregator,
            role="assistant",
            start_frame=LLMResponseStartFrame,
            end_frame=LLMResponseEndFrame,
            accumulator_frame=TextFrame,
            pass_through=True,
        )


@dataclass
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the OpenAI
    API. The context in this message is also mutable, and will be changed by the
    OpenAIContextAggregator frame processor.

    """
    context: OpenAILLMContext
