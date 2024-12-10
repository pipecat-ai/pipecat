#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List, Type

from pipecat.frames.frames import (
    BotInterruptionFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class LLMResponseAggregator(FrameProcessor):
    def __init__(
        self,
        *,
        messages: List[dict],
        role: str,
        start_frame,
        end_frame,
        accumulator_frame: Type[TextFrame],
        interim_accumulator_frame: Type[TextFrame] | None = None,
        handle_interruptions: bool = False,
        expect_stripped_words: bool = True,  # if True, need to add spaces between words
        interrupt_double_accumulator: bool = True,  # if True, interrupt if two or more accumulators are received
    ):
        super().__init__()

        self._messages = messages
        self._role = role
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._accumulator_frame = accumulator_frame
        self._interim_accumulator_frame = interim_accumulator_frame
        self._handle_interruptions = handle_interruptions
        self._expect_stripped_words = expect_stripped_words
        self._interrupt_double_accumulator = interrupt_double_accumulator

        self._reset()

    @property
    def messages(self):
        return self._messages

    @property
    def role(self):
        return self._role

    #
    # Frame processor
    #

    # Use cases implemented:
    #
    # S: Start, E: End, T: Transcription, I: Interim
    #
    #    S E             -> None                 -> User started speaking but no transcription.
    #    S T E           -> T                    -> Transcription between user started and stopped speaking.
    #    S E T           -> T                    -> Transcription after user stopped speaking.
    #    S I T E         -> T                    -> Transcription between user started and stopped speaking (with interims).
    #    S I E T         -> T                    -> Transcription after user stopped speaking (with interims).
    #    S I E I T       -> T                    -> Transcription after user stopped speaking (with interims).
    #    S E I T         -> T                    -> Transcription after user stopped speaking (with interims).
    #    S T1 I E S T2 E -> "T1 T2"              -> Merge two transcriptions if we got a first interim.
    #    S I E T1 I T2   -> T1 [Interruption] T2 -> Single user started/stopped, double transcription.
    #    S T1 E T2       -> T1 [Interruption] T2 -> Single user started/stopped, double transcription.
    #    S E T1 B T2     -> T1 [Interruption] T2 -> Single user started/stopped, double transcription.
    #    S E T1 T2       -> T1 [Interruption] T2 -> Single user started/stopped, double transcription.

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        send_aggregation = False

        if isinstance(frame, self._start_frame):
            self._aggregating = True
            self._seen_start_frame = True
            self._seen_end_frame = False
            await self.push_frame(frame, direction)
        elif isinstance(frame, self._end_frame):
            self._seen_end_frame = True
            self._seen_start_frame = False

            # We might have received the end frame but we might still be
            # aggregating (i.e. we have seen interim results but not the final
            # text).
            self._aggregating = self._seen_interim_results or len(self._aggregation) == 0

            # Send the aggregation if we are not aggregating anymore (i.e. no
            # more interim results received).
            send_aggregation = not self._aggregating
        elif isinstance(frame, self._accumulator_frame):
            if (
                self._interrupt_double_accumulator
                and self._sent_aggregation_after_last_interruption
            ):
                await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)
                self._sent_aggregation_after_last_interruption = False

            if self._expect_stripped_words:
                self._aggregation += f" {frame.text}" if self._aggregation else frame.text
            else:
                self._aggregation += frame.text

            # If we haven't seen the start frame but we got an accumulator frame
            # it means two things: it was develiver before the end frame or it
            # was delivered late. In both cases so we want to send the
            # aggregation.
            send_aggregation = not self._seen_start_frame

            # We just got our final result, so let's reset interim results.
            self._seen_interim_results = False
        elif self._interim_accumulator_frame and isinstance(frame, self._interim_accumulator_frame):
            if (
                self._interrupt_double_accumulator
                and self._sent_aggregation_after_last_interruption
            ):
                await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)
                self._sent_aggregation_after_last_interruption = False
            self._seen_interim_results = True
        elif isinstance(frame, StartInterruptionFrame) and self._handle_interruptions:
            await self._push_aggregation()
            # Reset anyways
            self._reset()
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMMessagesAppendFrame):
            self._add_messages(frame.messages)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            self._set_messages(frame.messages)
        elif isinstance(frame, LLMSetToolsFrame):
            self._set_tools(frame.tools)
        else:
            await self.push_frame(frame, direction)

        if send_aggregation:
            await self._push_aggregation()

        if isinstance(frame, self._end_frame):
            await self.push_frame(frame, direction)

    async def _push_aggregation(self):
        if len(self._aggregation) > 0:
            self._messages.append({"role": self._role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            self._sent_aggregation_after_last_interruption = True

            frame = LLMMessagesFrame(self._messages)
            await self.push_frame(frame)

    # TODO-CB: Types
    def _add_messages(self, messages):
        self._messages.extend(messages)

    def _set_messages(self, messages):
        self._reset()
        self._messages.clear()
        self._messages.extend(messages)

    def _set_tools(self, tools):
        # noop in the base class
        pass

    def _reset(self):
        self._aggregation = ""
        self._aggregating = False
        self._seen_start_frame = False
        self._seen_end_frame = False
        self._seen_interim_results = False
        self._sent_aggregation_after_last_interruption = False


class LLMUserResponseAggregator(LLMResponseAggregator):
    def __init__(self, messages: List[dict] = [], **kwargs):
        super().__init__(
            messages=messages,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            interim_accumulator_frame=InterimTranscriptionFrame,
            **kwargs,
        )


class LLMAssistantResponseAggregator(LLMResponseAggregator):
    def __init__(self, messages: List[dict] = [], **kwargs):
        super().__init__(
            messages=messages,
            role="assistant",
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True,
            **kwargs,
        )


class LLMContextAggregator(LLMResponseAggregator):
    def __init__(self, *, context: OpenAILLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    @property
    def context(self):
        return self._context

    def get_context_frame(self) -> OpenAILLMContextFrame:
        return OpenAILLMContextFrame(context=self._context)

    async def push_context_frame(self):
        frame = self.get_context_frame()
        await self.push_frame(frame)

    # TODO-CB: Types
    def _add_messages(self, messages):
        self._context.add_messages(messages)

    def _set_messages(self, messages):
        self._context.set_messages(messages)

    def _set_tools(self, tools: List):
        self._context.set_tools(tools)

    async def _push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message({"role": self._role, "content": self._aggregation})

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            self._sent_aggregation_after_last_interruption = True

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)


class LLMAssistantContextAggregator(LLMContextAggregator):
    def __init__(self, context: OpenAILLMContext, **kwargs):
        super().__init__(
            messages=[],
            context=context,
            role="assistant",
            start_frame=LLMFullResponseStartFrame,
            end_frame=LLMFullResponseEndFrame,
            accumulator_frame=TextFrame,
            handle_interruptions=True,
            **kwargs,
        )


class LLMUserContextAggregator(LLMContextAggregator):
    def __init__(self, context: OpenAILLMContext, **kwargs):
        super().__init__(
            messages=[],
            context=context,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            interim_accumulator_frame=InterimTranscriptionFrame,
            **kwargs,
        )


class LLMFullResponseAggregator(FrameProcessor):
    """This class aggregates Text frames between LLMFullResponseStartFrame and
    LLMFullResponseEndFrame, then emits the concatenated text as a single text
    frame.

    given the following frames:

        LLMFullResponseStartFrame()
        TextFrame("Hello,")
        TextFrame(" world.")
        TextFrame(" I am")
        TextFrame(" an LLM.")
        LLMFullResponseEndFrame()

    this processor will push,

        LLMFullResponseStartFrame()
        TextFrame("Hello, world. I am an LLM.")
        LLMFullResponseEndFrame()

    when passed the last frame.

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         if isinstance(frame, TextFrame):
    ...             print(frame.text)
    ...         else:
    ...             print(frame.__class__.__name__)

    >>> aggregator = LLMFullResponseAggregator()
    >>> asyncio.run(print_frames(aggregator, LLMFullResponseStartFrame()))
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello,")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" world.")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" I am")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" an LLM.")))
    >>> asyncio.run(print_frames(aggregator, LLMFullResponseEndFrame()))
    LLMFullResponseStartFrame
    Hello, world. I am an LLM.
    LLMFullResponseEndFrame

    """

    def __init__(self):
        super().__init__()
        self._aggregation = ""
        self._seen_start_frame = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._seen_start_frame = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._seen_start_frame = False
            await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame)
            self._aggregation = ""
        elif isinstance(frame, TextFrame) and self._seen_start_frame:
            self._aggregation += frame.text
        else:
            await self.push_frame(frame, direction)
