#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMMessagesFrame,
    LLMResponseStartFrame,
    TextFrame,
    LLMResponseEndFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame)


class LLMResponseAggregator(FrameProcessor):

    def __init__(
        self,
        *,
        messages: List[dict],
        role: str,
        start_frame,
        end_frame,
        accumulator_frame: TextFrame,
        interim_accumulator_frame: TextFrame | None = None
    ):
        super().__init__()

        self._messages = messages
        self._role = role
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._accumulator_frame = accumulator_frame
        self._interim_accumulator_frame = interim_accumulator_frame
        self._seen_start_frame = False
        self._seen_end_frame = False
        self._seen_interim_results = False

        self._aggregation = ""
        self._aggregating = False

    #
    # Frame processor
    #

    # Use cases implemented:
    #
    # S: Start, E: End, T: Transcription, I: Interim, X: Text
    #
    #        S E -> None
    #      S T E -> X
    #    S I T E -> X
    #    S I E T -> X
    #  S I E I T -> X
    #
    # The following case would not be supported:
    #
    #    S I E T1 I T2 -> X
    #
    # and T2 would be dropped.

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        send_aggregation = False

        if isinstance(frame, self._start_frame):
            self._seen_start_frame = True
            self._aggregating = True
        elif isinstance(frame, self._end_frame):
            self._seen_end_frame = True

            # We might have received the end frame but we might still be
            # aggregating (i.e. we have seen interim results but not the final
            # text).
            self._aggregating = self._seen_interim_results

            # Send the aggregation if we are not aggregating anymore (i.e. no
            # more interim results received).
            send_aggregation = not self._aggregating
        elif isinstance(frame, self._accumulator_frame):
            if self._aggregating:
                self._aggregation += f" {frame.text}"
                # We have recevied a complete sentence, so if we have seen the
                # end frame and we were still aggregating, it means we should
                # send the aggregation.
                send_aggregation = self._seen_end_frame

            # We just got our final result, so let's reset interim results.
            self._seen_interim_results = False
        elif self._interim_accumulator_frame and isinstance(frame, self._interim_accumulator_frame):
            self._seen_interim_results = True
        else:
            await self.push_frame(frame, direction)

        if send_aggregation:
            await self._push_aggregation()

    async def _push_aggregation(self):
        if len(self._aggregation) > 0:
            self._messages.append({"role": self._role, "content": self._aggregation})
            frame = LLMMessagesFrame(self._messages)
            await self.push_frame(frame)

            # Reset
            self._aggregation = ""
            self._seen_start_frame = False
            self._seen_end_frame = False
            self._seen_interim_results = False


class LLMAssistantResponseAggregator(LLMResponseAggregator):
    def __init__(self, messages: List[dict] = []):
        super().__init__(
            messages=messages,
            role="assistant",
            start_frame=LLMResponseStartFrame,
            end_frame=LLMResponseEndFrame,
            accumulator_frame=TextFrame
        )


class LLMUserResponseAggregator(LLMResponseAggregator):
    def __init__(self, messages: List[dict] = []):
        super().__init__(
            messages=messages,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            interim_accumulator_frame=InterimTranscriptionFrame
        )


class LLMFullResponseAggregator(FrameProcessor):
    """This class aggregates Text frames until it receives a
    LLMResponseEndFrame, then emits the concatenated text as
    a single text frame.

    given the following frames:

        TextFrame("Hello,")
        TextFrame(" world.")
        TextFrame(" I am")
        TextFrame(" an LLM.")
        LLMResponseEndFrame()]

    this processor will yield nothing for the first 4 frames, then

        TextFrame("Hello, world. I am an LLM.")
        LLMResponseEndFrame()

    when passed the last frame.

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         if isinstance(frame, TextFrame):
    ...             print(frame.text)
    ...         else:
    ...             print(frame.__class__.__name__)

    >>> aggregator = LLMFullResponseAggregator()
    >>> asyncio.run(print_frames(aggregator, TextFrame("Hello,")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" world.")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" I am")))
    >>> asyncio.run(print_frames(aggregator, TextFrame(" an LLM.")))
    >>> asyncio.run(print_frames(aggregator, LLMResponseEndFrame()))
    Hello, world. I am an LLM.
    LLMResponseEndFrame
    """

    def __init__(self):
        super().__init__()
        self._aggregation = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TextFrame):
            self._aggregation += frame.text
        elif isinstance(frame, LLMResponseEndFrame):
            await self.push_frame(TextFrame(self._aggregation))
            await self.push_frame(frame)
            self._aggregation = ""
        else:
            await self.push_frame(frame, direction)
