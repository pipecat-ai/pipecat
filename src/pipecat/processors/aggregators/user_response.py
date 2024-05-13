#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame)


class ResponseAggregator(FrameProcessor):
    """This frame processor aggregates frames between a start and an end frame
    into complete text frame sentences.

    For example, frame input/output:
        UserStartedSpeakingFrame() -> None
        TranscriptionFrame("Hello,") -> None
        TranscriptionFrame(" world.") -> None
        UserStoppedSpeakingFrame() -> TextFrame("Hello world.")

    Doctest:
    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         if isinstance(frame, TextFrame):
    ...             print(frame.text)

    >>> aggregator = ResponseAggregator(start_frame = UserStartedSpeakingFrame,
    ...                                 end_frame=UserStoppedSpeakingFrame,
    ...                                 accumulator_frame=TranscriptionFrame,
    ...                                 pass_through=False)
    >>> asyncio.run(print_frames(aggregator, UserStartedSpeakingFrame()))
    >>> asyncio.run(print_frames(aggregator, TranscriptionFrame("Hello,", 1, 1)))
    >>> asyncio.run(print_frames(aggregator, TranscriptionFrame("world.",  1, 2)))
    >>> asyncio.run(print_frames(aggregator, UserStoppedSpeakingFrame()))
    Hello, world.

    """

    def __init__(
        self,
        *,
        start_frame,
        end_frame,
        accumulator_frame: TextFrame,
        interim_accumulator_frame: TextFrame | None = None
    ):
        super().__init__()

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
            await self.push_frame(TextFrame(self._aggregation.strip()))

            # Reset
            self._aggregation = ""
            self._seen_start_frame = False
            self._seen_end_frame = False
            self._seen_interim_results = False


class UserResponseAggregator(ResponseAggregator):
    def __init__(self):
        super().__init__(
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            interim_accumulator_frame=InterimTranscriptionFrame,
        )
