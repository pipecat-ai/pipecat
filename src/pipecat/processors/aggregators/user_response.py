#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import TextFrame
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator


class UserResponseAggregator(LLMUserResponseAggregator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            frame = TextFrame(self._aggregation.strip())

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            await self.push_frame(frame)

            # Reset our accumulator state.
            await self.reset()
