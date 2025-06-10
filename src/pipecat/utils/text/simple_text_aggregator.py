#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


class SimpleTextAggregator(BaseTextAggregator):
    """This is a simple text aggregator. It aggregates text until an end of
    sentence is found.

    """

    def __init__(self):
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    async def aggregate(self, text: str) -> Optional[str]:
        result: Optional[str] = None

        self._text += text

        eos_end_marker = match_endofsentence(self._text)
        if eos_end_marker:
            result = self._text[:eos_end_marker]
            self._text = self._text[eos_end_marker:]

        return result

    async def handle_interruption(self):
        self._text = ""

    async def reset(self):
        self._text = ""
