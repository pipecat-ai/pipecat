#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import StartFrame, TextFrame
from pipecat.processors.aggregators.gated_llm_context import GatedLLMContextAggregator
from pipecat.tests.utils import run_test
from pipecat.utils.sync.event_notifier import EventNotifier


class TestGatedLLMContextAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_start_frame_pushed_once(self):
        aggregator = GatedLLMContextAggregator(notifier=EventNotifier())
        await run_test(
            aggregator,
            frames_to_send=[TextFrame(text="Hello")],
            expected_down_frames=[StartFrame, TextFrame],
            ignore_start=False,
        )


if __name__ == "__main__":
    unittest.main()
