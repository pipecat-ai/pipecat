#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`GatedLLMContextAggregator` (issue #6864 regression)."""

import pytest

from pipecat.frames.frames import StartFrame, TextFrame
from pipecat.processors.aggregators.gated_llm_context import GatedLLMContextAggregator
from pipecat.tests.utils import run_test
from pipecat.utils.sync.event_notifier import EventNotifier


@pytest.mark.asyncio
async def test_start_frame_is_pushed_downstream_once():
    # A StartFrame handled by the first branch must not also fall through to
    # the default branch and be pushed a second time (issue #6864).
    await run_test(
        GatedLLMContextAggregator(notifier=EventNotifier()),
        frames_to_send=[TextFrame(text="Hello")],
        expected_down_frames=[StartFrame, TextFrame],
        ignore_start=False,
    )
