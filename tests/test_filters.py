#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.filters.wake_check_filter import WakeCheckFilter
from tests.utils import run_test


class TestIdentifyFilter(unittest.IsolatedAsyncioTestCase):
    async def test_identity(self):
        filter = IdentityFilter()
        frames_to_send = [UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()]
        expected_returned_frames = [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
        await run_test(filter, frames_to_send, expected_returned_frames)


class TestWakeCheckFilter(unittest.IsolatedAsyncioTestCase):
    async def test_no_wake_word(self):
        filter = WakeCheckFilter(wake_phrases=["Hey, Pipecat"])
        frames_to_send = [TranscriptionFrame(user_id="test", text="Phrase 1", timestamp="")]
        expected_returned_frames = []
        await run_test(filter, frames_to_send, expected_returned_frames)

    async def test_wake_word(self):
        filter = WakeCheckFilter(wake_phrases=["Hey, Pipecat"])
        frames_to_send = [
            TranscriptionFrame(user_id="test", text="Hey, Pipecat", timestamp=""),
            TranscriptionFrame(user_id="test", text="Phrase 1", timestamp=""),
        ]
        expected_returned_frames = [TranscriptionFrame, TranscriptionFrame]
        (received_down, _) = await run_test(filter, frames_to_send, expected_returned_frames)
        assert received_down[-1].text == "Phrase 1"
