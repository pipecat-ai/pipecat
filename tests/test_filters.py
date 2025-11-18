#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.frame_filter import FrameFilter
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.filters.wake_check_filter import WakeCheckFilter
from pipecat.tests.utils import run_test


class TestIdentifyFilter(unittest.IsolatedAsyncioTestCase):
    async def test_identity(self):
        filter = IdentityFilter()
        frames_to_send = [UserStartedSpeakingFrame(), UserStoppedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )


class TestFrameFilter(unittest.IsolatedAsyncioTestCase):
    async def test_text_frame(self):
        filter = FrameFilter(types=(TextFrame,))
        frames_to_send = [TextFrame(text="Hello Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_end_frame(self):
        filter = FrameFilter(types=(EndFrame,))
        frames_to_send = [EndFrame()]
        expected_down_frames = [EndFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            send_end_frame=False,
        )

    async def test_system_frame(self):
        filter = FrameFilter(types=())
        frames_to_send = [UserStartedSpeakingFrame()]
        expected_down_frames = [UserStartedSpeakingFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )


class TestFunctionFilter(unittest.IsolatedAsyncioTestCase):
    async def test_passthrough(self):
        async def passthrough(frame: Frame):
            return True

        filter = FunctionFilter(filter=passthrough)
        frames_to_send = [TextFrame(text="Hello Pipecat!")]
        expected_down_frames = [TextFrame]
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_no_passthrough(self):
        async def no_passthrough(frame: Frame):
            return False

        filter = FunctionFilter(filter=no_passthrough)
        frames_to_send = [TextFrame(text="Hello Pipecat!")]
        expected_down_frames = []
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )


class TestWakeCheckFilter(unittest.IsolatedAsyncioTestCase):
    async def test_no_wake_word(self):
        filter = WakeCheckFilter(wake_phrases=["Hey, Pipecat"])
        frames_to_send = [TranscriptionFrame(user_id="test", text="Phrase 1", timestamp="")]
        expected_down_frames = []
        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

    async def test_wake_word(self):
        filter = WakeCheckFilter(wake_phrases=["Hey, Pipecat"])
        frames_to_send = [
            TranscriptionFrame(user_id="test", text="Hey, Pipecat", timestamp=""),
            TranscriptionFrame(user_id="test", text="Phrase 1", timestamp=""),
        ]
        expected_down_frames = [TranscriptionFrame, TranscriptionFrame]
        (received_down, _) = await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-1].text == "Phrase 1"
