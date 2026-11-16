#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import LLMTextFrame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.text_transformer import StatelessTextTransformer
from pipecat.tests.utils import run_test


class TestStatelessTextTransformer(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_frame_type_and_fields(self):
        down_frames, _ = await run_test(
            StatelessTextTransformer(str.upper),
            frames_to_send=[
                LLMTextFrame(text="hello"),
                TranscriptionFrame(text="hi", user_id="user1", timestamp="now"),
            ],
            expected_down_frames=[LLMTextFrame, TranscriptionFrame],
        )

        llm_text, transcription = down_frames
        self.assertEqual(llm_text.text, "HELLO")
        self.assertTrue(llm_text.includes_inter_frame_spaces)
        self.assertEqual(transcription.text, "HI")
        self.assertEqual(transcription.user_id, "user1")

    async def test_async_transform_keeps_direction(self):
        async def upper(text: str) -> str:
            return text.upper()

        _, up_frames = await run_test(
            StatelessTextTransformer(upper),
            frames_to_send=[TextFrame(text="hello")],
            frames_to_send_direction=FrameDirection.UPSTREAM,
            expected_down_frames=[],
            expected_up_frames=[TextFrame],
        )

        self.assertEqual(up_frames[0].text, "HELLO")
