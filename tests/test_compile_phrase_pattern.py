#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.wake_phrase_user_turn_start_strategy import (
    WakePhraseUserTurnStartStrategy,
    _WakeState,
)
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.string import compile_phrase_pattern
from tests.frame_processor_helpers import frame_processor_setup


class TestCompilePhrasePattern(unittest.TestCase):
    def test_spaced_phrase_needs_word_boundaries(self):
        pattern = compile_phrase_pattern("hey pipecat")
        self.assertIsNotNone(pattern.search("well Hey   Pipecat what's up"))
        self.assertIsNotNone(pattern.search("heypipecat"))
        self.assertIsNone(pattern.search("they pipecats"))

    def test_chinese_phrase_inside_unspaced_sentence(self):
        pattern = compile_phrase_pattern("小助手")
        self.assertIsNotNone(pattern.search("小助手今天天气怎么样"))
        self.assertIsNotNone(pattern.search("你好小助手"))
        self.assertIsNotNone(pattern.search(" 你好小助手今天天气怎么样"))
        self.assertIsNone(pattern.search("你好小明"))

    def test_japanese_phrase_inside_unspaced_sentence(self):
        pattern = compile_phrase_pattern("ねえアシスタント")
        self.assertIsNotNone(pattern.search("ねえアシスタント今日の天気は"))

    def test_mixed_phrase_keeps_boundary_on_latin_side(self):
        pattern = compile_phrase_pattern("hey 小助手")
        self.assertIsNotNone(pattern.search("hey 小助手今天天气怎么样"))
        self.assertIsNone(pattern.search("they 小助手"))

    def test_empty_phrase_keeps_previous_behavior(self):
        self.assertEqual(compile_phrase_pattern("").pattern, r"\b\b")


class TestWakePhraseUnspacedScripts(unittest.IsolatedAsyncioTestCase):
    async def test_chinese_wake_phrase_followed_by_request(self):
        strategy = WakePhraseUserTurnStartStrategy(phrases=["小助手"])
        await strategy.setup(frame_processor_setup(TaskManager()))
        await asyncio.sleep(0)

        result = await strategy.process_frame(
            TranscriptionFrame(text="小助手，今天天气怎么样？", user_id="user1", timestamp="")
        )
        self.assertEqual(result, ProcessFrameResult.STOP)
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()

    async def test_chinese_wake_phrase_inside_sentence(self):
        strategy = WakePhraseUserTurnStartStrategy(phrases=["小助手"])
        await strategy.setup(frame_processor_setup(TaskManager()))
        await asyncio.sleep(0)

        await strategy.process_frame(
            TranscriptionFrame(text="你好小助手帮我订一张机票", user_id="user1", timestamp="")
        )
        self.assertEqual(strategy.state, _WakeState.AWAKE)

        await strategy.cleanup()


if __name__ == "__main__":
    unittest.main()
