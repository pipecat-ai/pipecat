#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
)
from pipecat.turns.user_mute import (
    AlwaysUserMuteStrategy,
    FirstSpeechUserMuteStrategy,
    FunctionCallUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)


class TestAlwaysUserMuteStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_user_mute_strategy(self):
        strategy = AlwaysUserMuteStrategy()

        self.assertTrue(await strategy.process_frame(BotStartedSpeakingFrame()))
        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        self.assertFalse(await strategy.process_frame(BotStoppedSpeakingFrame()))
        self.assertFalse(await strategy.process_frame(InterruptionFrame()))


class TestFirstSpeechUserMuteStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_user_mute_strategy(self):
        strategy = FirstSpeechUserMuteStrategy()

        self.assertFalse(await strategy.process_frame(InterruptionFrame()))
        self.assertTrue(await strategy.process_frame(BotStartedSpeakingFrame()))
        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        self.assertFalse(await strategy.process_frame(BotStoppedSpeakingFrame()))
        self.assertFalse(await strategy.process_frame(InterruptionFrame()))


class TestMuteUntilFirstBotCompleteUserMuteStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_user_mute_strategy(self):
        strategy = MuteUntilFirstBotCompleteUserMuteStrategy()

        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        self.assertTrue(await strategy.process_frame(BotStartedSpeakingFrame()))
        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        self.assertFalse(await strategy.process_frame(BotStoppedSpeakingFrame()))
        self.assertFalse(await strategy.process_frame(InterruptionFrame()))


class TestFunctionCallUserMuteStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_user_mute_strategy(self):
        strategy = FunctionCallUserMuteStrategy()

        self.assertFalse(await strategy.process_frame(InterruptionFrame()))
        # First function call (cancelled)
        self.assertTrue(
            await strategy.process_frame(
                FunctionCallsStartedFrame(
                    function_calls=[
                        FunctionCallFromLLM(
                            function_name="fn_1", tool_call_id="1", arguments={}, context=None
                        )
                    ]
                )
            )
        )
        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        self.assertFalse(
            await strategy.process_frame(
                FunctionCallCancelFrame(function_name="fn_1", tool_call_id="1")
            )
        )
        self.assertFalse(await strategy.process_frame(InterruptionFrame()))

        # Second function call (finished)
        self.assertTrue(
            await strategy.process_frame(
                FunctionCallsStartedFrame(
                    function_calls=[
                        FunctionCallFromLLM(
                            function_name="fn_2", tool_call_id="2", arguments={}, context=None
                        )
                    ]
                )
            )
        )
        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        self.assertFalse(
            await strategy.process_frame(
                FunctionCallResultFrame(
                    function_name="fn_2", tool_call_id="2", arguments={}, result={}
                )
            )
        )
        self.assertFalse(await strategy.process_frame(InterruptionFrame()))

        # Multiple function calls
        self.assertTrue(
            await strategy.process_frame(
                FunctionCallsStartedFrame(
                    function_calls=[
                        FunctionCallFromLLM(
                            function_name="fn_3", tool_call_id="3", arguments={}, context=None
                        ),
                        FunctionCallFromLLM(
                            function_name="fn_4", tool_call_id="4", arguments={}, context=None
                        ),
                    ]
                )
            )
        )
        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        # First function call is done, we still should be muted since there's
        # another one ongoing.
        self.assertTrue(
            await strategy.process_frame(
                FunctionCallResultFrame(
                    function_name="fn_3", tool_call_id="3", arguments={}, result={}
                )
            )
        )
        self.assertTrue(await strategy.process_frame(InterruptionFrame()))
        # Last function call finishes.
        self.assertFalse(
            await strategy.process_frame(
                FunctionCallResultFrame(
                    function_name="fn_4", tool_call_id="4", arguments={}, result={}
                )
            )
        )
        self.assertFalse(await strategy.process_frame(InterruptionFrame()))


if __name__ == "__main__":
    unittest.main()
