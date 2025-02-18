#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    STTMuteFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.tests.utils import SleepFrame, run_test


class TestSTTMuteFilter(unittest.IsolatedAsyncioTestCase):
    async def test_first_speech_strategy(self):
        filter = STTMuteFilter(config=STTMuteConfig(strategies={STTMuteStrategy.FIRST_SPEECH}))

        frames_to_send = [
            BotStartedSpeakingFrame(),  # First bot speech starts
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # First bot speech ends
            BotStartedSpeakingFrame(),  # Second bot speech
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStoppedSpeakingFrame(),
        ]

        expected_returned_frames = [
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
            BotStartedSpeakingFrame,
            UserStartedSpeakingFrame,  # Now passes through
            InputAudioRawFrame,  # Now passes through
            UserStoppedSpeakingFrame,  # Now passes through
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

    async def test_always_strategy(self):
        filter = STTMuteFilter(config=STTMuteConfig(strategies={STTMuteStrategy.ALWAYS}))

        frames_to_send = [
            BotStartedSpeakingFrame(),  # First speech starts
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # First speech ends
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStartedSpeakingFrame(),  # Second speech starts
            UserStartedSpeakingFrame(),  # Should be suppressed again
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed again
            UserStoppedSpeakingFrame(),  # Should be suppressed again
            BotStoppedSpeakingFrame(),  # Second speech ends
        ]

        expected_returned_frames = [
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
        ]

        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

    # TODO: Revisit once we figure out how to test SystemFrames and DataFrames
    # async def test_function_call_strategy(self):
    #     filter = STTMuteFilter(config=STTMuteConfig(strategies={STTMuteStrategy.FUNCTION_CALL}))

    #     frames_to_send = [
    #         UserStartedSpeakingFrame(),  # Should pass through initially
    #         UserStoppedSpeakingFrame(),
    #         FunctionCallInProgressFrame(
    #             function_name="get_weather",
    #             tool_call_id="call_123",
    #             arguments='{"location": "San Francisco"}',
    #         ),  # Start function call
    #         UserStartedSpeakingFrame(),  # Should be suppressed
    #         UserStoppedSpeakingFrame(),  # Should be suppressed
    #         FunctionCallResultFrame(
    #             function_name="get_weather",
    #             tool_call_id="call_123",
    #             arguments='{"location": "San Francisco"}',
    #             result={"temperature": 22},
    #         ),  # End function call
    #         UserStartedSpeakingFrame(),  # Should pass through again
    #         UserStoppedSpeakingFrame(),
    #     ]

    #     expected_returned_frames = [
    #         UserStartedSpeakingFrame,
    #         UserStoppedSpeakingFrame,
    #         FunctionCallInProgressFrame,
    #         STTMuteFrame,  # mute=True
    #         FunctionCallResultFrame,
    #         STTMuteFrame,  # mute=False
    #         UserStartedSpeakingFrame,
    #         UserStoppedSpeakingFrame,
    #     ]

    #     await run_test(
    #         filter,
    #         frames_to_send=frames_to_send,
    #         expected_down_frames=expected_returned_frames,
    #     )

    async def test_mute_until_first_bot_complete_strategy(self):
        filter = STTMuteFilter(
            config=STTMuteConfig(strategies={STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE})
        )

        frames_to_send = [
            UserStartedSpeakingFrame(),  # Should be suppressed (starts muted)
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStartedSpeakingFrame(),  # First bot speech
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # First speech ends, unmutes
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStartedSpeakingFrame(),  # Second speech
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStoppedSpeakingFrame(),
        ]

        expected_returned_frames = [
            STTMuteFrame,  # mute=True after first speech
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False after first speech
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            UserStoppedSpeakingFrame,
            BotStoppedSpeakingFrame,
        ]

        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

    async def test_incompatible_strategies(self):
        with self.assertRaises(ValueError):
            STTMuteFilter(
                config=STTMuteConfig(
                    strategies={
                        STTMuteStrategy.FIRST_SPEECH,
                        STTMuteStrategy.MUTE_UNTIL_FIRST_BOT_COMPLETE,
                    }
                )
            )

    async def test_custom_strategy(self):
        async def custom_mute_logic(processor: STTMuteFilter) -> bool:
            return processor._bot_is_speaking

        filter = STTMuteFilter(
            config=STTMuteConfig(
                strategies={STTMuteStrategy.CUSTOM},
                should_mute_callback=custom_mute_logic,
            )
        )

        frames_to_send = [
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStartedSpeakingFrame(),  # Bot starts speaking
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # Bot stops speaking
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
        ]

        expected_returned_frames = [
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            UserStoppedSpeakingFrame,
        ]

        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )
