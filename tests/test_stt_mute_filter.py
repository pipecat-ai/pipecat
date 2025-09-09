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
    InterimTranscriptionFrame,
    STTMuteFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.tests.utils import SleepFrame, run_test


class TestSTTMuteFilter(unittest.IsolatedAsyncioTestCase):
    async def test_first_speech_strategy(self):
        filter = STTMuteFilter(config=STTMuteConfig(strategies={STTMuteStrategy.FIRST_SPEECH}))

        frames_to_send = [
            BotStartedSpeakingFrame(),  # First bot speech starts
            VADUserStartedSpeakingFrame(),  # Should be suppressed
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            VADUserStoppedSpeakingFrame(),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # First bot speech ends
            BotStartedSpeakingFrame(),  # Second bot speech
            VADUserStartedSpeakingFrame(),  # Should pass through
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            VADUserStoppedSpeakingFrame(),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStoppedSpeakingFrame(),
        ]

        expected_returned_frames = [
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
            BotStartedSpeakingFrame,
            VADUserStartedSpeakingFrame,  # Now passes through
            UserStartedSpeakingFrame,  # Now passes through
            InputAudioRawFrame,  # Now passes through
            VADUserStoppedSpeakingFrame,  # Now passes through
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
            VADUserStartedSpeakingFrame(),  # Should be suppressed
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            VADUserStoppedSpeakingFrame(),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # First speech ends
            VADUserStartedSpeakingFrame(),  # Should pass through
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            VADUserStoppedSpeakingFrame(),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStartedSpeakingFrame(),  # Second speech starts
            VADUserStartedSpeakingFrame(),  # Should be suppressed again
            UserStartedSpeakingFrame(),  # Should be suppressed again
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed again
            VADUserStoppedSpeakingFrame(),  # Should be suppressed again
            UserStoppedSpeakingFrame(),  # Should be suppressed again
            BotStoppedSpeakingFrame(),  # Second speech ends
        ]

        expected_returned_frames = [
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
            VADUserStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            VADUserStoppedSpeakingFrame,
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

    async def test_transcription_frames_with_always_strategy(self):
        filter = STTMuteFilter(config=STTMuteConfig(strategies={STTMuteStrategy.ALWAYS}))

        frames_to_send = [
            # Bot speaking - should mute
            BotStartedSpeakingFrame(),
            SleepFrame(),  # Wait for StartedSpeaking to process
            InterimTranscriptionFrame(
                user_id="user1", text="This should be suppressed", timestamp="1234567890"
            ),
            TranscriptionFrame(
                user_id="user1", text="This should be suppressed", timestamp="1234567890"
            ),
            SleepFrame(),  # Wait for transcription frames to queue
            BotStoppedSpeakingFrame(),
            # Bot not speaking - should pass through
            InterimTranscriptionFrame(
                user_id="user1", text="This should pass", timestamp="1234567891"
            ),
            TranscriptionFrame(
                user_id="user1", text="This should pass through", timestamp="1234567891"
            ),
        ]

        expected_returned_frames = [
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
            InterimTranscriptionFrame,  # Only passes through after bot stops speaking
            TranscriptionFrame,  # Only passes through after bot stops speaking
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
    #         VADUserStartedSpeakingFrame(),  # Should pass through initially
    #         UserStartedSpeakingFrame(),  # Should pass through initially
    #         VADUserStoppedSpeakingFrame(),
    #         UserStoppedSpeakingFrame(),
    #         FunctionCallInProgressFrame(
    #             function_name="get_weather",
    #             tool_call_id="call_123",
    #             arguments='{"location": "San Francisco"}',
    #         ),  # Start function call
    #         VADUserStartedSpeakingFrame(),  # Should be suppressed
    #         UserStartedSpeakingFrame(),  # Should be suppressed
    #         VADUserStoppedSpeakingFrame(),  # Should be suppressed
    #         UserStoppedSpeakingFrame(),  # Should be suppressed
    #         FunctionCallResultFrame(
    #             function_name="get_weather",
    #             tool_call_id="call_123",
    #             arguments='{"location": "San Francisco"}',
    #             result={"temperature": 22},
    #         ),  # End function call
    #         VADUserStartedSpeakingFrame(),  # Should pass through again
    #         UserStartedSpeakingFrame(),  # Should pass through again
    #         VADUserStoppedSpeakingFrame(),
    #         UserStoppedSpeakingFrame(),
    #     ]

    #     expected_returned_frames = [
    #         VADUserStartedSpeakingFrame,
    #         UserStartedSpeakingFrame,
    #         VADUserStoppedSpeakingFrame,
    #         UserStoppedSpeakingFrame,
    #         FunctionCallInProgressFrame,
    #         STTMuteFrame,  # mute=True
    #         FunctionCallResultFrame,
    #         STTMuteFrame,  # mute=False
    #         VADUserStartedSpeakingFrame,
    #         UserStartedSpeakingFrame,
    #         VADUserStoppedSpeakingFrame,
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
            VADUserStartedSpeakingFrame(),  # Should be suppressed (starts muted)
            UserStartedSpeakingFrame(),  # Should be suppressed (starts muted)
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            VADUserStoppedSpeakingFrame(),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStartedSpeakingFrame(),  # First bot speech
            VADUserStartedSpeakingFrame(),  # Should be suppressed
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            VADUserStoppedSpeakingFrame(),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # First speech ends, unmutes
            VADUserStartedSpeakingFrame(),  # Should pass through
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            VADUserStoppedSpeakingFrame(),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStartedSpeakingFrame(),  # Second speech
            VADUserStartedSpeakingFrame(),  # Should pass through
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            VADUserStoppedSpeakingFrame(),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStoppedSpeakingFrame(),
        ]

        expected_returned_frames = [
            STTMuteFrame,  # mute=True after first speech
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False after first speech
            VADUserStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            VADUserStoppedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            VADUserStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            VADUserStoppedSpeakingFrame,
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
            VADUserStartedSpeakingFrame(),  # Should pass through
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            VADUserStoppedSpeakingFrame(),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
            BotStartedSpeakingFrame(),  # Bot starts speaking
            VADUserStartedSpeakingFrame(),  # Should be suppressed
            UserStartedSpeakingFrame(),  # Should be suppressed
            InputAudioRawFrame(
                audio=b"", sample_rate=16000, num_channels=1
            ),  # Should be suppressed
            VADUserStoppedSpeakingFrame(),  # Should be suppressed
            UserStoppedSpeakingFrame(),  # Should be suppressed
            BotStoppedSpeakingFrame(),  # Bot stops speaking
            VADUserStartedSpeakingFrame(),  # Should pass through
            UserStartedSpeakingFrame(),  # Should pass through
            InputAudioRawFrame(audio=b"", sample_rate=16000, num_channels=1),  # Should pass through
            VADUserStoppedSpeakingFrame(),  # Should pass through
            UserStoppedSpeakingFrame(),  # Should pass through
        ]

        expected_returned_frames = [
            VADUserStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            VADUserStoppedSpeakingFrame,
            UserStoppedSpeakingFrame,
            BotStartedSpeakingFrame,
            STTMuteFrame,  # mute=True
            BotStoppedSpeakingFrame,
            STTMuteFrame,  # mute=False
            VADUserStartedSpeakingFrame,
            UserStartedSpeakingFrame,
            InputAudioRawFrame,
            VADUserStoppedSpeakingFrame,
            UserStoppedSpeakingFrame,
        ]

        await run_test(
            filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )
