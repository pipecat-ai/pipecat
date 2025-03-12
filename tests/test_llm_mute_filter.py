import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DTMFFrame,
    KeypadEntry,
    TranscriptionFrame,
)
from pipecat.processors.filters.llm_mute_filter import LLMMuteConfig, LLMMuteFilter, LLMMuteStrategy
from pipecat.tests.utils import SleepFrame, run_test


class TestLLMMuteFilter(unittest.IsolatedAsyncioTestCase):
    async def test_first_speech_strategy(self):
        """When using the FIRST_SPEECH strategy:
        - The first bot speech should trigger mute and suppress VAD frames.
        - Once bot speech ends, new transcription frames pass through.
        - Also verifies that only transcriptions from unmuted periods are recorded.
        """
        llm_filter = LLMMuteFilter(
            stt_service=None, config=LLMMuteConfig(strategies={LLMMuteStrategy.FIRST_SPEECH})
        )

        frames_to_send = [
            BotStartedSpeakingFrame(),  # Bot starts speaking → should trigger mute internally.
            TranscriptionFrame(
                text="Hello",
                user_id="None",
                timestamp="None",
            ),  # VAD frame; should be suppressed (while muted).
            SleepFrame(sleep=0.2),  # Sleep to simulate bot speech.
            BotStoppedSpeakingFrame(),  # Bot stops → mute turns off.
            SleepFrame(sleep=0.1),
            TranscriptionFrame(
                text="World",
                user_id="None",
                timestamp="None",
            ),  # Allowed transcription → should be pushed and recorded.
            DTMFFrame(button=KeypadEntry.FIVE),  # Non-VAD frame → always passed through.
        ]

        # Expected frames are those that are pushed (the suppressed TranscriptionFrame "Hello" is omitted).
        expected_returned_frames = [
            BotStartedSpeakingFrame,  # always passed (non-VAD)
            BotStoppedSpeakingFrame,  # always passed (non-VAD)
            TranscriptionFrame,  # only "World" passes since mute is off
            DTMFFrame,  # always passed
        ]

        await run_test(
            llm_filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )
        self.assertEqual(llm_filter._user_tx, ["World"])

    async def test_first_speech_strategy_all_tx_missing(self):
        llm_filter = LLMMuteFilter(
            stt_service=None, config=LLMMuteConfig(strategies={LLMMuteStrategy.FIRST_SPEECH})
        )

        frames_to_send = [
            BotStartedSpeakingFrame(),  # Bot starts speaking → should trigger mute internally.
            TranscriptionFrame(
                text="Hello",
                user_id="None",
                timestamp="None",
            ),  # VAD frame; should be suppressed (while muted).
            TranscriptionFrame(
                text="World",
                user_id="None",
                timestamp="None",
            ),  # VAD frame; should be suppressed (while muted).
            SleepFrame(sleep=0.6),  # Sleep to simulate bot speech.
            BotStoppedSpeakingFrame(),  # Bot stops → mute turns off.
            DTMFFrame(button=KeypadEntry.FIVE),  # Non-VAD frame → always passed through.
        ]

        # Expected frames are those that are pushed (the suppressed TranscriptionFrame "Hello" is omitted).
        expected_returned_frames = [
            BotStartedSpeakingFrame,  # always passed (non-VAD)
            BotStoppedSpeakingFrame,  # always passed (non-VAD)
            DTMFFrame,  # always passed
        ]

        await run_test(
            llm_filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )
        self.assertEqual(llm_filter._user_tx, [])

    async def test_first_speech_strategy_voice_mail_case(self):
        llm_filter = LLMMuteFilter(
            stt_service=None, config=LLMMuteConfig(strategies={LLMMuteStrategy.FIRST_SPEECH})
        )

        frames_to_send = [
            BotStartedSpeakingFrame(),  # Bot starts speaking → should trigger mute internally.
            TranscriptionFrame(
                text="Voice Mail",
                user_id="None",
                timestamp="None",
            ),  # VAD frame; should be suppressed (while muted).
            TranscriptionFrame(
                text="World",
                user_id="None",
                timestamp="None",
            ),  # VAD frame; should be suppressed (while muted).
            SleepFrame(sleep=0.6),  # Sleep to simulate bot speech.
            BotStoppedSpeakingFrame(),  # Bot stops → mute turns off.
            DTMFFrame(button=KeypadEntry.FIVE),  # Non-VAD frame → always passed through.
        ]

        # Expected frames are those that are pushed (the suppressed TranscriptionFrame "Hello" is omitted).
        expected_returned_frames = [
            BotStartedSpeakingFrame,  # always passed (non-VAD)
            TranscriptionFrame,  # only "Voice Mail" passes since mute is on
            BotStoppedSpeakingFrame,  # always passed (non-VAD)
            DTMFFrame,  # always passed
        ]

        await run_test(
            llm_filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )
        self.assertEqual(llm_filter._user_tx, ["Voice Mail"])

    async def test_first_speech_strategy_all_tx_present(self):
        """When using the FIRST_SPEECH strategy:
        - The first bot speech should trigger mute and suppress VAD frames.
        - Once bot speech ends, new transcription frames pass through.
        - Also verifies that all transcriptions are recorded.
        """
        llm_filter = LLMMuteFilter(
            stt_service=None, config=LLMMuteConfig(strategies={LLMMuteStrategy.FIRST_SPEECH})
        )

        frames_to_send = [
            BotStartedSpeakingFrame(),  # Bot starts speaking → should trigger mute internally.
            SleepFrame(sleep=0),  # Sleep to simulate bot speech.
            BotStoppedSpeakingFrame(),  # Bot stops → mute turns off
            TranscriptionFrame(
                text="Hello",
                user_id="None",
                timestamp="None",
            ),  # VAD frame; should be suppressed (while muted).
            TranscriptionFrame(
                text="World",
                user_id="None",
                timestamp="None",
            ),  # Allowed transcription → should be pushed and recorded.
            DTMFFrame(button=KeypadEntry.FIVE),  # Non-VAD frame → always passed through.
        ]

        # Expected frames are those that are pushed (the suppressed TranscriptionFrame "Hello" is omitted).
        expected_returned_frames = [
            BotStartedSpeakingFrame,  # always passed (non-VAD)
            BotStoppedSpeakingFrame,  # always passed (non-VAD)
            TranscriptionFrame,  # Hello passes
            TranscriptionFrame,  # World passes
            DTMFFrame,  # always passed
        ]

        await run_test(
            llm_filter,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )
        self.assertEqual(
            llm_filter._user_tx,
            [
                "Hello",
                "World",
            ],
        )
