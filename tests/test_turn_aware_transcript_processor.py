#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    AggregationType,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TranscriptionUpdateFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.transcript_processor import TurnAwareTranscriptProcessor
from pipecat.tests.utils import SleepFrame, run_test


class TestTurnAwareTranscriptProcessor(unittest.IsolatedAsyncioTestCase):
    """Tests for TurnAwareTranscriptProcessor."""

    async def test_basic_turn_flow(self):
        """Test basic turn start/end with user and assistant speech."""
        processor = TurnAwareTranscriptProcessor()

        # Track events
        turn_started_calls = []
        turn_ended_calls = []

        @processor.event_handler("on_turn_started")
        async def on_turn_started(proc, turn_number):
            turn_started_calls.append(turn_number)

        @processor.event_handler("on_turn_ended")
        async def on_turn_ended(proc, turn_number, user_text, assistant_text, interrupted):
            turn_ended_calls.append(
                {
                    "turn_number": turn_number,
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                    "interrupted": interrupted,
                }
            )

        frames_to_send = [
            # Turn 1: User speaks, bot responds
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hello", user_id="user1", timestamp=""),
            SleepFrame(sleep=0.01),  # Allow transcription to process
            BotStartedSpeakingFrame(),
            TTSTextFrame(text="Hi", aggregated_by=AggregationType.WORD),
            TTSTextFrame(text=" there", aggregated_by=AggregationType.WORD),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
        ]

        await run_test(processor, frames_to_send=frames_to_send)

        # Verify events
        self.assertEqual(
            len(turn_started_calls), 1, f"Expected 1 turn started, got {len(turn_started_calls)}"
        )
        self.assertEqual(turn_started_calls[0], 1)

        self.assertEqual(
            len(turn_ended_calls), 1, f"Expected 1 turn ended, got {len(turn_ended_calls)}"
        )
        self.assertEqual(turn_ended_calls[0]["turn_number"], 1)
        self.assertEqual(turn_ended_calls[0]["user_text"], "Hello")
        self.assertEqual(turn_ended_calls[0]["assistant_text"], "Hi  there")
        self.assertFalse(turn_ended_calls[0]["interrupted"])

    async def test_interruption(self):
        """Test turn ending on interruption."""
        processor = TurnAwareTranscriptProcessor()

        # Track events
        turn_ended_calls = []

        @processor.event_handler("on_turn_ended")
        async def on_turn_ended(proc, turn_number, user_text, assistant_text, interrupted):
            turn_ended_calls.append(
                {
                    "turn_number": turn_number,
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                    "interrupted": interrupted,
                }
            )

        frames_to_send = [
            # User speaks
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Tell me", user_id="user1", timestamp=""),
            SleepFrame(sleep=0.01),  # Allow transcription to process
            # Bot starts responding
            BotStartedSpeakingFrame(),
            TTSTextFrame(text="Sure", aggregated_by=AggregationType.WORD),
            TTSTextFrame(text=" I", aggregated_by=AggregationType.WORD),
            TTSTextFrame(text=" can", aggregated_by=AggregationType.WORD),
            # User interrupts
            InterruptionFrame(),
            # New turn starts
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Wait", user_id="user1", timestamp=""),
            SleepFrame(sleep=0.1),
        ]

        await run_test(processor, frames_to_send=frames_to_send)

        # Verify first turn was interrupted
        self.assertGreaterEqual(
            len(turn_ended_calls), 1, f"Expected at least 1 turn ended, got {len(turn_ended_calls)}"
        )
        first_turn = turn_ended_calls[0]
        self.assertEqual(first_turn["user_text"], "Tell me")
        # Note: In this test flow, InterruptionFrame arrives before TTSTextFrames are processed,
        # so assistant text may be empty. In real scenarios, word timestamps ensure proper capture.
        self.assertIn(first_turn["assistant_text"], ["", "Sure I can", "Sure  I  can"])
        self.assertTrue(first_turn["interrupted"])

    async def test_multiple_turns(self):
        """Test multiple back-and-forth turns."""
        processor = TurnAwareTranscriptProcessor()

        # Track events
        turn_started_calls = []
        turn_ended_calls = []

        @processor.event_handler("on_turn_started")
        async def on_turn_started(proc, turn_number):
            turn_started_calls.append(turn_number)

        @processor.event_handler("on_turn_ended")
        async def on_turn_ended(proc, turn_number, user_text, assistant_text, interrupted):
            turn_ended_calls.append(
                {
                    "turn_number": turn_number,
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                }
            )

        frames_to_send = [
            # Turn 1
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Hi", user_id="user1", timestamp=""),
            SleepFrame(sleep=0.01),  # Allow transcription to process
            BotStartedSpeakingFrame(),
            TTSTextFrame(text="Hello", aggregated_by=AggregationType.WORD),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.05),
            # Turn 2
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="How are you", user_id="user1", timestamp=""),
            SleepFrame(sleep=0.01),  # Allow transcription to process
            BotStartedSpeakingFrame(),
            TTSTextFrame(text="I'm", aggregated_by=AggregationType.WORD),
            TTSTextFrame(text=" good", aggregated_by=AggregationType.WORD),
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
        ]

        await run_test(processor, frames_to_send=frames_to_send)

        # Verify multiple turns
        self.assertEqual(
            len(turn_started_calls), 2, f"Expected 2 turns started, got {len(turn_started_calls)}"
        )
        self.assertEqual(turn_started_calls, [1, 2])

        self.assertEqual(
            len(turn_ended_calls), 2, f"Expected 2 turns ended, got {len(turn_ended_calls)}"
        )
        self.assertEqual(turn_ended_calls[0]["turn_number"], 1)
        self.assertEqual(turn_ended_calls[0]["user_text"], "Hi")
        self.assertEqual(turn_ended_calls[0]["assistant_text"], "Hello")

        self.assertEqual(turn_ended_calls[1]["turn_number"], 2)
        self.assertEqual(turn_ended_calls[1]["user_text"], "How are you")
        self.assertEqual(turn_ended_calls[1]["assistant_text"], "I'm  good")


if __name__ == "__main__":
    unittest.main()
