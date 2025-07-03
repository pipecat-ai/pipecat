#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EndFrame,
    InputDTMFFrame,
    KeypadEntry,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.tests.utils import SleepFrame, run_test


class TestDTMFAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_basic_aggregation_with_pound(self):
        """Test basic DTMF aggregation ending with pound key."""
        aggregator = DTMFAggregator()
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            InputDTMFFrame(button=KeypadEntry.THREE),
            InputDTMFFrame(button=KeypadEntry.POUND),
        ]
        expected_down_frames = [
            InputDTMFFrame,
            InputDTMFFrame,
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,
        ]

        received_down_frames, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Find and verify the TranscriptionFrame
        transcription_frames = [
            f for f in received_down_frames if isinstance(f, TranscriptionFrame)
        ]
        self.assertEqual(len(transcription_frames), 1)
        self.assertEqual(transcription_frames[0].text, "DTMF: 123#")

    async def test_timeout_aggregation(self):
        """Test DTMF aggregation with timeout flush."""
        aggregator = DTMFAggregator(timeout=0.1)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            SleepFrame(sleep=0.2),  # This should trigger timeout
            InputDTMFFrame(button=KeypadEntry.THREE),
            SleepFrame(sleep=0.2),  # This should trigger another timeout
        ]
        expected_down_frames = [
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,  # First aggregation "12"
            InputDTMFFrame,
            TranscriptionFrame,  # Second aggregation "3"
        ]

        received_down_frames, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        # Find the TranscriptionFrames
        transcription_frames = [
            f for f in received_down_frames if isinstance(f, TranscriptionFrame)
        ]
        self.assertEqual(len(transcription_frames), 2)
        self.assertEqual(transcription_frames[0].text, "DTMF: 12")
        self.assertEqual(transcription_frames[1].text, "DTMF: 3")

    async def test_multiple_aggregations(self):
        """Test multiple DTMF sequences with pound termination."""
        aggregator = DTMFAggregator(timeout=0.2)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            InputDTMFFrame(button=KeypadEntry.POUND),  # First sequence
            SleepFrame(sleep=0.1),
            InputDTMFFrame(button=KeypadEntry.FOUR),
            InputDTMFFrame(button=KeypadEntry.FIVE),
            SleepFrame(sleep=0.3),  # Second sequence via timeout
        ]
        expected_down_frames = [
            InputDTMFFrame,
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,  # "12#"
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,  # "45"
        ]

        received_down_frames, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        transcription_frames = [
            f for f in received_down_frames if isinstance(f, TranscriptionFrame)
        ]
        self.assertEqual(len(transcription_frames), 2)
        self.assertEqual(transcription_frames[0].text, "DTMF: 12#")
        self.assertEqual(transcription_frames[1].text, "DTMF: 45")

    async def test_end_frame_flush(self):
        """Test that EndFrame flushes pending aggregation."""
        aggregator = DTMFAggregator(timeout=1.0)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            SleepFrame(sleep=0.1),  # Allow time for aggregation
            EndFrame(),
        ]
        expected_down_frames = [
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,  # Should flush before EndFrame
            EndFrame,
        ]

        received_down_frames, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            send_end_frame=False,  # We're sending one in the test to test EndFrame logic
        )

        transcription_frames = [
            f for f in received_down_frames if isinstance(f, TranscriptionFrame)
        ]
        self.assertEqual(len(transcription_frames), 1)
        self.assertEqual(transcription_frames[0].text, "DTMF: 12")

    async def test_custom_prefix(self):
        """Test custom prefix configuration."""
        aggregator = DTMFAggregator(prefix="Menu: ")
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.POUND),
        ]
        expected_down_frames = [
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,
        ]

        received_down_frames, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        transcription_frames = [
            f for f in received_down_frames if isinstance(f, TranscriptionFrame)
        ]
        self.assertEqual(len(transcription_frames), 1)
        self.assertEqual(transcription_frames[0].text, "Menu: 1#")

    async def test_custom_termination_digit(self):
        """Test custom termination digit configuration."""
        aggregator = DTMFAggregator(termination_digit=KeypadEntry.STAR)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            InputDTMFFrame(button=KeypadEntry.STAR),  # Custom terminator
        ]
        expected_down_frames = [
            InputDTMFFrame,
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,
        ]

        received_down_frames, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        transcription_frames = [
            f for f in received_down_frames if isinstance(f, TranscriptionFrame)
        ]
        self.assertEqual(len(transcription_frames), 1)
        self.assertEqual(transcription_frames[0].text, "DTMF: 12*")

    async def test_all_keypad_entries(self):
        """Test all possible keypad entries."""
        aggregator = DTMFAggregator()
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ZERO),
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            InputDTMFFrame(button=KeypadEntry.THREE),
            InputDTMFFrame(button=KeypadEntry.FOUR),
            InputDTMFFrame(button=KeypadEntry.FIVE),
            InputDTMFFrame(button=KeypadEntry.SIX),
            InputDTMFFrame(button=KeypadEntry.SEVEN),
            InputDTMFFrame(button=KeypadEntry.EIGHT),
            InputDTMFFrame(button=KeypadEntry.NINE),
            InputDTMFFrame(button=KeypadEntry.STAR),
            InputDTMFFrame(button=KeypadEntry.POUND),
        ]

        # All the InputDTMFFrames plus one TranscriptionFrame
        expected_down_frames = [InputDTMFFrame] * len(frames_to_send) + [TranscriptionFrame]

        received_down_frames, _ = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )

        transcription_frames = [
            f for f in received_down_frames if isinstance(f, TranscriptionFrame)
        ]
        self.assertEqual(len(transcription_frames), 1)
        self.assertEqual(transcription_frames[0].text, "DTMF: 0123456789*#")
