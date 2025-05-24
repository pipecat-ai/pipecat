#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EndFrame,
    ImageRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.processors.aggregators.gated import GatedAggregator
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.tests.utils import SleepFrame, run_test


class TestSentenceAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_sentence_aggregator(self):
        aggregator = SentenceAggregator()

        sentence = "Hello, world. How are you? I am fine!"

        frames_to_send = []
        for word in sentence.split(" "):
            frames_to_send.append(TextFrame(text=word + " "))

        expected_down_frames = [TextFrame, TextFrame, TextFrame]

        (received_down, _) = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        assert received_down[-3].text == "Hello, world. "
        assert received_down[-2].text == "How are you? "
        assert received_down[-1].text == "I am fine! "


class TestGatedAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_gated_aggregator(self):
        gated_aggregator = GatedAggregator(
            gate_open_fn=lambda frame: isinstance(frame, ImageRawFrame),
            gate_close_fn=lambda frame: isinstance(frame, LLMFullResponseStartFrame),
            start_open=False,
        )

        frames_to_send = [
            LLMFullResponseStartFrame(),
            TextFrame("Hello, "),
            TextFrame("world."),
            OutputAudioRawFrame(audio=b"hello", sample_rate=16000, num_channels=1),
            OutputImageRawFrame(image=b"image", size=(0, 0), format="RGB"),
            OutputAudioRawFrame(audio=b"world", sample_rate=16000, num_channels=1),
            LLMFullResponseEndFrame(),
        ]

        expected_down_frames = [
            OutputImageRawFrame,
            LLMFullResponseStartFrame,
            TextFrame,
            TextFrame,
            OutputAudioRawFrame,
            OutputAudioRawFrame,
            LLMFullResponseEndFrame,
        ]

        (received_down, _) = await run_test(
            gated_aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )


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

        # The TranscriptionFrame should be the last frame
        transcription_frame = received_down_frames[-1]
        self.assertEqual(transcription_frame.text, "DTMF: 123#")

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
        aggregator = DTMFAggregator(timeout=0.1)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            InputDTMFFrame(button=KeypadEntry.POUND),  # First sequence
            InputDTMFFrame(button=KeypadEntry.FOUR),
            InputDTMFFrame(button=KeypadEntry.FIVE),
            SleepFrame(sleep=0.2),  # Second sequence via timeout
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

    async def test_interruption_frame_flush(self):
        """Test that StartInterruptionFrame flushes pending aggregation."""
        aggregator = DTMFAggregator(timeout=1.0)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            SleepFrame(sleep=0.1),  # Allow time for aggregation
            StartInterruptionFrame(),
        ]
        expected_down_frames = [
            InputDTMFFrame,
            InputDTMFFrame,
            TranscriptionFrame,  # Should flush before interruption
            StartInterruptionFrame,
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
        expected_down_frames = [InputDTMFFrame] * 12 + [TranscriptionFrame]

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
