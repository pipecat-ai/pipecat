#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    ImageRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
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
    async def test_basic_aggregation(self):
        aggregator = DTMFAggregator()
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            InputDTMFFrame(button=KeypadEntry.THREE),
            InputDTMFFrame(button=KeypadEntry.POUND),
        ]
        expected_returned_frames = [TranscriptionFrame]
        received_down_frames, _ = received_down_frames, received_up_frames = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

        self.assertEqual(
            received_down_frames[0].text,
            "123#",
        )

    async def test_timeout_aggregation(self):
        aggregator = DTMFAggregator(timeout=0.1)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            SleepFrame(sleep=0.2),
            InputDTMFFrame(button=KeypadEntry.THREE),
        ]
        expected_returned_frames = [TranscriptionFrame, TranscriptionFrame]
        received_down_frames, received_up_frames = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

        self.assertEqual(
            received_down_frames[0].text,
            "12",
        )

        self.assertEqual(
            received_down_frames[1].text,
            "3",
        )

    async def test_multiple_aggregations(self):
        aggregator = DTMFAggregator(timeout=0.1)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            InputDTMFFrame(button=KeypadEntry.POUND),
            InputDTMFFrame(button=KeypadEntry.FOUR),
            InputDTMFFrame(button=KeypadEntry.FIVE),
        ]
        expected_returned_frames = [TranscriptionFrame, TranscriptionFrame]
        received_down_frames, received_up_frames = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

        self.assertEqual(
            received_down_frames[0].text,
            "12#",
        )

        self.assertEqual(
            received_down_frames[1].text,
            "45",
        )

    async def test_end_frame_flush(self):
        aggregator = DTMFAggregator(timeout=1.0)
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            InputDTMFFrame(button=KeypadEntry.TWO),
            EndFrame(),
        ]
        expected_returned_frames = [TranscriptionFrame]
        received_down_frames, received_up_frames = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

        self.assertEqual(
            received_down_frames[0].text,
            "12",
        )

    async def test_non_dtmf_frame_pass_through(self):
        aggregator = DTMFAggregator(timeout=0.1)
        test_frame = Frame()
        frames_to_send = [
            InputDTMFFrame(button=KeypadEntry.ONE),
            test_frame,
            InputDTMFFrame(button=KeypadEntry.POUND),
        ]
        expected_returned_frames = [Frame, TranscriptionFrame]
        received_down_frames, received_up_frames = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

        self.assertEqual(
            received_down_frames[1].text,
            "1#",
        )

    async def test_no_dtmf_input(self):
        aggregator = DTMFAggregator(timeout=0.1)
        frames_to_send = [Frame(), EndFrame()]
        expected_returned_frames = [Frame]
        received_down_frames, received_up_frames = await run_test(
            aggregator,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )

        self.assertEqual(
            len(received_down_frames),
            1,
        )
