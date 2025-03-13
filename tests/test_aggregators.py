#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.frames.frames import (
    ImageRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    TextFrame,
)
from pipecat.processors.aggregators.gated import GatedAggregator
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.tests.utils import run_test


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
