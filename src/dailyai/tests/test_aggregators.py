import asyncio
from doctest import OutputChecker
from typing import Text
import unittest

import llm
from dailyai.pipeline.aggregators import GatedAccumulator, SentenceAggregator, StatelessTextTransformer
from dailyai.pipeline.frames import AudioQueueFrame, EndStreamQueueFrame, ImageQueueFrame, LLMResponseEndQueueFrame, LLMResponseStartQueueFrame, TextQueueFrame

from dailyai.pipeline.pipeline import Pipeline


class TestDailyFrameAggregators(unittest.IsolatedAsyncioTestCase):
    async def test_sentence_aggregator(self):
        sentence = "Hello, world. How are you? I am fine"
        expected_sentences = ["Hello, world.", " How are you?", " I am fine "]
        aggregator = SentenceAggregator()
        for word in sentence.split(" "):
            async for sentence in aggregator.process_frame(TextQueueFrame(word + " ")):
                self.assertIsInstance(sentence, TextQueueFrame)
                if isinstance(sentence, TextQueueFrame):
                    self.assertEqual(sentence.text, expected_sentences.pop(0))

        async for sentence in aggregator.process_frame(EndStreamQueueFrame()):
            if len(expected_sentences):
                self.assertIsInstance(sentence, TextQueueFrame)
                if isinstance(sentence, TextQueueFrame):
                    self.assertEqual(sentence.text, expected_sentences.pop(0))
            else:
                self.assertIsInstance(sentence, EndStreamQueueFrame)

        self.assertEqual(expected_sentences, [])

    async def test_gated_accumulator(self):
        gated_accumulator = GatedAccumulator(
            gate_open_fn=lambda frame: isinstance(frame, ImageQueueFrame),
            gate_close_fn=lambda frame: isinstance(frame, LLMResponseStartQueueFrame),
            start_open=False,
        )

        frames = [
            LLMResponseStartQueueFrame(),
            TextQueueFrame("Hello, "),
            TextQueueFrame("world."),
            AudioQueueFrame(b"hello"),
            ImageQueueFrame("image", b"image"),
            AudioQueueFrame(b"world"),
            LLMResponseEndQueueFrame(),
        ]

        expected_output_frames = [
            ImageQueueFrame("image", b"image"),
            LLMResponseStartQueueFrame(),
            TextQueueFrame("Hello, "),
            TextQueueFrame("world."),
            AudioQueueFrame(b"hello"),
            AudioQueueFrame(b"world"),
            LLMResponseEndQueueFrame(),
        ]
        for frame in frames:
            async for out_frame in gated_accumulator.process_frame(frame):
                self.assertEqual(out_frame, expected_output_frames.pop(0))
        self.assertEqual(expected_output_frames, [])

    async def test_parallel_pipeline(self):
        pass
