import asyncio
import doctest
import functools
import unittest

from dailyai.pipeline.aggregators import (
    GatedAggregator,
    ParallelPipeline,
    SentenceAggregator,
    StatelessTextTransformer,
)
from dailyai.pipeline.frames import (
    AudioFrame,
    EndFrame,
    ImageFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    Frame,
    TextFrame,
)

from dailyai.pipeline.pipeline import Pipeline


class TestDailyFrameAggregators(unittest.IsolatedAsyncioTestCase):
    async def test_sentence_aggregator(self):
        sentence = "Hello, world. How are you? I am fine"
        expected_sentences = ["Hello, world.", " How are you?", " I am fine "]
        aggregator = SentenceAggregator()
        for word in sentence.split(" "):
            async for sentence in aggregator.process_frame(TextFrame(word + " ")):
                self.assertIsInstance(sentence, TextFrame)
                if isinstance(sentence, TextFrame):
                    self.assertEqual(sentence.text, expected_sentences.pop(0))

        async for sentence in aggregator.process_frame(EndFrame()):
            if len(expected_sentences):
                self.assertIsInstance(sentence, TextFrame)
                if isinstance(sentence, TextFrame):
                    self.assertEqual(sentence.text, expected_sentences.pop(0))
            else:
                self.assertIsInstance(sentence, EndFrame)

        self.assertEqual(expected_sentences, [])

    async def test_gated_accumulator(self):
        gated_aggregator = GatedAggregator(
            gate_open_fn=lambda frame: isinstance(frame, ImageFrame),
            gate_close_fn=lambda frame: isinstance(frame, LLMResponseStartFrame),
            start_open=False,
        )

        frames = [
            LLMResponseStartFrame(),
            TextFrame("Hello, "),
            TextFrame("world."),
            AudioFrame(b"hello"),
            ImageFrame("image", b"image"),
            AudioFrame(b"world"),
            LLMResponseEndFrame(),
        ]

        expected_output_frames = [
            ImageFrame("image", b"image"),
            LLMResponseStartFrame(),
            TextFrame("Hello, "),
            TextFrame("world."),
            AudioFrame(b"hello"),
            AudioFrame(b"world"),
            LLMResponseEndFrame(),
        ]
        for frame in frames:
            async for out_frame in gated_aggregator.process_frame(frame):
                self.assertEqual(out_frame, expected_output_frames.pop(0))
        self.assertEqual(expected_output_frames, [])

    async def test_parallel_pipeline(self):

        async def slow_add(sleep_time:float, name:str, x: str):
            await asyncio.sleep(sleep_time)
            return ":".join([x, name])

        pipe1_annotation = StatelessTextTransformer(functools.partial(slow_add, 0.1, 'pipe1'))
        pipe2_annotation = StatelessTextTransformer(functools.partial(slow_add, 0.2, 'pipe2'))
        sentence_aggregator = SentenceAggregator()
        add_dots = StatelessTextTransformer(lambda x: x + ".")

        source = asyncio.Queue()
        sink = asyncio.Queue()
        pipeline = Pipeline(
            [
                ParallelPipeline(
                    [[pipe1_annotation], [sentence_aggregator, pipe2_annotation]]
                ),
                add_dots,
            ],
            source,
            sink,
        )

        frames = [
            TextFrame("Hello, "),
            TextFrame("world."),
            EndFrame()
        ]

        expected_output_frames: list[Frame] = [
            TextFrame(text='Hello, :pipe1.'),
            TextFrame(text='world.:pipe1.'),
            TextFrame(text='Hello, world.:pipe2.'),
            EndFrame()
        ]

        for frame in frames:
            await source.put(frame)

        await pipeline.run_pipeline()

        while not sink.empty():
            frame = await sink.get()
            self.assertEqual(frame, expected_output_frames.pop(0))


def load_tests(loader, tests, ignore):
    """ Run doctests on the aggregators module. """
    from dailyai.pipeline import aggregators
    tests.addTests(doctest.DocTestSuite(aggregators))
    return tests
