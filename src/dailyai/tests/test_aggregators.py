import asyncio
import functools
import unittest

from dailyai.pipeline.aggregators import (
    GatedAccumulator,
    ParallelPipeline,
    SentenceAggregator,
    StatelessTextTransformer,
)
from dailyai.pipeline.frames import (
    AudioQueueFrame,
    EndStreamQueueFrame,
    ImageQueueFrame,
    LLMResponseEndQueueFrame,
    LLMResponseStartQueueFrame,
    QueueFrame,
    TextQueueFrame,
)

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
            source,
            sink,
            [ParallelPipeline([[pipe1_annotation], [sentence_aggregator, pipe2_annotation]]), add_dots],
        )

        frames = [
            TextQueueFrame("Hello, "),
            TextQueueFrame("world."),
            EndStreamQueueFrame()
        ]

        expected_output_frames: list[QueueFrame] = [
            TextQueueFrame(text='Hello, :pipe1.'),
            TextQueueFrame(text='world.:pipe1.'),
            TextQueueFrame(text='Hello, world.:pipe2.'),
            EndStreamQueueFrame()
        ]

        for frame in frames:
            await source.put(frame)

        await pipeline.run_pipeline()

        while not sink.empty():
            frame = await sink.get()
            self.assertEqual(frame, expected_output_frames.pop(0))
