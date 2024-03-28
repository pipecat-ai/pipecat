import asyncio
import unittest
from unittest.mock import Mock

from dailyai.pipeline.aggregators import SentenceAggregator, StatelessTextTransformer
from dailyai.pipeline.frame_processor import FrameProcessor
from dailyai.pipeline.frames import EndFrame, TextFrame

from dailyai.pipeline.pipeline import Pipeline


class TestDailyPipeline(unittest.IsolatedAsyncioTestCase):

    async def test_pipeline_simple(self):
        aggregator = SentenceAggregator()

        outgoing_queue = asyncio.Queue()
        incoming_queue = asyncio.Queue()
        pipeline = Pipeline([aggregator], incoming_queue, outgoing_queue)

        await incoming_queue.put(TextFrame("Hello, "))
        await incoming_queue.put(TextFrame("world."))
        await incoming_queue.put(EndFrame())

        await pipeline.run_pipeline()

        self.assertEqual(await outgoing_queue.get(), TextFrame("Hello, world."))
        self.assertIsInstance(await outgoing_queue.get(), EndFrame)

    async def test_pipeline_multiple_stages(self):
        sentence_aggregator = SentenceAggregator()
        to_upper = StatelessTextTransformer(lambda x: x.upper())
        add_space = StatelessTextTransformer(lambda x: x + " ")

        outgoing_queue = asyncio.Queue()
        incoming_queue = asyncio.Queue()
        pipeline = Pipeline(
            [add_space, sentence_aggregator, to_upper],
            incoming_queue,
            outgoing_queue
        )

        sentence = "Hello, world. It's me, a pipeline."
        for c in sentence:
            await incoming_queue.put(TextFrame(c))
        await incoming_queue.put(EndFrame())

        await pipeline.run_pipeline()

        self.assertEqual(
            await outgoing_queue.get(), TextFrame("H E L L O ,   W O R L D .")
        )
        self.assertEqual(
            await outgoing_queue.get(),
            TextFrame("   I T ' S   M E ,   A   P I P E L I N E ."),
        )
        # leftover little bit because of the spacing
        self.assertEqual(
            await outgoing_queue.get(),
            TextFrame(" "),
        )
        self.assertIsInstance(await outgoing_queue.get(), EndFrame)


class TestLogFrame(unittest.TestCase):
    class MockProcessor(FrameProcessor):
        def __init__(self, name):
            self.name = name

        def __str__(self):
            return self.name

    def setUp(self):
        self.processor1 = self.MockProcessor('processor1')
        self.processor2 = self.MockProcessor('processor2')
        self.pipeline = Pipeline(
            processors=[self.processor1, self.processor2])
        self.pipeline._name = 'MyClass'
        self.pipeline._logger = Mock()

    def test_log_frame_from_source(self):
        frame = Mock(__class__=Mock(__name__='MyFrame'))
        self.pipeline._log_frame(frame, depth=1)
        self.pipeline._logger.debug.assert_called_once_with(
            'MyClass  source -> MyFrame -> processor1')

    def test_log_frame_to_sink(self):
        frame = Mock(__class__=Mock(__name__='MyFrame'))
        self.pipeline._log_frame(frame, depth=3)
        self.pipeline._logger.debug.assert_called_once_with(
            'MyClass      processor2 -> MyFrame -> sink')

    def test_log_frame_repeated_log(self):
        frame = Mock(__class__=Mock(__name__='MyFrame'))
        self.pipeline._log_frame(frame, depth=2)
        self.pipeline._logger.debug.assert_called_once_with(
            'MyClass    processor1 -> MyFrame -> processor2')
        self.pipeline._log_frame(frame, depth=2)
        self.pipeline._logger.debug.assert_called_with('MyClass    ... repeated')

    def test_log_frame_reset_repeated_log(self):
        frame1 = Mock(__class__=Mock(__name__='MyFrame1'))
        frame2 = Mock(__class__=Mock(__name__='MyFrame2'))
        self.pipeline._log_frame(frame1, depth=2)
        self.pipeline._logger.debug.assert_called_once_with(
            'MyClass    processor1 -> MyFrame1 -> processor2')
        self.pipeline._log_frame(frame1, depth=2)
        self.pipeline._logger.debug.assert_called_with('MyClass    ... repeated')
        self.pipeline._log_frame(frame2, depth=2)
        self.pipeline._logger.debug.assert_called_with(
            'MyClass    processor1 -> MyFrame2 -> processor2')
