import asyncio
import unittest

from dailyai.queue_frame import EndStreamQueueFrame, TextQueueFrame
from dailyai.services.ai_services import PipeService
from dailyai.queue_aggregators import QueueFrameAggregator, QueueMergeGateOnFirst, QueueTee

class IncomingPipeService(PipeService):
    def __init__(self):
        super().__init__()
        self.out_queue = asyncio.Queue()

class QueueTeeTest(unittest.IsolatedAsyncioTestCase):

    async def test_queue_tee(self):
        originpipe = IncomingPipeService()
        inpipe1 = PipeService(originpipe.out_queue)
        outpipe1 = PipeService()
        outpipe2 = PipeService()
        teepipe = QueueTee(source_queue=inpipe1.out_queue, out_services=[outpipe1, outpipe2])

        originpipe.out_queue.put_nowait(TextQueueFrame("test"))
        originpipe.out_queue.put_nowait(EndStreamQueueFrame())

        await asyncio.gather(*[pipe.process_queue() for pipe in [originpipe, inpipe1, outpipe1, outpipe2, teepipe]])

        def validateOutputPipe(pipe: PipeService):
            self.assertEqual(pipe.out_queue.qsize(), 2)
            frame = pipe.out_queue.get_nowait()
            self.assertIsInstance(frame, TextQueueFrame)
            if isinstance(frame, TextQueueFrame):
                self.assertEqual(frame.text, "test")
            self.assertIsInstance(pipe.out_queue.get_nowait(), EndStreamQueueFrame)

        validateOutputPipe(outpipe1)
        validateOutputPipe(outpipe2)

class QueueFrameAggregatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_queue_frame_aggregator(self):
        def aggregate_sentences(accumulation, frame):
            if not accumulation:
                accumulation = ""
            if isinstance(frame, TextQueueFrame):
                accumulation += frame.text
            if accumulation.endswith((".", "!", "?")):
                return ("", TextQueueFrame(accumulation))
            return (accumulation, None)

        def finalize_sentences(accumulation):
            return TextQueueFrame(accumulation)

        originpipe = IncomingPipeService()
        aggregator_pipe = QueueFrameAggregator(
            source_queue=originpipe.out_queue,
            aggregator=aggregate_sentences,
            finalizer=finalize_sentences,
        )

        originpipe.out_queue.put_nowait(TextQueueFrame("testing, "))
        originpipe.out_queue.put_nowait(TextQueueFrame("one."))
        originpipe.out_queue.put_nowait(TextQueueFrame("two."))
        originpipe.out_queue.put_nowait(TextQueueFrame("three."))
        originpipe.out_queue.put_nowait(TextQueueFrame("can you "))
        originpipe.out_queue.put_nowait(TextQueueFrame("hear me"))
        originpipe.out_queue.put_nowait(EndStreamQueueFrame())

        await asyncio.gather(originpipe.process_queue(), aggregator_pipe.process_queue())

        self.assertEqual(aggregator_pipe.out_queue.qsize(), 5)
        expected_text = ["testing, one.", "two.", "three.", "can you hear me"]
        for exepectation in expected_text:
            frame = aggregator_pipe.out_queue.get_nowait()
            print(frame)
            self.assertIsInstance(frame, TextQueueFrame)
            if isinstance(frame, TextQueueFrame):
                self.assertEqual(frame.text, exepectation)

        self.assertIsInstance(aggregator_pipe.out_queue.get_nowait(), EndStreamQueueFrame)

class QueueMergeGateOnFirstTest(unittest.IsolatedAsyncioTestCase):
    async def test_queue_merge_gate_on_first(self):
        pipe1 = IncomingPipeService()
        pipe2 = IncomingPipeService()

        merge_pipe = QueueMergeGateOnFirst(
            source_queues=[pipe1.out_queue, pipe2.out_queue],
        )

        evt = asyncio.Event()

        async def add_items_to_first_pipe():
            await evt.wait()
            await pipe1.out_queue.put(TextQueueFrame("pipe1.1"))
            await pipe1.out_queue.put(TextQueueFrame("pipe1.2"))
            await pipe1.out_queue.put(EndStreamQueueFrame())

        async def add_items_to_second_pipe():
            await pipe2.out_queue.put(TextQueueFrame("pipe2.1"))
            evt.set()
            await pipe2.out_queue.put(EndStreamQueueFrame())

        await asyncio.gather(
            *[pipe.process_queue() for pipe in [pipe1, pipe2, merge_pipe]],
            add_items_to_first_pipe(),
            add_items_to_second_pipe())

        self.assertEqual(merge_pipe.out_queue.qsize(), 4)
        frame = merge_pipe.out_queue.get_nowait()
        assert isinstance(frame, TextQueueFrame)
        if isinstance(frame, TextQueueFrame):
            self.assertEqual(frame.text, "pipe1.1")

        frame = merge_pipe.out_queue.get_nowait()
        assert isinstance(frame, TextQueueFrame)
        if isinstance(frame, TextQueueFrame):
            self.assertEqual(frame.text, "pipe2.1")

        frame = merge_pipe.out_queue.get_nowait()
        assert isinstance(frame, TextQueueFrame)
        if isinstance(frame, TextQueueFrame):
            self.assertEqual(frame.text, "pipe1.2")

        frame = merge_pipe.out_queue.get_nowait()
        assert isinstance(frame, EndStreamQueueFrame)
