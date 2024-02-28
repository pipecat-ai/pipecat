import asyncio
from math import pi
import unittest

from unittest.mock import MagicMock, patch

from dailyai.queue_frame import AudioQueueFrame, EndStreamQueueFrame, ImageQueueFrame, TextQueueFrame
from dailyai.services.ai_services import PipeService

class TestDailyTransport(unittest.IsolatedAsyncioTestCase):
    class IncomingPipeService(PipeService):
        def __init__(self):
            super().__init__()
            self.sink_queue = asyncio.Queue()

    async def test_pipe_chain(self):
        pipe1 = TestDailyTransport.IncomingPipeService()
        pipe2 = PipeService(pipe1)
        pipe3 = PipeService(pipe2)

        await pipe1.sink_queue.put(TextQueueFrame("test"))
        await pipe1.sink_queue.put(EndStreamQueueFrame())

        await asyncio.gather(pipe1.process_queue(), pipe2.process_queue(), pipe3.process_queue())

        self.assertEqual(pipe3.sink_queue.qsize(), 2)
        frame = await pipe3.sink_queue.get()
        self.assertIsInstance(frame, TextQueueFrame)
        if isinstance(frame, TextQueueFrame):
            self.assertEqual(frame.text, "test")

        frame = await pipe3.sink_queue.get()
        self.assertIsInstance(frame, EndStreamQueueFrame)
