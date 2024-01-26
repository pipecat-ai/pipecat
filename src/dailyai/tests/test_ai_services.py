from re import A
import unittest

from typing import AsyncGenerator, Generator

from dailyai.services.ai_services import AIService
from dailyai.queue_frame import EndStreamQueueFrame, QueueFrame, TextQueueFrame


class SimpleAIService(AIService):
    async def process_frame(self, frame: QueueFrame) -> AsyncGenerator[QueueFrame, None]:
        yield frame


class TestBaseAIService(unittest.IsolatedAsyncioTestCase):
    async def test_async_input(self):
        service = SimpleAIService()

        input_frames = [
            TextQueueFrame("hello"),
            EndStreamQueueFrame()
        ]

        async def iterate_frames() -> AsyncGenerator[QueueFrame, None]:
            for frame in input_frames:
                yield frame

        output_frames = []
        async for frame in service.run(iterate_frames()):
            output_frames.append(frame)

        self.assertEqual(input_frames, output_frames)

    async def test_nonasync_input(self):
        service = SimpleAIService()

        input_frames = [TextQueueFrame("hello"), EndStreamQueueFrame()]

        def iterate_frames() -> Generator[QueueFrame, None, None]:
            for frame in input_frames:
                yield frame

        output_frames = []
        async for frame in service.run(iterate_frames()):
            output_frames.append(frame)

        self.assertEqual(input_frames, output_frames)


if __name__ == "__main__":
    unittest.main()
