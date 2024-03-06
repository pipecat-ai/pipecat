import unittest

from typing import AsyncGenerator, Generator

from dailyai.services.ai_services import AIService
from dailyai.pipeline.frames import EndFrame, Frame, TextFrame


class SimpleAIService(AIService):
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        yield frame


class TestBaseAIService(unittest.IsolatedAsyncioTestCase):
    async def test_async_input(self):
        service = SimpleAIService()

        input_frames = [
            TextFrame("hello"),
            EndFrame()
        ]

        async def iterate_frames() -> AsyncGenerator[Frame, None]:
            for frame in input_frames:
                yield frame

        output_frames = []
        async for frame in service.run(iterate_frames()):
            output_frames.append(frame)

        self.assertEqual(input_frames, output_frames)

    async def test_nonasync_input(self):
        service = SimpleAIService()

        input_frames = [TextFrame("hello"), EndFrame()]

        def iterate_frames() -> Generator[Frame, None, None]:
            for frame in input_frames:
                yield frame

        output_frames = []
        async for frame in service.run(iterate_frames()):
            output_frames.append(frame)

        self.assertEqual(input_frames, output_frames)


if __name__ == "__main__":
    unittest.main()
