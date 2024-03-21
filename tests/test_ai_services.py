import unittest

from typing import AsyncGenerator

from dailyai.services.ai_services import AIService
from dailyai.pipeline.frames import EndFrame, Frame, TextFrame


class SimpleAIService(AIService):
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        yield frame


class TestBaseAIService(unittest.IsolatedAsyncioTestCase):
    async def test_simple_processing(self):
        service = SimpleAIService()

        input_frames = [
            TextFrame("hello"),
            EndFrame()
        ]

        output_frames = []
        for input_frame in input_frames:
            async for output_frame in service.process_frame(input_frame):
                output_frames.append(output_frame)

        self.assertEqual(input_frames, output_frames)


if __name__ == "__main__":
    unittest.main()
