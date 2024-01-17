from re import A
import unittest

from typing import AsyncGenerator, Generator

from dailyai.services.ai_services import AIService, SentenceAggregator
from dailyai.queue_frame import QueueFrame, FrameType

class SimpleAIService(AIService):
    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.TEXT_CHUNK])

    def possible_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.TEXT_CHUNK])

    async def process_frame(self, requested_frame_types: set[FrameType], frame: QueueFrame) -> QueueFrame | None:
        return frame

class TestBaseAIService(unittest.IsolatedAsyncioTestCase):
    async def test_async_input(self):
        service = SimpleAIService()

        input_frames = [
            QueueFrame(FrameType.TEXT_CHUNK, "hello"),
            QueueFrame(FrameType.END_STREAM, None),
        ]
        async def iterate_frames() -> AsyncGenerator[QueueFrame, None]:
            for frame in input_frames:
                yield frame

        output_frames = []
        async for frame in service.run(set([FrameType.TEXT_CHUNK]), iterate_frames()):
            output_frames.append(frame)

        self.assertEqual(input_frames, output_frames)

    async def test_nonasync_input(self):
        service = SimpleAIService()

        input_frames = [
            QueueFrame(FrameType.TEXT_CHUNK, "hello"),
            QueueFrame(FrameType.END_STREAM, None),
        ]

        def iterate_frames() -> Generator[QueueFrame, None, None]:
            for frame in input_frames:
                yield frame

        output_frames = []
        async for frame in service.run(set([FrameType.TEXT_CHUNK]), iterate_frames()):
            output_frames.append(frame)

        self.assertEqual(input_frames, output_frames)


class TestSentenceAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_clause(self) -> None:
        input_frames = [
            QueueFrame(FrameType.TEXT_CHUNK, "hello"),
            QueueFrame(FrameType.END_STREAM, None),
        ]

        service = SentenceAggregator()
        output_frames = []
        async for frame in service.run(set([FrameType.SENTENCE]), input_frames):
            output_frames.append(frame)

        self.assertEqual(1, len(output_frames))
        self.assertEqual(QueueFrame(FrameType.SENTENCE, "hello"), output_frames[0])

    async def test_sentence(self) -> None:
        input_frames = [
            QueueFrame(FrameType.TEXT_CHUNK, "hello, "),
            QueueFrame(FrameType.TEXT_CHUNK, "world."),
            QueueFrame(FrameType.END_STREAM, None),
        ]

        service = SentenceAggregator()
        output_frames = []
        async for frame in service.run(set([FrameType.SENTENCE]), input_frames):
            output_frames.append(frame)

        self.assertEqual(1, len(output_frames))
        self.assertEqual(QueueFrame(FrameType.SENTENCE, "hello, world."), output_frames[0])

    async def test_sentence_and_clause(self) -> None:
        input_frames = [
            QueueFrame(FrameType.TEXT_CHUNK, "hello, "),
            QueueFrame(FrameType.TEXT_CHUNK, "world."),
            QueueFrame(FrameType.TEXT_CHUNK, " How are"),
            QueueFrame(FrameType.END_STREAM, None),
        ]

        service = SentenceAggregator()
        output_frames = []
        async for frame in service.run(set([FrameType.SENTENCE]), input_frames):
            output_frames.append(frame)

        self.assertEqual(2, len(output_frames))
        self.assertEqual(
            QueueFrame(FrameType.SENTENCE, "hello, world."), output_frames[0]
        )
        self.assertEqual(
            QueueFrame(FrameType.SENTENCE, " How are"), output_frames[1]
        )

    async def test_two_sentences(self) -> None:
        input_frames = [
            QueueFrame(FrameType.TEXT_CHUNK, "hello, "),
            QueueFrame(FrameType.TEXT_CHUNK, "world."),
            QueueFrame(FrameType.TEXT_CHUNK, " How are"),
            QueueFrame(FrameType.TEXT_CHUNK, " you doing?"),
            QueueFrame(FrameType.END_STREAM, None),
        ]

        service = SentenceAggregator()
        output_frames = []
        async for frame in service.run(set([FrameType.SENTENCE]), input_frames):
            output_frames.append(frame)

        self.assertEqual(2, len(output_frames))
        self.assertEqual(
            QueueFrame(FrameType.SENTENCE, "hello, world."), output_frames[0]
        )
        self.assertEqual(QueueFrame(FrameType.SENTENCE, " How are you doing?"), output_frames[1])


if __name__ == "__main__":
    unittest.main()
