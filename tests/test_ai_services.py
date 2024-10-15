import unittest

from typing import AsyncGenerator

from pipecat.services.ai_services import AIService, match_endofsentence
from pipecat.frames.frames import EndFrame, Frame, TextFrame


class SimpleAIService(AIService):
    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        yield frame


class TestBaseAIService(unittest.IsolatedAsyncioTestCase):
    async def test_simple_processing(self):
        service = SimpleAIService()

        input_frames = [TextFrame("hello"), EndFrame()]

        output_frames = []
        for input_frame in input_frames:
            async for output_frame in service.process_frame(input_frame):
                output_frames.append(output_frame)

        self.assertEqual(input_frames, output_frames)

    async def test_endofsentence(self):
        assert match_endofsentence("This is a sentence.")
        assert match_endofsentence("This is a sentence! ")
        assert match_endofsentence("This is a sentence?")
        assert match_endofsentence("This is a sentence:")
        assert match_endofsentence("This is a sentence;")
        assert not match_endofsentence("This is not a sentence")
        assert not match_endofsentence("This is not a sentence,")
        assert not match_endofsentence("This is not a sentence, ")
        assert not match_endofsentence("Ok, Mr. Smith let's ")
        assert not match_endofsentence("Dr. Walker, I presume ")
        assert not match_endofsentence("Prof. Walker, I presume ")
        assert not match_endofsentence("zweitens, und 3.")
        assert not match_endofsentence("Heute ist Dienstag, der 3.")  # 3. Juli 2024
        assert not match_endofsentence("America, or the U.")  # U.S.A.
        assert not match_endofsentence("It still early, it's 3:00 a.")  # 3:00 a.m.

    async def test_endofsentence_zh(self):
        chinese_sentences = [
            "你好。",
            "你好！",
            "吃了吗？",
            "安全第一；",
            "他说：",
        ]
        for i in chinese_sentences:
            assert match_endofsentence(i)
        assert not match_endofsentence("你好，")


if __name__ == "__main__":
    unittest.main()
