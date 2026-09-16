#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A Gemma model's channel-switch tokens never leave an OpenAI-compatible service as text."""

import unittest

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.openai.llm import OpenAILLMService


class TestGemmaChannelTokens(unittest.IsolatedAsyncioTestCase):
    def _service(self, model: str, cls=OpenAILLMService):
        service = cls(settings=cls.Settings(model=model))
        self.spoken: list[str] = []

        async def push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, LLMTextFrame):
                self.spoken.append(frame.text)

        service.push_frame = push_frame
        return service

    async def _stream(self, service, *chunks: str) -> str:
        service._reset_channel_state()
        self.spoken.clear()
        for chunk in chunks:
            await service._push_llm_text(chunk)
        return "".join(self.spoken)

    async def test_plain_text_passes_through(self):
        service = self._service("gemma4:12b")
        self.assertEqual(
            await self._stream(service, "Thank you", " for your help!"), "Thank you for your help!"
        )

    async def test_a_trailing_switch_before_a_tool_call_is_dropped(self):
        service = self._service("gemma4:12b")
        self.assertEqual(
            await self._stream(service, "Thank you for your help!", "<channel|>"),
            "Thank you for your help!",
        )

    async def test_a_channel_section_is_dropped_whole(self):
        # A thought the model opened by name, closed by the next switch.
        service = self._service("gemma4:12b")
        self.assertEqual(
            await self._stream(
                service,
                "Sure.",
                "\n<channel|>",
                "thought",
                "\nweighing it",
                "<channel|>",
                " Six PM.",
            ),
            "Sure.\n Six PM.",
        )

    async def test_a_switch_inside_one_chunk_splits_it(self):
        service = self._service("gemma4:12b")
        self.assertEqual(await self._stream(service, "Bye!<channel|>tool stuff"), "Bye!")

    async def test_a_new_response_starts_outside_any_channel(self):
        service = self._service("gemma4:12b")
        await self._stream(service, "Bye!", "<channel|>", "dropped")
        self.assertEqual(await self._stream(service, "Hello again"), "Hello again")

    async def test_other_models_are_left_alone(self):
        service = self._service("gpt-4.1")
        self.assertEqual(await self._stream(service, "a<channel|>b"), "a<channel|>b")

    async def test_ollama_inherits_it(self):
        service = self._service("gemma4:12b", cls=OLLamaLLMService)
        self.assertEqual(await self._stream(service, "Bye!", "<channel|>"), "Bye!")


if __name__ == "__main__":
    unittest.main()
