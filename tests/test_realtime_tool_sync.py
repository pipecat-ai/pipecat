#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Realtime services sync direct-function handlers on LLMSetToolsFrame.

Continuous (speech-to-speech) sessions don't get a fresh context frame per turn,
so they sync registered handlers from the ``LLMSetToolsFrame`` themselves — the
base service only does this on ``LLMContextFrame``.
"""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import LLMSetToolsFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


async def sample_tool(params: FunctionCallParams, query: str):
    """A sample direct function.

    Args:
        query: The query to run.
    """
    await params.result_callback({})


class TestRealtimeToolSync(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> OpenAIRealtimeLLMService:
        service = OpenAIRealtimeLLMService(api_key="test-key")
        # Stub the parts that would touch the network / pipeline so we can drive
        # process_frame without a live session.
        service._send_session_update = AsyncMock()
        service.push_frame = AsyncMock()
        return service

    async def test_set_tools_frame_registers_direct_function(self):
        service = self._service()
        await service.process_frame(
            LLMSetToolsFrame(tools=[sample_tool]), FrameDirection.DOWNSTREAM
        )
        self.assertTrue(service.has_function("sample_tool"))

    async def test_set_tools_frame_prunes_deadvertised_direct_function(self):
        service = self._service()
        await service.process_frame(
            LLMSetToolsFrame(tools=[sample_tool]), FrameDirection.DOWNSTREAM
        )
        self.assertTrue(service.has_function("sample_tool"))
        # Clearing the advertised tools prunes the auto-registered handler.
        await service.process_frame(LLMSetToolsFrame(tools=[]), FrameDirection.DOWNSTREAM)
        self.assertNotIn("sample_tool", service._functions)


if __name__ == "__main__":
    unittest.main()
