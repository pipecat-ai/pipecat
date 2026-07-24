#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Inworld Realtime context synchronization."""

import unittest
from unittest.mock import AsyncMock

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService


class TestInworldRealtimeContext(unittest.IsolatedAsyncioTestCase):
    async def test_tool_update_does_not_consume_server_vad_user_turn(self):
        context = LLMContext([{"role": "developer", "content": "Be helpful."}])
        service = InworldRealtimeLLMService(api_key="test")
        service._context = context
        service._last_context_message_count = len(context.get_messages())
        service._server_vad_handled_turn = True
        service._process_completed_function_calls = AsyncMock()
        service.send_client_event = AsyncMock()

        context.add_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": '{"temp": 22}'},
            ]
        )
        await service._handle_context(context)

        self.assertTrue(service._server_vad_handled_turn)
        service.send_client_event.assert_not_awaited()

        context.add_message({"role": "user", "content": "What is the weather?"})
        await service._handle_context(context)

        self.assertFalse(service._server_vad_handled_turn)
        service.send_client_event.assert_not_awaited()
        self.assertEqual(service._last_context_message_count, len(context.get_messages()))


if __name__ == "__main__":
    unittest.main()
