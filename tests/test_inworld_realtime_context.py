#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Inworld Realtime context synchronization."""

import unittest
from unittest.mock import AsyncMock, Mock, patch

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService


class TestInworldRealtimeContext(unittest.IsolatedAsyncioTestCase):
    async def test_connect_generates_unique_session_keys(self):
        services = [InworldRealtimeLLMService(api_key="test") for _ in range(2)]
        websocket = AsyncMock()

        def discard_task(coro, *args, **kwargs):
            coro.close()
            return None

        for service in services:
            service.create_task = Mock(side_effect=discard_task)

        with patch(
            "pipecat.services.inworld.realtime.llm.websocket_connect",
            new=AsyncMock(return_value=websocket),
        ) as websocket_connect:
            with patch("pipecat.services.inworld.realtime.llm.time.time", return_value=1.0):
                await services[0]._connect()
                await services[1]._connect()

        session_keys = [
            call.kwargs["uri"].split("key=", 1)[1].split("&", 1)[0]
            for call in websocket_connect.await_args_list
        ]
        self.assertEqual(len(session_keys), 2)
        self.assertNotEqual(session_keys[0], session_keys[1])

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
