#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ultravox delivers a tool's result once, on the channel its call is on.

An async call gets a placeholder ``client_tool_result`` as soon as Ultravox
invokes it, so its actual result has to reach the model as user-side text
whether the context holds it as a deferred async message or, when it arrived
before the conversation moved on, as an ordinary tool result. A synchronous
call's result goes out as the ``client_tool_result`` itself.
"""

import unittest
from unittest.mock import AsyncMock

import pytest

from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext


class TestUltravoxToolResultDelivery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mod = pytest.importorskip("pipecat.services.ultravox.llm")
        self.mod = mod
        self.service = mod.UltravoxRealtimeLLMService(
            params=mod.OneShotInputParams(api_key="test-key", system_prompt="test"),
        )
        self.service._send = AsyncMock()
        self.service._send_user_text = AsyncMock()

    def _tool_result(self, tool_call_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

    async def test_sync_result_is_sent_as_client_tool_result(self):
        context = LLMContext(messages=[self._tool_result("call-1", '{"temperature": 72}')])

        await self.service._handle_context(context)

        self.service._send.assert_awaited_once_with(
            {
                "type": "client_tool_result",
                "invocationId": "call-1",
                "result": '{"temperature": 72}',
            }
        )
        self.service._send_user_text.assert_not_awaited()
        self.assertIn("call-1", self.service._completed_tool_calls)

    async def test_async_result_settled_in_place_is_sent_as_user_text(self):
        # The placeholder already answered the invocation on Ultravox's side.
        self.service._started_placeholder_sent.add("call-1")
        context = LLMContext(messages=[self._tool_result("call-1", '{"temperature": 72}')])

        await self.service._handle_context(context)

        self.service._send.assert_not_awaited()
        self.service._send_user_text.assert_awaited_once_with(
            self.mod._ASYNC_TOOL_FINAL_RESULT_TEMPLATE.format(
                tool_call_id="call-1", result='{"temperature": 72}'
            )
        )
        self.assertIn("call-1", self.service._completed_tool_calls)

    async def test_deferred_async_result_is_sent_as_user_text(self):
        self.service._started_placeholder_sent.add("call-1")
        context = LLMContext(
            messages=[
                async_tool_messages.build_started_message("call-1"),
                async_tool_messages.build_final_result_message("call-1", '{"temperature": 72}'),
            ]
        )

        await self.service._handle_context(context)

        self.service._send.assert_not_awaited()
        self.service._send_user_text.assert_awaited_once_with(
            self.mod._ASYNC_TOOL_FINAL_RESULT_TEMPLATE.format(
                tool_call_id="call-1", result='{"temperature": 72}'
            )
        )

    async def test_result_is_delivered_once_across_context_pushes(self):
        self.service._started_placeholder_sent.add("call-1")
        context = LLMContext(messages=[self._tool_result("call-1", '{"temperature": 72}')])

        await self.service._handle_context(context)
        await self.service._handle_context(context)

        self.service._send_user_text.assert_awaited_once()
