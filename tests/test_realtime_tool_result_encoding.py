#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The context aggregator stores tool call results as already JSON-encoded
strings. `_send_tool_result()` on the realtime LLM services must send that
string through unchanged, not re-encode it with another `json.dumps()`
(which would turn `{"temp": 72}` into the string literal `"{\\"temp\\": 72}"`).
"""

import unittest
from unittest.mock import AsyncMock

from pipecat.services.inworld.realtime.llm import InworldRealtimeLLMService
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService


class TestOpenAIRealtimeToolResultEncoding(unittest.IsolatedAsyncioTestCase):
    async def test_send_tool_result_passes_through_pre_encoded_json(self):
        service = OpenAIRealtimeLLMService(api_key="test")
        service.send_client_event = AsyncMock()

        already_encoded = '{"temperature": 72}'
        await service._send_tool_result("call_123", already_encoded)

        sent_event = service.send_client_event.call_args.args[0]
        self.assertEqual(sent_event.item.output, already_encoded)

    async def test_send_tool_result_preserves_completed_sentinel(self):
        service = OpenAIRealtimeLLMService(api_key="test")
        service.send_client_event = AsyncMock()

        await service._send_tool_result("call_123", "COMPLETED")

        sent_event = service.send_client_event.call_args.args[0]
        self.assertEqual(sent_event.item.output, "COMPLETED")


class TestInworldRealtimeToolResultEncoding(unittest.IsolatedAsyncioTestCase):
    async def test_send_tool_result_passes_through_pre_encoded_json(self):
        service = InworldRealtimeLLMService(api_key="test")
        service.send_client_event = AsyncMock()

        already_encoded = '{"temperature": 72}'
        await service._send_tool_result("call_123", already_encoded)

        sent_event = service.send_client_event.call_args.args[0]
        self.assertEqual(sent_event.item.output, already_encoded)


class TestXAIRealtimeToolResultEncoding(unittest.IsolatedAsyncioTestCase):
    async def test_send_tool_result_passes_through_pre_encoded_json(self):
        service = GrokRealtimeLLMService(api_key="test")
        service.send_client_event = AsyncMock()

        already_encoded = '{"temperature": 72}'
        await service._send_tool_result("call_123", already_encoded)

        sent_event = service.send_client_event.call_args.args[0]
        self.assertEqual(sent_event.item.output, already_encoded)


if __name__ == "__main__":
    unittest.main()
