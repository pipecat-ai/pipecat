#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Assistant text that accompanies tool calls survives adapter conversion."""

import unittest

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext

TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}


def _messages(content):
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": content, "tool_calls": [TOOL_CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 20}'},
    ]


class TestAssistantTextWithToolCalls(unittest.TestCase):
    def _params(self, adapter, content):
        context = LLMContext(messages=_messages(content))
        if isinstance(adapter, AnthropicLLMAdapter):
            return adapter.get_llm_invocation_params(context, enable_prompt_caching=False)
        return adapter.get_llm_invocation_params(context)

    def test_anthropic_str_content(self):
        params = self._params(AnthropicLLMAdapter(), "Let me check.")
        assistant = params["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"][0], {"type": "text", "text": "Let me check."})
        self.assertEqual(assistant["content"][1]["type"], "tool_use")

    def test_anthropic_list_content(self):
        params = self._params(AnthropicLLMAdapter(), [{"type": "text", "text": "Let me check."}])
        assistant = params["messages"][1]
        self.assertEqual(assistant["content"][0], {"type": "text", "text": "Let me check."})
        self.assertEqual(assistant["content"][1]["type"], "tool_use")

    def test_anthropic_no_text_has_only_tool_use(self):
        params = self._params(AnthropicLLMAdapter(), None)
        self.assertEqual([b["type"] for b in params["messages"][1]["content"]], ["tool_use"])

    def test_bedrock_str_content(self):
        params = self._params(AWSBedrockLLMAdapter(), "Let me check.")
        assistant = params["messages"][1]
        self.assertEqual(assistant["content"][0], {"text": "Let me check."})
        self.assertIn("toolUse", assistant["content"][1])

    def test_bedrock_no_text_has_only_tool_use(self):
        params = self._params(AWSBedrockLLMAdapter(), None)
        self.assertEqual(len(params["messages"][1]["content"]), 1)
        self.assertIn("toolUse", params["messages"][1]["content"][0])

    def test_gemini_str_content(self):
        params = self._params(GeminiLLMAdapter(), "Let me check.")
        model = params["messages"][1]
        self.assertEqual(model.role, "model")
        self.assertEqual(model.parts[0].text, "Let me check.")
        self.assertIsNotNone(model.parts[1].function_call)

    def test_gemini_no_text_has_only_function_call(self):
        params = self._params(GeminiLLMAdapter(), None)
        parts = params["messages"][1].parts
        self.assertEqual(len(parts), 1)
        self.assertIsNotNone(parts[0].function_call)


if __name__ == "__main__":
    unittest.main()
