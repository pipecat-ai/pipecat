#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A single malformed message must not silently discard the rest of the
conversation context: the Anthropic and AWS Bedrock adapters should skip
only the message that fails to convert and keep the rest of the history.
"""

import unittest

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext

# A `data:` URL with no comma: `url.split(",")[1]` raises IndexError while
# extracting the base64 payload. Surrounded by valid messages of different
# roles so a merge of adjacent same-role messages can't mask a dropped message.
_MESSAGES_WITH_ONE_MALFORMED = [
    {"role": "user", "content": "What's in this image?"},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64"},
            }
        ],
    },
    {"role": "assistant", "content": "Here is a description."},
]


class TestAnthropicMessageConversionErrors(unittest.TestCase):
    def setUp(self):
        self.adapter = AnthropicLLMAdapter()

    def test_malformed_message_is_skipped_and_rest_are_kept(self):
        context = LLMContext(messages=_MESSAGES_WITH_ONE_MALFORMED)

        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][1]["role"], "assistant")

    def test_valid_messages_are_not_affected(self):
        context = LLMContext(messages=[{"role": "user", "content": "hello"}])

        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        self.assertEqual(len(params["messages"]), 1)


class TestBedrockMessageConversionErrors(unittest.TestCase):
    def setUp(self):
        self.adapter = AWSBedrockLLMAdapter()

    def test_malformed_message_is_skipped_and_rest_are_kept(self):
        context = LLMContext(messages=_MESSAGES_WITH_ONE_MALFORMED)

        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][1]["role"], "assistant")

    def test_valid_messages_are_not_affected(self):
        context = LLMContext(messages=[{"role": "user", "content": "hello"}])

        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["messages"]), 1)


if __name__ == "__main__":
    unittest.main()
