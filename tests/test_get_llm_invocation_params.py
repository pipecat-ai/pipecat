#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Unit tests for OpenAI adapter's get_llm_invocation_params() method.

These tests focus specifically on the "messages" field generation, ensuring:
1. LLMStandardMessage objects are passed through unchanged
2. LLMSpecificMessage objects with llm='openai' are included and their content extracted
3. LLMSpecificMessage objects with llm != 'openai' are filtered out
4. Complex message structures (like multi-part content) are preserved
5. Edge cases like empty message lists are handled correctly
"""

import unittest

from openai.types.chat import ChatCompletionMessage

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
    LLMStandardMessage,
)


class TestOpenAIGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = OpenAILLMAdapter()

    def test_standard_messages_passed_through_unchanged(self):
        """Test that LLMStandardMessage objects are passed through unchanged to OpenAI params."""
        # Create standard messages (OpenAI format)
        standard_messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        ]

        # Create context with these messages
        context = LLMContext(messages=standard_messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Verify messages are passed through unchanged
        self.assertEqual(params["messages"], standard_messages)
        self.assertEqual(len(params["messages"]), 3)

        # Verify content matches exactly
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")
        self.assertEqual(params["messages"][1]["content"], "Hello, how are you?")
        self.assertEqual(params["messages"][2]["content"], "I'm doing well, thank you for asking!")

    def test_openai_specific_messages_included(self):
        """Test that LLMSpecificMessage objects with llm='openai' are included."""
        # Create a mix of standard and OpenAI-specific messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            LLMSpecificMessage(
                llm="openai", message={"role": "user", "content": "OpenAI specific message"}
            ),
            {"role": "assistant", "content": "Standard response"},
        ]

        # Create context with these messages
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Verify all messages are included (OpenAI-specific should be included)
        self.assertEqual(len(params["messages"]), 3)

        # First message should be standard
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")

        # Second message should be the OpenAI-specific one
        self.assertEqual(
            params["messages"][1], {"role": "user", "content": "OpenAI specific message"}
        )

        # Third message should be standard
        self.assertEqual(params["messages"][2]["content"], "Standard response")

    def test_non_openai_specific_messages_filtered_out(self):
        """Test that LLMSpecificMessage objects with llm != 'openai' are filtered out."""
        # Create messages with different LLM-specific ones
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            LLMSpecificMessage(
                llm="anthropic", message={"role": "user", "content": "Anthropic specific message"}
            ),
            LLMSpecificMessage(
                llm="gemini", message={"role": "user", "content": "Gemini specific message"}
            ),
            {"role": "user", "content": "Standard user message"},
            LLMSpecificMessage(
                llm="openai", message={"role": "assistant", "content": "OpenAI specific response"}
            ),
        ]

        # Create context with these messages
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Should only include standard messages and OpenAI-specific ones
        # (3 total: system, standard user, openai assistant)
        self.assertEqual(len(params["messages"]), 3)

        # Verify the correct messages are included
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")
        self.assertEqual(params["messages"][1]["content"], "Standard user message")
        self.assertEqual(
            params["messages"][2], {"role": "assistant", "content": "OpenAI specific response"}
        )

    def test_complex_message_content_preserved(self):
        """Test that complex message content (like multi-part messages) is preserved."""
        # Create a message with complex content structure (text + image)
        complex_image_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."},
                },
            ],
        }

        # Create a message with multiple text blocks
        multi_text_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me analyze this step by step:"},
                {"type": "text", "text": "1. First, I'll examine the visual elements"},
                {"type": "text", "text": "2. Then I'll provide my conclusions"},
            ],
        }

        messages = [
            {"role": "system", "content": "You are a helpful assistant that can analyze images."},
            complex_image_message,
            multi_text_message,
        ]

        # Create context with these messages
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Verify complex content is preserved
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][1], complex_image_message)
        self.assertEqual(params["messages"][2], multi_text_message)

        # Verify the image message structure is maintained
        image_content = params["messages"][1]["content"]
        self.assertIsInstance(image_content, list)
        self.assertEqual(len(image_content), 2)
        self.assertEqual(image_content[0]["type"], "text")
        self.assertEqual(image_content[1]["type"], "image_url")

        # Verify the multi-text message structure is maintained
        text_content = params["messages"][2]["content"]
        self.assertIsInstance(text_content, list)
        self.assertEqual(len(text_content), 3)
        for i, text_block in enumerate(text_content):
            self.assertEqual(text_block["type"], "text")
        self.assertEqual(text_content[0]["text"], "Let me analyze this step by step:")
        self.assertEqual(text_content[1]["text"], "1. First, I'll examine the visual elements")
        self.assertEqual(text_content[2]["text"], "2. Then I'll provide my conclusions")


if __name__ == "__main__":
    unittest.main()
