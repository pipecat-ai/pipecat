#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Unit tests for LLM adapters' get_llm_invocation_params() method.

These tests focus specifically on the "messages" field generation for different adapters, ensuring:

For OpenAI adapter:
1. LLMStandardMessage objects are passed through unchanged
2. LLMSpecificMessage objects with llm='openai' are included and others are filtered out
3. Complex message structures (like multi-part content) are preserved
4. System instructions are preserved throughout messages at any position

For Gemini adapter:
1. LLMStandardMessage objects are converted to Gemini Content format
2. LLMSpecificMessage objects with llm='google' are included and others are filtered out
3. Complex message structures (image, audio, multi-text) are converted to appropriate Gemini format
4. System messages are extracted as system_instruction (without duplication)
5. Single system instruction is converted to user message when no other messages exist
6. Multiple system instructions: first extracted, later ones converted to user messages

For Anthropic adapter:
1. LLMStandardMessage objects are converted to Anthropic MessageParam format
2. LLMSpecificMessage objects with llm='anthropic' are included and others are filtered out
3. Complex message structures (image, multi-text) are converted to appropriate Anthropic format
4. System messages: first extracted as system parameter, later ones converted to user messages
5. Consecutive messages with same role are merged into multi-content-block messages
6. Empty text content is converted to "(empty)"

For AWS Bedrock adapter:
1. LLMStandardMessage objects are converted to AWS Bedrock format
2. LLMSpecificMessage objects with llm='aws' are included and others are filtered out
3. Complex message structures (image, multi-text) are converted to appropriate AWS Bedrock format
4. System messages: first extracted as system parameter, later ones converted to user messages
5. Consecutive messages with same role are merged into multi-content-block messages
6. Empty text content is converted to "(empty)"
"""

import unittest

from google.genai.types import Content, Part
from openai.types.chat import ChatCompletionMessage

from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
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

    def test_llm_specific_message_filtering(self):
        """Test that OpenAI-specific messages are included and others are filtered out."""
        # Create messages with different LLM-specific ones
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            AnthropicLLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "Anthropic specific message"}
            ),
            GeminiLLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "Gemini specific message"}
            ),
            {"role": "user", "content": "Standard user message"},
            self.adapter.create_llm_specific_message(
                {"role": "assistant", "content": "OpenAI specific response"}
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

    def test_system_instructions_preserved_throughout_messages(self):
        """Test that OpenAI adapter preserves system instructions sprinkled throughout messages."""
        # Create messages with system instructions at different positions
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "Remember to be concise."},
            {"role": "user", "content": "Tell me about Python."},
            {"role": "system", "content": "Use simple language."},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        # Create context with these messages
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # OpenAI should preserve all messages unchanged, including multiple system messages
        self.assertEqual(len(params["messages"]), 7)

        # Verify system messages are preserved at their original positions
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")

        self.assertEqual(params["messages"][3]["role"], "system")
        self.assertEqual(params["messages"][3]["content"], "Remember to be concise.")

        self.assertEqual(params["messages"][5]["role"], "system")
        self.assertEqual(params["messages"][5]["content"], "Use simple language.")

        # Verify other messages remain unchanged
        self.assertEqual(params["messages"][1]["role"], "user")
        self.assertEqual(params["messages"][2]["role"], "assistant")
        self.assertEqual(params["messages"][4]["role"], "user")
        self.assertEqual(params["messages"][6]["role"], "assistant")


class TestGeminiGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = GeminiLLMAdapter()

    def test_standard_messages_converted_to_gemini_format(self):
        """Test that LLMStandardMessage objects are converted to Gemini Content format."""
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

        # Verify system instruction is extracted
        self.assertEqual(params["system_instruction"], "You are a helpful assistant.")

        # Verify messages are converted to Gemini format (2 messages: user + model)
        self.assertEqual(len(params["messages"]), 2)

        # Check first message (user)
        user_msg = params["messages"][0]
        self.assertIsInstance(user_msg, Content)
        self.assertEqual(user_msg.role, "user")
        self.assertEqual(len(user_msg.parts), 1)
        self.assertEqual(user_msg.parts[0].text, "Hello, how are you?")

        # Check second message (assistant -> model)
        model_msg = params["messages"][1]
        self.assertIsInstance(model_msg, Content)
        self.assertEqual(model_msg.role, "model")
        self.assertEqual(len(model_msg.parts), 1)
        self.assertEqual(model_msg.parts[0].text, "I'm doing well, thank you for asking!")

    def test_llm_specific_message_filtering(self):
        """Test that Gemini-specific messages are included and others are filtered out."""
        # Create messages with different LLM-specific ones
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            OpenAILLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "OpenAI specific message"}
            ),
            AnthropicLLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "Anthropic specific message"}
            ),
            {"role": "user", "content": "Standard user message"},
            self.adapter.create_llm_specific_message(
                Content(role="model", parts=[Part(text="Gemini specific response")]),
            ),
        ]

        # Create context with these messages
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Should only include standard messages and Gemini-specific ones
        # (2 total: converted standard user + gemini model)
        self.assertEqual(len(params["messages"]), 2)

        # Verify system instruction
        self.assertEqual(params["system_instruction"], "You are a helpful assistant.")

        # Verify the correct messages are included
        self.assertEqual(params["messages"][0].role, "user")
        self.assertEqual(params["messages"][0].parts[0].text, "Standard user message")

        self.assertEqual(params["messages"][1].role, "model")
        self.assertEqual(params["messages"][1].parts[0].text, "Gemini specific response")

    def test_complex_message_content_preserved(self):
        """Test that complex message content (like multi-part messages) is preserved and converted.

        This test covers image, audio, and multi-text content conversion to Gemini format.
        """
        # Create a message with complex content structure (text + image)
        # Using a minimal valid base64 image data
        complex_image_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    },
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

        # Create a message with audio input (text + audio)
        # Using a minimal valid base64 audio data (16 bytes of WAV header)
        audio_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you transcribe this audio?"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
                        "format": "wav",
                    },
                },
            ],
        }

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze images and audio.",
            },
            complex_image_message,
            multi_text_message,
            audio_message,
        ]

        # Create context with these messages
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Verify system instruction
        self.assertEqual(
            params["system_instruction"],
            "You are a helpful assistant that can analyze images and audio.",
        )

        # Verify complex content is converted to Gemini format
        # Note: Gemini adapter may add system instruction back as user message in some cases
        self.assertGreaterEqual(len(params["messages"]), 3)

        # Find the different message types
        user_with_image = None
        model_with_text = None
        user_with_audio = None

        for msg in params["messages"]:
            if msg.role == "user" and len(msg.parts) == 2:
                # Check if it's image or audio based on the text content
                if hasattr(msg.parts[0], "text") and "image" in msg.parts[0].text:
                    user_with_image = msg
                elif hasattr(msg.parts[0], "text") and "audio" in msg.parts[0].text:
                    user_with_audio = msg
            elif msg.role == "model" and len(msg.parts) == 3:
                model_with_text = msg

        # Verify the image message structure is converted properly
        self.assertIsNotNone(user_with_image, "Should have user message with image")
        self.assertEqual(len(user_with_image.parts), 2)

        # First part should be text
        self.assertEqual(user_with_image.parts[0].text, "What's in this image?")

        # Second part should be image data (converted to Blob)
        self.assertIsNotNone(user_with_image.parts[1].inline_data)
        self.assertEqual(user_with_image.parts[1].inline_data.mime_type, "image/jpeg")

        # Verify the audio message structure is converted properly
        self.assertIsNotNone(user_with_audio, "Should have user message with audio")
        self.assertEqual(len(user_with_audio.parts), 2)

        # First part should be text
        self.assertEqual(user_with_audio.parts[0].text, "Can you transcribe this audio?")

        # Second part should be audio data (converted to Blob)
        self.assertIsNotNone(user_with_audio.parts[1].inline_data)
        self.assertEqual(user_with_audio.parts[1].inline_data.mime_type, "audio/wav")

        # Verify the multi-text message structure is converted properly
        self.assertIsNotNone(model_with_text, "Should have model message with multi-text")
        self.assertEqual(len(model_with_text.parts), 3)

        # All parts should be text
        expected_texts = [
            "Let me analyze this step by step:",
            "1. First, I'll examine the visual elements",
            "2. Then I'll provide my conclusions",
        ]
        for i, expected_text in enumerate(expected_texts):
            self.assertEqual(model_with_text.parts[i].text, expected_text)

    def test_single_system_instruction_converted_to_user(self):
        """Test that when there's only a system instruction, it gets converted to user message."""
        # Create context with only a system message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        # System instruction should be extracted
        self.assertEqual(params["system_instruction"], "You are a helpful assistant.")

        # But since there are no other messages, it should also be added back as a user message
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0].role, "user")
        self.assertEqual(params["messages"][0].parts[0].text, "You are a helpful assistant.")

    def test_multiple_system_instructions_handling(self):
        """Test that first system instruction is extracted, later ones converted to user messages."""
        # Create messages with multiple system instructions
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "Remember to be concise."},
            {"role": "user", "content": "Tell me about Python."},
            {"role": "system", "content": "Use simple language."},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        # First system instruction should be extracted
        self.assertEqual(params["system_instruction"], "You are a helpful assistant.")

        # Should have 6 messages (original 7 minus 1 system instruction that was extracted)
        self.assertEqual(len(params["messages"]), 6)

        # Find the converted system messages (should be user role now)
        converted_system_messages = []
        for msg in params["messages"]:
            if msg.role == "user" and (
                msg.parts[0].text == "Remember to be concise."
                or msg.parts[0].text == "Use simple language."
            ):
                converted_system_messages.append(msg.parts[0].text)

        # Should have 2 converted system messages
        self.assertEqual(len(converted_system_messages), 2)
        self.assertIn("Remember to be concise.", converted_system_messages)
        self.assertIn("Use simple language.", converted_system_messages)

        # Verify that regular user and assistant messages are preserved
        user_messages = [msg for msg in params["messages"] if msg.role == "user"]
        model_messages = [msg for msg in params["messages"] if msg.role == "model"]

        # Should have 4 user messages: 2 original + 2 converted from system
        self.assertEqual(len(user_messages), 4)
        # Should have 2 model messages (converted from assistant)
        self.assertEqual(len(model_messages), 2)


class TestAnthropicGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = AnthropicLLMAdapter()

    def test_standard_messages_converted_to_anthropic_format(self):
        """Test that LLMStandardMessage objects are converted to Anthropic MessageParam format."""
        # Create standard messages
        standard_messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        # Create context
        context = LLMContext(messages=standard_messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # Verify system instruction is extracted
        self.assertEqual(params["system"], "You are a helpful assistant.")

        # Verify messages are in the params (2 messages after system extraction)
        self.assertIn("messages", params)
        self.assertEqual(len(params["messages"]), 2)

        # Check first message (user)
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertEqual(user_msg["content"], "Hello, how are you?")

        # Check second message (assistant)
        assistant_msg = params["messages"][1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertEqual(assistant_msg["content"], "I'm doing well, thank you!")

    def test_llm_specific_message_filtering(self):
        """Test that Anthropic-specific messages are included and others are filtered out."""
        # Create anthropic-specific message content
        anthropic_message_content = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "fake_data"},
                },
            ],
        }

        messages = [
            {"role": "user", "content": "Standard message"},
            OpenAILLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "OpenAI specific"}
            ),
            GeminiLLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "Google specific"}
            ),
            self.adapter.create_llm_specific_message(anthropic_message_content),
            {"role": "assistant", "content": "Response"},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # Should only have 2 messages after merging consecutive user messages: merged user + standard response
        # (openai and google specific filtered out, standard + anthropic-specific merged)
        self.assertEqual(len(params["messages"]), 2)

        # First message: merged user message (standard + anthropic-specific)
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        # Should have 3 content blocks: standard text + anthropic text + anthropic image
        self.assertEqual(len(user_msg["content"]), 3)
        self.assertEqual(user_msg["content"][0]["type"], "text")
        self.assertEqual(user_msg["content"][0]["text"], "Standard message")
        self.assertEqual(user_msg["content"][1]["type"], "text")
        self.assertEqual(user_msg["content"][1]["text"], "Hello")
        self.assertEqual(user_msg["content"][2]["type"], "image")

        # Second message: standard response
        self.assertEqual(params["messages"][1]["content"], "Response")

    def test_consecutive_same_role_messages_merged(self):
        """Test that consecutive messages with the same role are merged into multi-content blocks."""
        messages = [
            {"role": "user", "content": "First user message"},
            {"role": "user", "content": "Second user message"},
            {"role": "user", "content": "Third user message"},
            {"role": "assistant", "content": "First assistant message"},
            {"role": "assistant", "content": "Second assistant message"},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # Should have 2 messages after merging (1 user, 1 assistant)
        self.assertEqual(len(params["messages"]), 2)

        # Check merged user message
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(len(user_msg["content"]), 3)
        self.assertEqual(user_msg["content"][0]["type"], "text")
        self.assertEqual(user_msg["content"][0]["text"], "First user message")
        self.assertEqual(user_msg["content"][1]["type"], "text")
        self.assertEqual(user_msg["content"][1]["text"], "Second user message")
        self.assertEqual(user_msg["content"][2]["type"], "text")
        self.assertEqual(user_msg["content"][2]["text"], "Third user message")

        # Check merged assistant message
        assistant_msg = params["messages"][1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIsInstance(assistant_msg["content"], list)
        self.assertEqual(len(assistant_msg["content"]), 2)
        self.assertEqual(assistant_msg["content"][0]["type"], "text")
        self.assertEqual(assistant_msg["content"][0]["text"], "First assistant message")
        self.assertEqual(assistant_msg["content"][1]["type"], "text")
        self.assertEqual(assistant_msg["content"][1]["text"], "Second assistant message")

    def test_empty_text_converted_to_empty_placeholder(self):
        """Test that empty text content is converted to "(empty)" string."""
        messages = [
            {"role": "user", "content": ""},  # Empty string
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},  # Empty text in list content
                    {"type": "text", "text": "Valid text"},
                ],
            },
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # Check that empty string content was converted
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["content"], "(empty)")

        # Check that empty text in list content was converted
        assistant_msg = params["messages"][1]
        self.assertIsInstance(assistant_msg["content"], list)
        self.assertEqual(assistant_msg["content"][0]["text"], "(empty)")
        self.assertEqual(assistant_msg["content"][1]["text"], "Valid text")

    def test_complex_message_content_preserved(self):
        """Test that complex message structures (text + image) are properly converted to Anthropic format."""
        # Create a complex message with both text and image content
        complex_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,fake_image_data"},
                },
                {"type": "text", "text": "Please describe it in detail."},
            ],
        }

        messages = [
            complex_message,
            {"role": "assistant", "content": "I can see the image clearly."},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # Verify complex message structure is preserved and converted
        self.assertEqual(len(params["messages"]), 2)
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(len(user_msg["content"]), 3)

        # Note: Anthropic adapter reorders single images to come before text, as per Anthropic docs
        # Check image part (should be moved to first position and converted from image_url to image)
        self.assertEqual(user_msg["content"][0]["type"], "image")
        self.assertIn("source", user_msg["content"][0])
        self.assertEqual(user_msg["content"][0]["source"]["type"], "base64")
        self.assertEqual(user_msg["content"][0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(user_msg["content"][0]["source"]["data"], "fake_image_data")

        # Check first text part (moved to second position)
        self.assertEqual(user_msg["content"][1]["type"], "text")
        self.assertEqual(user_msg["content"][1]["text"], "What do you see in this image?")

        # Check second text part (moved to third position)
        self.assertEqual(user_msg["content"][2]["type"], "text")
        self.assertEqual(user_msg["content"][2]["text"], "Please describe it in detail.")

    def test_multiple_system_instructions_handling(self):
        """Test that first system instruction is extracted, later ones converted to user messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "Remember to be concise."},  # Later system message
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # System instruction should be extracted from first message
        self.assertEqual(params["system"], "You are a helpful assistant.")

        # Should have 3 messages remaining (system message was removed, later system converted to user)
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Hello")
        self.assertEqual(params["messages"][1]["role"], "assistant")
        self.assertEqual(params["messages"][1]["content"], "Hi there!")

        # Later system message should be converted to user role
        self.assertEqual(params["messages"][2]["role"], "user")
        self.assertEqual(params["messages"][2]["content"], "Remember to be concise.")

    def test_single_system_message_converted_to_user(self):
        """Test that a single system message is converted to user role when no other messages exist."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # System should be NOT_GIVEN since we only have one message
        from anthropic import NOT_GIVEN

        self.assertEqual(params["system"], NOT_GIVEN)

        # Single system message should be converted to user role
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")


class TestAWSBedrockGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = AWSBedrockLLMAdapter()

    def test_standard_messages_converted_to_aws_bedrock_format(self):
        """Test that LLMStandardMessage objects are converted to AWS Bedrock format."""
        # Create standard messages
        standard_messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        # Create context
        context = LLMContext(messages=standard_messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Verify system instruction is extracted (in AWS Bedrock format)
        self.assertIsInstance(params["system"], list)
        self.assertEqual(len(params["system"]), 1)
        self.assertEqual(params["system"][0]["text"], "You are a helpful assistant.")

        # Verify messages are in the params (2 messages after system extraction)
        self.assertIn("messages", params)
        self.assertEqual(len(params["messages"]), 2)

        # Check first message (user) - should be converted to AWS Bedrock format
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(len(user_msg["content"]), 1)
        self.assertEqual(user_msg["content"][0]["text"], "Hello, how are you?")

        # Check second message (assistant) - should be converted to AWS Bedrock format
        assistant_msg = params["messages"][1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIsInstance(assistant_msg["content"], list)
        self.assertEqual(len(assistant_msg["content"]), 1)
        self.assertEqual(assistant_msg["content"][0]["text"], "I'm doing well, thank you!")

    def test_llm_specific_message_filtering(self):
        """Test that AWS-specific messages are included and others are filtered out."""
        # Create aws-specific message content (which is what AWS Bedrock uses)
        aws_message_content = {
            "role": "user",
            "content": [
                {"text": "Hello"},
                {"image": {"format": "jpeg", "source": {"bytes": b"fake_image_data"}}},
            ],
        }

        messages = [
            {"role": "user", "content": "Standard message"},
            OpenAILLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "OpenAI specific"}
            ),
            GeminiLLMAdapter().create_llm_specific_message(
                {"role": "user", "content": "Google specific"}
            ),
            self.adapter.create_llm_specific_message(message=aws_message_content),
            {"role": "assistant", "content": "Response"},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Should only have 2 messages after merging consecutive user messages: merged user + standard response
        # (openai and google specific filtered out, standard + aws-specific merged)
        self.assertEqual(len(params["messages"]), 2)

        # First message: merged user message (standard + aws-specific)
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        # Should have 3 content blocks: standard text + aws text + aws image
        self.assertEqual(len(user_msg["content"]), 3)
        self.assertEqual(user_msg["content"][0]["text"], "Standard message")
        self.assertEqual(user_msg["content"][1]["text"], "Hello")
        self.assertIn("image", user_msg["content"][2])

        # Second message: standard response
        self.assertEqual(params["messages"][1]["content"][0]["text"], "Response")

    def test_consecutive_same_role_messages_merged(self):
        """Test that consecutive messages with the same role are merged into multi-content blocks."""
        messages = [
            {"role": "user", "content": "First user message"},
            {"role": "user", "content": "Second user message"},
            {"role": "user", "content": "Third user message"},
            {"role": "assistant", "content": "First assistant message"},
            {"role": "assistant", "content": "Second assistant message"},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Should have 2 messages after merging (1 user, 1 assistant)
        self.assertEqual(len(params["messages"]), 2)

        # Check merged user message
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(len(user_msg["content"]), 3)
        self.assertEqual(user_msg["content"][0]["text"], "First user message")
        self.assertEqual(user_msg["content"][1]["text"], "Second user message")
        self.assertEqual(user_msg["content"][2]["text"], "Third user message")

        # Check merged assistant message
        assistant_msg = params["messages"][1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertIsInstance(assistant_msg["content"], list)
        self.assertEqual(len(assistant_msg["content"]), 2)
        self.assertEqual(assistant_msg["content"][0]["text"], "First assistant message")
        self.assertEqual(assistant_msg["content"][1]["text"], "Second assistant message")

    def test_empty_text_converted_to_empty_placeholder(self):
        """Test that empty text content is converted to "(empty)" string."""
        messages = [
            {"role": "user", "content": ""},  # Empty string
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""},  # Empty text in list content
                    {"type": "text", "text": "Valid text"},
                ],
            },
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Check that empty string content was converted
        user_msg = params["messages"][0]
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(user_msg["content"][0]["text"], "(empty)")

        # Check that empty text in list content was converted
        assistant_msg = params["messages"][1]
        self.assertIsInstance(assistant_msg["content"], list)
        self.assertEqual(assistant_msg["content"][0]["text"], "(empty)")
        self.assertEqual(assistant_msg["content"][1]["text"], "Valid text")

    def test_complex_message_content_preserved(self):
        """Test that complex message structures (text + image) are properly converted to AWS Bedrock format."""
        # Create a complex message with both text and image content
        # Use a valid base64 string for the image
        complex_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    },
                },
                {"type": "text", "text": "Please describe it in detail."},
            ],
        }

        messages = [
            complex_message,
            {"role": "assistant", "content": "I can see the image clearly."},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Verify complex message structure is preserved and converted
        self.assertEqual(len(params["messages"]), 2)
        user_msg = params["messages"][0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(len(user_msg["content"]), 3)

        # Note: AWS Bedrock adapter reorders single images to come before text, like Anthropic
        # Check image part (should be moved to first position and converted from image_url to image)
        self.assertIn("image", user_msg["content"][0])
        self.assertEqual(user_msg["content"][0]["image"]["format"], "jpeg")
        self.assertIn("source", user_msg["content"][0]["image"])
        self.assertIn("bytes", user_msg["content"][0]["image"]["source"])

        # Check first text part (moved to second position)
        self.assertEqual(user_msg["content"][1]["text"], "What do you see in this image?")

        # Check second text part (moved to third position)
        self.assertEqual(user_msg["content"][2]["text"], "Please describe it in detail.")

    def test_multiple_system_instructions_handling(self):
        """Test that first system instruction is extracted, later ones converted to user messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "Remember to be concise."},  # Later system message
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # System instruction should be extracted from first message (in AWS Bedrock format)
        self.assertIsInstance(params["system"], list)
        self.assertEqual(len(params["system"]), 1)
        self.assertEqual(params["system"][0]["text"], "You are a helpful assistant.")

        # Should have 3 messages remaining (system message was removed, later system converted to user)
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"][0]["text"], "Hello")
        self.assertEqual(params["messages"][1]["role"], "assistant")
        self.assertEqual(params["messages"][1]["content"][0]["text"], "Hi there!")

        # Later system message should be converted to user role
        self.assertEqual(params["messages"][2]["role"], "user")
        self.assertEqual(params["messages"][2]["content"][0]["text"], "Remember to be concise.")

    def test_single_system_message_handling(self):
        """Test that a single system message is extracted as system parameter and no messages remain."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # System should be extracted (in AWS Bedrock format)
        self.assertIsInstance(params["system"], list)
        self.assertEqual(len(params["system"]), 1)
        self.assertEqual(params["system"][0]["text"], "You are a helpful assistant.")

        # No messages should remain after system extraction
        self.assertEqual(len(params["messages"]), 0)


if __name__ == "__main__":
    unittest.main()
