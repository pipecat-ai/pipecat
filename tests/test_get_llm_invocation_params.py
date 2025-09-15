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
2. LLMSpecificMessage objects with llm='openai' are included and their content extracted
3. LLMSpecificMessage objects with llm != 'openai' are filtered out
4. Complex message structures (like multi-part content) are preserved
5. System instructions are preserved throughout messages at any position

For Gemini adapter:
1. LLMStandardMessage objects are converted to Gemini Content format
2. LLMSpecificMessage objects with llm='google' are included unchanged
3. LLMSpecificMessage objects with llm != 'google' are filtered out
4. Complex message structures (image, audio, multi-text) are converted to appropriate Gemini format
5. System messages are extracted as system_instruction (without duplication)
6. Single system instruction is converted to user message when no other messages exist
7. Multiple system instructions: first extracted, later ones converted to user messages
"""

import unittest

from google.genai.types import Content, Part
from openai.types.chat import ChatCompletionMessage

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

    def test_gemini_specific_messages_included(self):
        """Test that LLMSpecificMessage objects with llm='google' are included unchanged."""
        # Create a Gemini-specific message
        gemini_message = Content(role="user", parts=[Part(text="Gemini specific message")])

        # Create a mix of standard and Gemini-specific messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            LLMSpecificMessage(llm="google", message=gemini_message),
            {"role": "assistant", "content": "Standard response"},
        ]

        # Create context with these messages
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # Verify system instruction
        self.assertEqual(params["system_instruction"], "You are a helpful assistant.")

        # Verify messages (2 total: gemini-specific user + converted model)
        self.assertEqual(len(params["messages"]), 2)

        # First message should be the Gemini-specific one (unchanged)
        self.assertEqual(params["messages"][0], gemini_message)
        self.assertEqual(params["messages"][0].parts[0].text, "Gemini specific message")

        # Second message should be converted standard message
        self.assertEqual(params["messages"][1].role, "model")
        self.assertEqual(params["messages"][1].parts[0].text, "Standard response")

    def test_non_gemini_specific_messages_filtered_out(self):
        """Test that LLMSpecificMessage objects with llm != 'google' are filtered out."""
        # Create messages with different LLM-specific ones
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            LLMSpecificMessage(
                llm="openai", message={"role": "user", "content": "OpenAI specific message"}
            ),
            LLMSpecificMessage(
                llm="anthropic", message={"role": "user", "content": "Anthropic specific message"}
            ),
            {"role": "user", "content": "Standard user message"},
            LLMSpecificMessage(
                llm="google",
                message=Content(role="model", parts=[Part(text="Gemini specific response")]),
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


if __name__ == "__main__":
    unittest.main()
