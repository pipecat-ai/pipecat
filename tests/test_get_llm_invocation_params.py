#
# Copyright (c) 2024-2026, Daily
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
5. system_instruction is prepended as a system message, with conflict warnings
6. Developer messages pass through when convert_developer_to_user is False
7. Developer messages are converted to user when convert_developer_to_user is True

For Gemini adapter:
1. LLMStandardMessage objects are converted to Gemini Content format
2. LLMSpecificMessage objects with llm='google' are included and others are filtered out
3. Complex message structures (image, audio, multi-text) are converted to appropriate Gemini format
4. System messages are extracted as system_instruction (without duplication)
5. Single system instruction is converted to user message when no other messages exist
6. Multiple system instructions: first extracted, later ones converted to user messages
7. system_instruction overrides context system message, with conflict warnings
8. Developer messages are converted to user
9. Malformed messages raise LLMContextConversionError (preserving the underlying cause)

For Anthropic adapter:
1. LLMStandardMessage objects are converted to Anthropic MessageParam format
2. LLMSpecificMessage objects with llm='anthropic' are included and others are filtered out
3. Complex message structures (image, multi-text) are converted to appropriate Anthropic format
4. System messages: first extracted as system parameter, later ones converted to user messages
5. Consecutive messages with same role are merged into multi-content-block messages
6. Empty text content is converted to "(empty)"
7. system_instruction overrides context system message, with conflict warnings
8. Developer messages are converted to user
9. Malformed messages raise LLMContextConversionError (preserving the underlying cause)

For AWS Bedrock adapter:
1. LLMStandardMessage objects are converted to AWS Bedrock format
2. LLMSpecificMessage objects with llm='aws' are included and others are filtered out
3. Complex message structures (image, multi-text) are converted to appropriate AWS Bedrock format
4. System messages: first extracted as system parameter, later ones converted to user messages
5. Consecutive messages with same role are merged into multi-content-block messages
6. Empty text content is converted to "(empty)"
7. system_instruction overrides context system message, with conflict warnings
8. Developer messages are converted to user
9. Malformed messages raise LLMContextConversionError (preserving the underlying cause)

For OpenAI Responses adapter:
1. LLMContext messages are converted to Responses API input items
2. System and developer role messages are converted to developer role
3. Assistant tool_calls produce function_call input items
4. Tool messages produce function_call_output input items
5. Multimodal content conversion (text -> input_text, image_url -> input_image)
6. Tools schema flattening (nested function dict -> flat format)
7. system_instruction sets instructions (or becomes developer message if input is empty)
8. Developer messages pass through as developer role without triggering warnings

For BaseLLMAdapter helpers:
1. _extract_initial_system: system extraction and conversion logic
2. _resolve_system_instruction: conflict resolution between context and settings
"""

import unittest
from unittest.mock import patch

from google.genai.types import Content, FunctionCall, FunctionResponse, Part

from pipecat.adapters.base_llm_adapter import LLMContextConversionError
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.adapters.services.aws_nova_sonic_adapter import AWSNovaSonicLLMAdapter
from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.adapters.services.grok_realtime_adapter import GrokRealtimeLLMAdapter
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter
from pipecat.adapters.services.open_ai_responses_adapter import OpenAIResponsesLLMAdapter
from pipecat.adapters.services.perplexity_adapter import PerplexityLLMAdapter
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMStandardMessage,
)


class TestOpenAIGetLLMInvocationParams(unittest.TestCase):
    # In production, BaseOpenAILLMService always passes convert_developer_to_user
    # to the adapter (True or False depending on the service's supports_developer_role).
    # Tests below use False to simulate native OpenAI usage, except for the
    # developer-conversion-specific tests which use True.

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
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=False)

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
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=False)

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
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=False)

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
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=False)

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

    def test_system_instruction_only(self):
        """system_instruction alone is prepended as a system message."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(
            context, system_instruction="Be helpful.", convert_developer_to_user=False
        )

        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "Be helpful.")
        self.assertEqual(params["messages"][1]["role"], "user")

    def test_initial_system_message_only(self):
        """Initial system message without system_instruction passes through."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=False)

        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "You are helpful.")

    def test_both_system_instruction_and_system_message_warns(self):
        """system_instruction + initial system message warns but allows both."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise.", convert_developer_to_user=False
            )
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("may be unintended", warning_msg)

        # Both are present: prepended system_instruction + original system message
        self.assertEqual(params["messages"][0]["content"], "Be concise.")
        self.assertEqual(params["messages"][1]["content"], "You are helpful.")

    def test_both_system_instruction_and_developer_message_no_warning(self):
        """system_instruction + initial developer message does NOT warn."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise.", convert_developer_to_user=False
            )
            mock_logger.warning.assert_not_called()

        # system_instruction prepended, developer message stays in messages
        self.assertEqual(params["messages"][0]["content"], "Be concise.")
        self.assertEqual(params["messages"][1]["role"], "developer")

    def test_warning_fires_only_once(self):
        """Conflict warning fires only once per adapter instance."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise.", convert_developer_to_user=False
            )
            self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise.", convert_developer_to_user=False
            )
            mock_logger.warning.assert_called_once()

    def test_developer_messages_converted_to_user(self):
        """Developer messages are converted to user role when convert_developer_to_user is True."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Extra context.")

    def test_developer_conversion_does_not_affect_other_roles(self):
        """convert_developer_to_user only affects developer messages, not system/user/assistant."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "System prompt."},
            {"role": "developer", "content": "Dev guidance."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][1]["role"], "user")
        self.assertEqual(params["messages"][1]["content"], "Dev guidance.")
        self.assertEqual(params["messages"][2]["role"], "user")
        self.assertEqual(params["messages"][3]["role"], "assistant")


class TestGeminiGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = GeminiLLMAdapter()

    def test_malformed_message_raises_conversion_error(self):
        """Test that a malformed message raises LLMContextConversionError, preserving the underlying cause."""
        # A data URL with no comma has no base64 payload, so the adapter's
        # url.split(",")[1] raises IndexError during conversion.
        malformed_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64"}},
            ],
        }
        context = LLMContext(messages=[malformed_message])

        with self.assertRaises(LLMContextConversionError) as ctx:
            self.adapter.get_llm_invocation_params(context)

        # The underlying cause is preserved for debugging.
        self.assertIsInstance(ctx.exception.__cause__, IndexError)
        self.assertIn("Error mapping context messages to provider format", str(ctx.exception))

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

        # When there's only one message, it's converted to user in-place (not extracted)
        # so system_instruction is None
        self.assertIsNone(params["system_instruction"])

        # The system message should be converted to a user message
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

    def test_system_instruction_only(self):
        """system_instruction alone becomes the system_instruction parameter."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

        self.assertEqual(params["system_instruction"], "Be helpful.")

    def test_initial_system_message_only(self):
        """Initial system message is extracted as system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "You are helpful.")
        self.assertEqual(len(params["messages"]), 1)

    def test_initial_developer_message_becomes_user(self):
        """Initial developer message without system_instruction becomes user, not system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertIsNone(params["system_instruction"])
        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0].role, "user")
        self.assertEqual(params["messages"][0].parts[0].text, "Extra context.")

    def test_both_system_instruction_and_system_message_warns(self):
        """system_instruction + initial system message warns and uses system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_both_system_instruction_and_developer_message_no_warning(self):
        """system_instruction + initial developer message: no warning, developer becomes user."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_non_initial_system_message_not_extracted(self):
        """Non-initial system message is converted to user, not extracted as system instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Late system message"},
            {"role": "user", "content": "How are you?"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        # No system instruction should be extracted from non-initial position
        self.assertIsNone(params["system_instruction"])
        # The system message should have been converted to user role in the Gemini Content
        # (we check that 3 messages are present, meaning no extraction happened)
        self.assertEqual(len(params["messages"]), 3)

    def test_subsequent_developer_messages_converted_to_user(self):
        """Subsequent developer messages are converted to user role."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "developer", "content": "More instructions"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["messages"]), 2)
        # Second message (developer) should be converted to user in Google format
        self.assertEqual(params["messages"][1].role, "user")

    # --- _merge_parallel_tool_calls_for_thinking ---
    #
    # With thinking enabled, a batch of parallel tool calls arrives split across
    # separate messages (each call, then its response). The adapter regroups
    # both sides so the model turn's call count matches the following user
    # turn's response count.

    def _tool_call_message(self, call_id, *, signature=None):
        """Build a model Content with a single function_call Part."""
        part = Part(
            function_call=FunctionCall(id=call_id, name=f"tool_{call_id}", args={"q": call_id})
        )
        if signature:
            part.thought_signature = signature
        return Content(role="model", parts=[part])

    def _tool_response_message(self, call_id):
        """Build a user Content with a single function_response Part."""
        return Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        id=call_id, name=f"tool_{call_id}", response={"ok": True}
                    )
                )
            ],
        )

    def _thought_signature_dict(self, call_id):
        """Build a thought_signature dict bookmarked to a function call."""
        return {"signature": f"sig-{call_id}", "bookmark": {"function_call": call_id}}

    def test_merge_parallel_interleaved_calls_and_responses(self):
        """Parallel calls interleaved with responses merge into one model and one user turn."""
        messages = [
            self._tool_call_message("c1", signature="sig-c1"),
            self._tool_response_message("c1"),
            self._tool_call_message("c2"),
            self._tool_response_message("c2"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(
            [self._thought_signature_dict("c1")], messages
        )

        self.assertEqual([m.role for m in result], ["model", "user"])
        self.assertEqual([p.function_call.id for p in result[0].parts], ["c1", "c2"])
        self.assertEqual([p.function_response.id for p in result[1].parts], ["c1", "c2"])

    def test_merge_batch_ordered_calls_and_responses(self):
        """Calls grouped first, then responses, still merge into matching turns."""
        messages = [
            self._tool_call_message("c1", signature="sig-c1"),
            self._tool_call_message("c2"),
            self._tool_response_message("c1"),
            self._tool_response_message("c2"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(
            [self._thought_signature_dict("c1")], messages
        )

        self.assertEqual([m.role for m in result], ["model", "user"])
        self.assertEqual([p.function_call.id for p in result[0].parts], ["c1", "c2"])
        self.assertEqual([p.function_response.id for p in result[1].parts], ["c1", "c2"])

    def test_merge_three_parallel_calls(self):
        """Three parallel calls merge into a single model turn and user turn."""
        messages = [
            self._tool_call_message("c1", signature="sig-c1"),
            self._tool_response_message("c1"),
            self._tool_call_message("c2"),
            self._tool_response_message("c2"),
            self._tool_call_message("c3"),
            self._tool_response_message("c3"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(
            [self._thought_signature_dict("c1")], messages
        )

        self.assertEqual([m.role for m in result], ["model", "user"])
        self.assertEqual(len(result[0].parts), 3)
        self.assertEqual(len(result[1].parts), 3)

    def test_merge_single_call_unchanged(self):
        """A lone signed call and its response pass through as-is."""
        messages = [
            self._tool_call_message("c1", signature="sig-c1"),
            self._tool_response_message("c1"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(
            [self._thought_signature_dict("c1")], messages
        )

        self.assertEqual([m.role for m in result], ["model", "user"])
        self.assertEqual(len(result[0].parts), 1)
        self.assertEqual(len(result[1].parts), 1)

    def test_merge_sequential_calls_with_signatures_not_merged(self):
        """Sequential calls (each with its own signature) stay in separate turns."""
        messages = [
            self._tool_call_message("c1", signature="sig-c1"),
            self._tool_response_message("c1"),
            self._tool_call_message("c2", signature="sig-c2"),
            self._tool_response_message("c2"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(
            [self._thought_signature_dict("c1"), self._thought_signature_dict("c2")], messages
        )

        self.assertEqual([m.role for m in result], ["model", "user", "model", "user"])
        self.assertTrue(all(len(m.parts) == 1 for m in result))

    def test_merge_mixed_parallel_and_sequential_groups(self):
        """A parallel group followed by a sequential call are grouped independently."""
        messages = [
            self._tool_call_message("c1", signature="sig-c1"),
            self._tool_response_message("c1"),
            self._tool_call_message("c2"),
            self._tool_response_message("c2"),
            self._tool_call_message("c3", signature="sig-c3"),
            self._tool_response_message("c3"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(
            [self._thought_signature_dict("c1"), self._thought_signature_dict("c3")], messages
        )

        self.assertEqual([m.role for m in result], ["model", "user", "model", "user"])
        self.assertEqual([len(m.parts) for m in result], [2, 2, 1, 1])

    def test_merge_collects_interleaved_non_tool_message(self):
        """A non-tool message inside a group is collected and re-emitted after it."""
        text_message = Content(role="model", parts=[Part(text="thinking out loud")])
        messages = [
            self._tool_call_message("c1", signature="sig-c1"),
            self._tool_response_message("c1"),
            text_message,
            self._tool_call_message("c2"),
            self._tool_response_message("c2"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(
            [self._thought_signature_dict("c1")], messages
        )

        # Both calls still merge (and both responses), and the interleaved text
        # is preserved after the merged group rather than stopping the merge.
        self.assertEqual([m.role for m in result], ["model", "user", "model"])
        self.assertEqual([p.function_call.id for p in result[0].parts], ["c1", "c2"])
        self.assertEqual([p.function_response.id for p in result[1].parts], ["c1", "c2"])
        self.assertEqual(result[2].parts[0].text, "thinking out loud")

    def test_merge_no_thought_signatures_unchanged(self):
        """Without any function-call thought signatures, messages pass through unchanged."""
        messages = [
            self._tool_call_message("c1"),
            self._tool_response_message("c1"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking([], messages)

        self.assertEqual(len(result), 2)

    def test_merge_empty_messages(self):
        """An empty message list returns empty."""
        self.assertEqual(self.adapter._merge_parallel_tool_calls_for_thinking([], []), [])


class TestAnthropicGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = AnthropicLLMAdapter()

    def test_malformed_message_raises_conversion_error(self):
        """Test that a malformed message raises LLMContextConversionError, preserving the underlying cause."""
        # A data URL with no comma has no base64 payload, so the adapter's
        # url.split(",")[1] raises IndexError during conversion.
        malformed_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64"}},
            ],
        }
        context = LLMContext(messages=[malformed_message])

        with self.assertRaises(LLMContextConversionError) as ctx:
            self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # The underlying cause is preserved for debugging.
        self.assertIsInstance(ctx.exception.__cause__, IndexError)
        self.assertIn("Error mapping context messages to provider format", str(ctx.exception))

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

    def test_system_instruction_only(self):
        """system_instruction alone becomes the system parameter."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, system_instruction="Be helpful."
        )

        self.assertEqual(params["system"], "Be helpful.")
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")

    def test_initial_developer_message_becomes_user(self):
        """Initial developer message without system_instruction becomes user, not system."""
        from anthropic import NOT_GIVEN

        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        self.assertEqual(params["system"], NOT_GIVEN)
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Extra context.")

    def test_both_system_instruction_and_system_message_warns(self):
        """system_instruction + initial system message warns and uses system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context,
                enable_prompt_caching=False,
                system_instruction="Be concise.",
            )
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("Using system_instruction", warning_msg)

        self.assertEqual(params["system"], "Be concise.")

    def test_both_system_instruction_and_developer_message_no_warning(self):
        """system_instruction + initial developer message: no warning, developer becomes user."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context,
                enable_prompt_caching=False,
                system_instruction="Be concise.",
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["system"], "Be concise.")
        # Developer message should have been converted to "user"
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Extra context.")

    def test_subsequent_developer_messages_converted_to_user(self):
        """Subsequent developer messages are converted to user role."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "developer", "content": "More instructions"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        # Developer message was converted to "user"
        self.assertEqual(params["messages"][2]["role"], "user")
        self.assertEqual(params["messages"][2]["content"], "More instructions")

    def test_initial_system_discarded_when_system_instruction_provided(self):
        """Initial system message is discarded when system_instruction is provided."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "Old instruction."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger"):
            params = self.adapter.get_llm_invocation_params(
                context,
                enable_prompt_caching=False,
                system_instruction="New instruction.",
            )

        self.assertEqual(params["system"], "New instruction.")
        # Only the user message should remain
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")

    def test_ensure_last_message_is_user_appends_when_trailing_assistant(self):
        """ensure_last_message_is_user=True appends a user message after a trailing assistant."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, ensure_last_message_is_user=True
        )
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][-1]["role"], "user")
        self.assertEqual(params["messages"][-1]["content"], [{"type": "text", "text": "."}])

    def test_ensure_last_message_is_user_off_keeps_trailing_assistant(self):
        """Without the flag (default), a trailing assistant message is preserved."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)
        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][-1]["role"], "assistant")

    def test_ensure_last_message_is_user_noop_when_trailing_user(self):
        """ensure_last_message_is_user=True does nothing when the list already ends with a user."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, ensure_last_message_is_user=True
        )
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][-1]["role"], "user")
        self.assertEqual(params["messages"][-1]["content"], "How are you?")

    def test_ensure_last_message_is_user_handles_empty_list(self):
        """ensure_last_message_is_user=True handles an empty context."""
        params = self.adapter.get_llm_invocation_params(
            LLMContext(), enable_prompt_caching=False, ensure_last_message_is_user=True
        )
        self.assertEqual(len(params["messages"]), 0)

    def test_ensure_last_message_is_user_noop_when_tool_result_trailing(self):
        """ensure_last_message_is_user=True does nothing when a tool result (user role) trails."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "get_weather", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "content": "Sunny, 22°C", "tool_call_id": "t1"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, ensure_last_message_is_user=True
        )
        self.assertEqual(params["messages"][-1]["role"], "user")
        self.assertEqual(params["messages"][-1]["content"][0]["type"], "tool_result")


class TestAWSBedrockGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = AWSBedrockLLMAdapter()

    def test_malformed_message_raises_conversion_error(self):
        """Test that a malformed message raises LLMContextConversionError, preserving the underlying cause."""
        # A data URL with no comma has no base64 payload, so the adapter's
        # url.split(",")[1] raises IndexError during conversion.
        malformed_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64"}},
            ],
        }
        context = LLMContext(messages=[malformed_message])

        with self.assertRaises(LLMContextConversionError) as ctx:
            self.adapter.get_llm_invocation_params(context)

        # The underlying cause is preserved for debugging.
        self.assertIsInstance(ctx.exception.__cause__, IndexError)
        self.assertIn("Error mapping context messages to provider format", str(ctx.exception))

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
        """Test that a single system message is converted to user role when no other messages exist."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        # Create context
        context = LLMContext(messages=messages)

        # Get invocation params
        params = self.adapter.get_llm_invocation_params(context)

        # When there's only one message, it's converted to user in-place (not extracted)
        # so system is None
        self.assertIsNone(params["system"])

        # Single system message should be converted to user role
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(
            params["messages"][0]["content"][0]["text"], "You are a helpful assistant."
        )

    def test_system_instruction_only(self):
        """system_instruction alone becomes the system parameter."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

        self.assertEqual(params["system"], [{"text": "Be helpful."}])

    def test_initial_developer_message_becomes_user(self):
        """Initial developer message without system_instruction becomes user, not system."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertIsNone(params["system"])
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"][0]["text"], "Extra context.")

    def test_both_system_instruction_and_system_message_warns(self):
        """system_instruction + initial system message warns and uses system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(params["system"], [{"text": "Be concise."}])

    def test_both_system_instruction_and_developer_message_no_warning(self):
        """system_instruction + initial developer message: no warning, developer becomes user."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["system"], [{"text": "Be concise."}])
        self.assertEqual(params["messages"][0]["role"], "user")

    def test_subsequent_developer_messages_converted_to_user(self):
        """Subsequent developer messages are converted to user role."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "developer", "content": "More instructions"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["messages"][2]["role"], "user")

    def test_ensure_last_message_is_user_appends_when_trailing_assistant(self):
        """ensure_last_message_is_user=True appends a user message after a trailing assistant."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context, ensure_last_message_is_user=True)
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][-1]["role"], "user")
        self.assertEqual(params["messages"][-1]["content"], [{"text": "."}])

    def test_ensure_last_message_is_user_off_keeps_trailing_assistant(self):
        """Without the flag (default), a trailing assistant message is preserved."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context)
        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][-1]["role"], "assistant")

    def test_ensure_last_message_is_user_noop_when_trailing_user(self):
        """ensure_last_message_is_user=True does nothing when the list already ends with a user."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context, ensure_last_message_is_user=True)
        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][-1]["role"], "user")

    def test_ensure_last_message_is_user_handles_empty_list(self):
        """ensure_last_message_is_user=True handles an empty context."""
        params = self.adapter.get_llm_invocation_params(
            LLMContext(), ensure_last_message_is_user=True
        )
        self.assertEqual(len(params["messages"]), 0)

    def test_ensure_last_message_is_user_noop_when_tool_result_trailing(self):
        """ensure_last_message_is_user=True does nothing when a tool result (user role) trails."""
        context = LLMContext(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "get_weather", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "content": "Sunny, 22°C", "tool_call_id": "t1"},
            ]
        )
        params = self.adapter.get_llm_invocation_params(context, ensure_last_message_is_user=True)
        self.assertEqual(params["messages"][-1]["role"], "user")
        self.assertIn("toolResult", params["messages"][-1]["content"][0])


class TestPerplexityGetLLMInvocationParams(unittest.TestCase):
    # Perplexity doesn't support the "developer" role, so PerplexityLLMService
    # sets supports_developer_role = False. Tests below pass
    # convert_developer_to_user=True to match production behavior.

    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = PerplexityLLMAdapter()

    def test_standard_messages_pass_through(self):
        """Test that a valid [user, assistant, user] sequence passes through unchanged."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Hello")
        self.assertEqual(params["messages"][1]["role"], "assistant")
        self.assertEqual(params["messages"][1]["content"], "Hi there!")
        self.assertEqual(params["messages"][2]["role"], "user")
        self.assertEqual(params["messages"][2]["content"], "How are you?")

    def test_initial_system_message_preserved(self):
        """Test that a valid [system, user, assistant, user] sequence passes through unchanged."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Bye"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(len(params["messages"]), 4)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")
        self.assertEqual(params["messages"][1]["role"], "user")
        self.assertEqual(params["messages"][2]["role"], "assistant")
        self.assertEqual(params["messages"][3]["role"], "user")

    def test_consecutive_same_role_messages_merged(self):
        """Test that consecutive user messages are merged into list-of-dicts content."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "First message"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Third message"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(len(params["messages"]), 3)

        # First message should be merged users
        merged = params["messages"][0]
        self.assertEqual(merged["role"], "user")
        self.assertIsInstance(merged["content"], list)
        self.assertEqual(len(merged["content"]), 2)
        self.assertEqual(merged["content"][0]["type"], "text")
        self.assertEqual(merged["content"][0]["text"], "First message")
        self.assertEqual(merged["content"][1]["type"], "text")
        self.assertEqual(merged["content"][1]["text"], "Second message")

        self.assertEqual(params["messages"][1]["role"], "assistant")
        self.assertEqual(params["messages"][2]["role"], "user")

    def test_non_initial_system_converted_to_user(self):
        """Test that non-initial system messages are converted to user and merged with adjacent user."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Tell me about Python."},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        # system(initial), user, assistant, merged(system→user + user)
        self.assertEqual(len(params["messages"]), 4)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][1]["role"], "user")
        self.assertEqual(params["messages"][2]["role"], "assistant")

        # The converted system→user and the following user should be merged
        merged = params["messages"][3]
        self.assertEqual(merged["role"], "user")
        self.assertIsInstance(merged["content"], list)
        self.assertEqual(len(merged["content"]), 2)
        self.assertEqual(merged["content"][0]["text"], "Be concise.")
        self.assertEqual(merged["content"][1]["text"], "Tell me about Python.")

    def test_multiple_system_messages_at_start_preserved(self):
        """Test that multiple consecutive system messages at start pass through unchanged."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Always be polite."},
            {"role": "user", "content": "Hello"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(len(params["messages"]), 3)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "You are a helpful assistant.")
        self.assertEqual(params["messages"][1]["role"], "system")
        self.assertEqual(params["messages"][1]["content"], "Always be polite.")
        self.assertEqual(params["messages"][2]["role"], "user")
        self.assertEqual(params["messages"][2]["content"], "Hello")

    def test_trailing_assistant_removed(self):
        """Test that a trailing assistant message is removed."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Hello")

    def test_only_system_messages_preserved(self):
        """Test that system-only contexts are left unchanged (no system→user conversion).

        We intentionally do not convert trailing system messages to "user"
        because that would make the transformation unstable across calls —
        Perplexity has statefulness within a conversation, so a message that
        was "user" in one call but becomes "system" in the next causes errors.
        """
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "system")

    def test_system_exposed_after_trailing_assistant_removed(self):
        """Test that a system message exposed by trailing assistant removal stays system.

        It's important that initial system messages are never converted to
        "user", because Perplexity has statefulness within a conversation — if
        a message was sent as "system" in one call and then becomes "user" in a
        later call (after more messages are appended), the API rejects it.
        """
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Sure thing."},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        # Trailing assistant removed → [system], system stays as-is
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "system")
        self.assertEqual(params["messages"][0]["content"], "You are helpful.")

    def test_consecutive_assistants_merged_then_trailing_removed(self):
        """Test that consecutive assistant messages are merged, then trailing assistant is removed."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "First response"},
            {"role": "assistant", "content": "Second response"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        # After merging assistants we get [user, assistant(merged)], then trailing
        # assistant is removed, leaving just [user]
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Hello")

    def test_tool_messages_preserved(self):
        """Test that tool messages pass through without modification."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"id": "1", "function": {"name": "get_weather", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "Sunny, 72F", "tool_call_id": "1"},
            {"role": "user", "content": "Thanks!"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(len(params["messages"]), 4)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][1]["role"], "assistant")
        self.assertEqual(params["messages"][2]["role"], "tool")
        self.assertEqual(params["messages"][2]["content"], "Sunny, 72F")
        self.assertEqual(params["messages"][3]["role"], "user")

    def test_developer_message_converted_to_user(self):
        """Developer messages are converted to user role."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Extra context.")

    def test_developer_message_merged_with_adjacent_user(self):
        """Developer→user conversion merges with adjacent user messages."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]

        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        # developer→user merged with following user
        self.assertEqual(len(params["messages"]), 3)
        merged = params["messages"][0]
        self.assertEqual(merged["role"], "user")
        self.assertIsInstance(merged["content"], list)
        self.assertEqual(len(merged["content"]), 2)
        self.assertEqual(merged["content"][0]["text"], "Be concise.")
        self.assertEqual(merged["content"][1]["text"], "Hello")

    def test_empty_messages(self):
        """Test that empty messages list returns empty."""
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context, convert_developer_to_user=True)

        self.assertEqual(params["messages"], [])


class TestOpenAIResponsesGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = OpenAIResponsesLLMAdapter()

    def test_simple_user_assistant_messages(self):
        """Simple user/assistant text messages are converted correctly."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 2)
        self.assertEqual(params["input"][0], {"role": "user", "content": "Hello"})
        self.assertEqual(params["input"][1], {"role": "assistant", "content": "Hi there!"})

    def test_system_role_converted_to_developer(self):
        """System role messages are converted to developer role."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["input"][0]["role"], "developer")
        self.assertEqual(params["input"][0]["content"], "You are helpful.")

    def test_developer_role_kept_as_developer(self):
        """Developer role messages are kept as developer role."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["input"][0]["role"], "developer")
        self.assertEqual(params["input"][0]["content"], "Extra context.")

    def test_system_message_without_system_instruction_no_warning(self):
        """System message without system_instruction does not trigger a warning."""
        adapter = OpenAIResponsesLLMAdapter()
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            adapter.get_llm_invocation_params(context)
            mock_logger.warning.assert_not_called()

    def test_system_message_with_system_instruction_triggers_warning(self):
        """System message + system_instruction triggers a conflict warning."""
        adapter = OpenAIResponsesLLMAdapter()
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            adapter.get_llm_invocation_params(context, system_instruction="Be concise.")
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            self.assertIn("system_instruction", warning_msg)

    def test_developer_message_with_system_instruction_no_warning(self):
        """Developer message + system_instruction does NOT trigger a warning."""
        adapter = OpenAIResponsesLLMAdapter()
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = adapter.get_llm_invocation_params(context, system_instruction="Be concise.")
            mock_logger.warning.assert_not_called()

        # Developer message stays as developer, system_instruction becomes instructions
        self.assertEqual(params["input"][0]["role"], "developer")
        self.assertEqual(params["instructions"], "Be concise.")

    def test_non_initial_system_message_no_warning(self):
        """Non-initial system messages are converted without a warning."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "New instruction"},
        ]
        context = LLMContext(messages=messages)

        adapter = OpenAIResponsesLLMAdapter()
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["input"][1]["role"], "developer")
        self.assertEqual(params["input"][1]["content"], "New instruction")

    def test_conflict_warning_fires_only_once(self):
        """The conflict warning fires only once per adapter instance."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        adapter = OpenAIResponsesLLMAdapter()
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            adapter.get_llm_invocation_params(context, system_instruction="Be concise.")
            adapter.get_llm_invocation_params(context, system_instruction="Be concise.")
            mock_logger.warning.assert_called_once()

    def test_assistant_tool_calls_to_function_call(self):
        """Assistant messages with tool_calls produce function_call input items."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                        },
                        "type": "function",
                    }
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 1)
        fc = params["input"][0]
        self.assertEqual(fc["type"], "function_call")
        self.assertEqual(fc["call_id"], "call_123")
        self.assertEqual(fc["name"], "get_weather")
        self.assertEqual(fc["arguments"], '{"location": "SF"}')

    def test_multiple_tool_calls(self):
        """Multiple tool calls in one assistant message produce multiple function_call items."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                        "type": "function",
                    },
                    {
                        "id": "call_2",
                        "function": {
                            "name": "get_restaurant",
                            "arguments": '{"location": "SF"}',
                        },
                        "type": "function",
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 2)
        self.assertEqual(params["input"][0]["name"], "get_weather")
        self.assertEqual(params["input"][1]["name"], "get_restaurant")

    def test_tool_message_to_function_call_output(self):
        """Tool role messages produce function_call_output input items."""
        messages = [
            {
                "role": "tool",
                "content": '{"temperature": "72"}',
                "tool_call_id": "call_123",
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 1)
        fco = params["input"][0]
        self.assertEqual(fco["type"], "function_call_output")
        self.assertEqual(fco["call_id"], "call_123")
        self.assertEqual(fco["output"], '{"temperature": "72"}')

    def test_mixed_conversation(self):
        """Mixed conversation with text + function calls converts correctly."""
        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
                        "type": "function",
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"temp": "72"}',
                "tool_call_id": "call_abc",
            },
            {"role": "assistant", "content": "It's 72 degrees in SF."},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 4)
        self.assertEqual(params["input"][0]["role"], "user")
        self.assertEqual(params["input"][1]["type"], "function_call")
        self.assertEqual(params["input"][2]["type"], "function_call_output")
        self.assertEqual(params["input"][3]["role"], "assistant")

    def test_multimodal_text_conversion(self):
        """Multimodal text content parts are converted to input_text."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        content = params["input"][0]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[0]["text"], "What's in this image?")

    def test_multimodal_image_conversion(self):
        """Multimodal image_url content parts are converted to input_image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,abc123"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        content = params["input"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["image_url"], "data:image/jpeg;base64,abc123")
        self.assertEqual(content[1]["detail"], "auto")

    def test_multimodal_image_with_detail(self):
        """Image content parts preserve the detail setting when provided."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png", "detail": "high"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        content = params["input"][0]["content"]
        self.assertEqual(content[0]["detail"], "high")

    def test_tools_schema_flattening(self):
        """Tools schema with nested function dict is flattened to Responses API format."""
        weather_fn = FunctionSchema(
            name="get_weather",
            description="Get the current weather",
            properties={
                "location": {"type": "string", "description": "The city"},
            },
            required=["location"],
        )
        tools = ToolsSchema(standard_tools=[weather_fn])
        context = LLMContext(tools=tools)
        params = self.adapter.get_llm_invocation_params(context)

        tool_list = params["tools"]
        self.assertEqual(len(tool_list), 1)
        tool = tool_list[0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["name"], "get_weather")
        self.assertEqual(tool["description"], "Get the current weather")
        self.assertIn("properties", tool["parameters"])

    def test_empty_messages(self):
        """Empty messages list produces empty input list."""
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context)
        self.assertEqual(params["input"], [])

    def test_llm_specific_message_passthrough(self):
        """LLMSpecificMessage with llm='openai_responses' passes through."""
        specific_msg = self.adapter.create_llm_specific_message(
            {"type": "function_call", "call_id": "x", "name": "foo", "arguments": "{}"}
        )
        messages = [
            {"role": "user", "content": "Hello"},
            specific_msg,
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(len(params["input"]), 2)
        self.assertEqual(params["input"][0]["role"], "user")
        self.assertEqual(params["input"][1]["type"], "function_call")

    def test_reasoning_message_becomes_reasoning_item(self):
        """A persisted reasoning message converts to a Responses reasoning item."""
        reasoning_msg = self.adapter.create_llm_specific_message(
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Let me think."}],
                "encrypted_content": "ENCRYPTED",
            }
        )
        context = LLMContext(messages=[reasoning_msg])
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(
            params["input"][0],
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Let me think."}],
                "encrypted_content": "ENCRYPTED",
            },
        )

    def test_reasoning_precedes_assistant_message(self):
        """Reasoning round-trips positioned before the assistant reply it produced."""
        reasoning_msg = self.adapter.create_llm_specific_message(
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENCRYPTED"}
        )
        context = LLMContext(messages=[reasoning_msg, {"role": "assistant", "content": "Hello!"}])
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["input"][0]["type"], "reasoning")
        self.assertEqual(params["input"][1]["role"], "assistant")

    def test_reasoning_without_encrypted_content_omits_field(self):
        """encrypted_content is optional; omit it rather than send null."""
        reasoning_msg = self.adapter.create_llm_specific_message(
            {"type": "reasoning", "id": "rs_2", "summary": []}
        )
        context = LLMContext(messages=[reasoning_msg])
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["input"][0]["type"], "reasoning")
        self.assertEqual(params["input"][0]["id"], "rs_2")
        self.assertNotIn("encrypted_content", params["input"][0])

    def test_id_for_llm_specific_messages(self):
        """Adapter identifier is 'openai_responses'."""
        self.assertEqual(self.adapter.id_for_llm_specific_messages, "openai_responses")

    def test_system_instruction_with_messages_sets_instructions(self):
        """When system_instruction is provided and input is non-empty, sets instructions."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

        self.assertEqual(params["instructions"], "Be helpful.")
        self.assertEqual(len(params["input"]), 1)
        self.assertEqual(params["input"][0]["role"], "user")

    def test_system_instruction_with_empty_input_becomes_developer_message(self):
        """When system_instruction is provided but input is empty, it becomes a developer message."""
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

        self.assertNotIn("instructions", params)
        self.assertEqual(len(params["input"]), 1)
        self.assertEqual(params["input"][0]["role"], "developer")
        self.assertEqual(params["input"][0]["content"], "Be helpful.")

    def test_no_system_instruction_omits_instructions(self):
        """When no system_instruction is provided, instructions key is absent."""
        context = LLMContext(messages=[{"role": "user", "content": "Hi"}])
        params = self.adapter.get_llm_invocation_params(context)

        self.assertNotIn("instructions", params)


class TestOpenAIRealtimeGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = OpenAIRealtimeLLMAdapter()

    def test_system_message_extracted_as_instruction(self):
        """Initial system message is extracted as system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "You are helpful.")
        self.assertEqual(len(params["messages"]), 1)

    def test_developer_message_becomes_user(self):
        """Developer message is converted to user, not extracted as system instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertIsNone(params["system_instruction"])
        # Developer converted to user, then packed with the other user message
        self.assertEqual(len(params["messages"]), 1)

    def test_subsequent_developer_message_becomes_user(self):
        """Non-initial developer message is converted to user."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "developer", "content": "Extra context."},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "You are helpful.")
        # Developer message converted to user
        self.assertEqual(len(params["messages"]), 1)

    def test_empty_messages(self):
        """Empty messages list returns empty."""
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["messages"], [])
        self.assertIsNone(params["system_instruction"])

    def test_both_system_instruction_and_system_message_warns(self):
        """system_instruction + initial system message warns and uses system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_both_system_instruction_and_developer_message_no_warning(self):
        """system_instruction + initial developer message: no warning, developer becomes user."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_system_instruction_only(self):
        """system_instruction without context system message returns system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be concise.")

        self.assertEqual(params["system_instruction"], "Be concise.")


class TestGrokRealtimeGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = GrokRealtimeLLMAdapter()

    def test_system_message_extracted_as_instruction(self):
        """Initial system message is extracted as system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "You are helpful.")
        self.assertEqual(len(params["messages"]), 1)

    def test_developer_message_becomes_user(self):
        """Developer message is converted to user, not extracted as system instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertIsNone(params["system_instruction"])
        # Developer converted to user, then packed with the other user message
        self.assertEqual(len(params["messages"]), 1)

    def test_subsequent_developer_message_becomes_user(self):
        """Non-initial developer message is converted to user."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "developer", "content": "Extra context."},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "You are helpful.")
        self.assertEqual(len(params["messages"]), 1)

    def test_empty_messages(self):
        """Empty messages list returns empty."""
        context = LLMContext(messages=[])
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["messages"], [])
        self.assertIsNone(params["system_instruction"])

    def test_both_system_instruction_and_system_message_warns(self):
        """system_instruction + initial system message warns and uses system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_both_system_instruction_and_developer_message_no_warning(self):
        """system_instruction + initial developer message: no warning, developer becomes user."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_system_instruction_only(self):
        """system_instruction without context system message returns system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be concise.")

        self.assertEqual(params["system_instruction"], "Be concise.")


class TestAWSNovaSonicGetLLMInvocationParams(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AWSNovaSonicLLMAdapter()

    def test_system_message_extracted_as_instruction(self):
        """Initial system message is extracted as system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "You are helpful.")
        self.assertEqual(len(params["messages"]), 1)

    def test_developer_message_becomes_user(self):
        """Developer message is converted to user, not extracted as system instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertIsNone(params["system_instruction"])
        # Both messages should be present (developer as user, plus the real user)
        self.assertEqual(len(params["messages"]), 2)

    def test_subsequent_developer_message_becomes_user(self):
        """Non-initial developer message is converted to user."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "developer", "content": "Extra context."},
            {"role": "assistant", "content": "Hi"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "You are helpful.")
        # Developer becomes user, plus assistant
        self.assertEqual(len(params["messages"]), 2)

    def test_both_system_instruction_and_system_message_warns(self):
        """system_instruction + initial system message warns and uses system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_both_system_instruction_and_developer_message_no_warning(self):
        """system_instruction + initial developer message: no warning, developer becomes user."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)

        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            params = self.adapter.get_llm_invocation_params(
                context, system_instruction="Be concise."
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(params["system_instruction"], "Be concise.")

    def test_system_instruction_only(self):
        """system_instruction without context system message returns system_instruction."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be concise.")

        self.assertEqual(params["system_instruction"], "Be concise.")


class TestBaseLLMAdapterHelpers(unittest.TestCase):
    """Tests for the shared helper methods on BaseLLMAdapter."""

    def setUp(self):
        # Use OpenAILLMAdapter as a concrete implementation for testing the base helpers
        self.adapter = OpenAILLMAdapter()

    def test_extract_system_message(self):
        """System message is extracted from messages[0]."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
        content = self.adapter._extract_initial_system(messages, system_instruction=None)

        self.assertEqual(content, "Be helpful.")
        self.assertEqual(len(messages), 1)  # popped

    def test_extract_developer_not_extracted(self):
        """Developer message is not extracted by _extract_initial_system."""
        messages = [
            {"role": "developer", "content": "Context."},
            {"role": "user", "content": "Hello"},
        ]
        content = self.adapter._extract_initial_system(messages, system_instruction=None)

        self.assertIsNone(content)
        self.assertEqual(len(messages), 2)  # not popped
        self.assertEqual(messages[0]["role"], "developer")  # unchanged

    def test_developer_with_system_instruction_not_extracted(self):
        """Developer message with system_instruction is not handled by _extract_initial_system."""
        messages = [
            {"role": "developer", "content": "Context."},
            {"role": "user", "content": "Hello"},
        ]
        content = self.adapter._extract_initial_system(messages, system_instruction="Be helpful.")

        self.assertIsNone(content)
        self.assertEqual(len(messages), 2)  # not popped
        self.assertEqual(messages[0]["role"], "developer")  # unchanged by helper

    def test_single_system_message_becomes_user(self):
        """Single system message is converted to user instead of extracting (empty prevention)."""
        messages = [
            {"role": "system", "content": "Be helpful."},
        ]
        content = self.adapter._extract_initial_system(messages, system_instruction=None)

        self.assertIsNone(content)
        self.assertEqual(len(messages), 1)  # not popped
        self.assertEqual(messages[0]["role"], "user")

    def test_single_system_message_with_system_instruction_warns(self):
        """Single system message + system_instruction still warns even though content isn't extracted."""
        messages = [
            {"role": "system", "content": "Be helpful."},
        ]

        adapter = OpenAILLMAdapter()
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            content = adapter._extract_initial_system(messages, system_instruction="Be concise.")
            mock_logger.warning.assert_called_once()

        self.assertIsNone(content)
        self.assertEqual(messages[0]["role"], "user")

    def test_non_system_message_ignored(self):
        """Non-system/developer first message is ignored."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        content = self.adapter._extract_initial_system(messages, system_instruction=None)

        self.assertIsNone(content)
        self.assertEqual(len(messages), 1)

    def test_empty_messages(self):
        """Empty messages list returns None."""
        messages = []
        content = self.adapter._extract_initial_system(messages, system_instruction=None)

        self.assertIsNone(content)

    def test_resolve_both_system_discard(self):
        """Resolve with discard=True: system_instruction wins, warns."""
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            result = self.adapter._resolve_system_instruction(
                "from context", "from settings", discard_context_system=True
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(result, "from settings")

    def test_resolve_both_system_keep(self):
        """Resolve with discard=False: warns but returns system_instruction."""
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            result = self.adapter._resolve_system_instruction(
                "from context", "from settings", discard_context_system=False
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(result, "from settings")

    def test_resolve_only_system_instruction(self):
        """Only system_instruction: returns it, no warning."""
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            result = self.adapter._resolve_system_instruction(
                None, "from settings", discard_context_system=True
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(result, "from settings")

    def test_resolve_only_context_system_discard(self):
        """Only context system (discard=True): returns it."""
        result = self.adapter._resolve_system_instruction(
            "from context", None, discard_context_system=True
        )

        self.assertEqual(result, "from context")

    def test_resolve_only_context_system_keep(self):
        """Only context system (discard=False): returns None (already in messages)."""
        result = self.adapter._resolve_system_instruction(
            "from context", None, discard_context_system=False
        )

        self.assertIsNone(result)


class TestTrailingUserMessageInjection(unittest.TestCase):
    """Model gating for the no-prefill trailing-user-message injection.

    Claude 4.6+ models reject requests whose message list ends with an
    assistant message. Injection must default to ON for any model not known
    to support prefill (so future models are covered) and stay OFF for the
    frozen legacy set that still supports prefill.
    """

    def _anthropic(self, **settings):
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key="test-key", settings=AnthropicLLMService.Settings(**settings)
        )

    def _bedrock(self, **settings):
        from pipecat.services.aws.llm import AWSBedrockLLMService

        return AWSBedrockLLMService(
            aws_region="us-east-1", settings=AWSBedrockLLMService.Settings(**settings)
        )

    def test_anthropic_no_prefill_models_inject(self):
        """Models without prefill support — including unknown future ones — inject."""
        for model in (
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-8",
            "claude-sonnet-5",  # hypothetical future model: must default to inject
        ):
            service = self._anthropic(model=model)
            self.assertTrue(service._should_inject_trailing_user_message(), model)

    def test_anthropic_legacy_prefill_models_do_not_inject(self):
        """Models known to support prefill are left untouched."""
        for model in (
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-3-7-sonnet-latest",
        ):
            service = self._anthropic(model=model)
            self.assertFalse(service._should_inject_trailing_user_message(), model)

    def test_prefill_patterns_overridable_by_subclass(self):
        """Deployments with exotic model naming can override the pattern set."""
        from pipecat.services.anthropic.llm import AnthropicLLMService

        class CustomService(AnthropicLLMService):
            _PREFILL_SUPPORTED_PATTERNS = ("my-claude-proxy",)

        service = CustomService(
            api_key="test-key", settings=CustomService.Settings(model="my-claude-proxy-v2")
        )
        self.assertFalse(service._should_inject_trailing_user_message())

    def test_anthropic_invocation_params_append_trailing_user(self):
        """A trailing assistant message gets a user message appended, request-only."""
        service = self._anthropic(model="claude-sonnet-4-6")
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hold on a second."},
                {"role": "assistant", "content": "◐"},
            ]
        )
        params = service._get_llm_invocation_params(context)

        self.assertEqual(params["messages"][-1]["role"], "user")
        self.assertEqual(params["messages"][-1]["content"], [{"type": "text", "text": "."}])
        # The stored context is never mutated — the fix applies to the request only.
        self.assertEqual(context.messages[-1]["role"], "assistant")

    def test_anthropic_invocation_params_untouched_when_prefill_supported(self):
        """Legacy prefill-supporting models keep the trailing assistant message."""
        service = self._anthropic(model="claude-haiku-4-5")
        context = LLMContext(
            messages=[
                {"role": "user", "content": "Hold on a second."},
                {"role": "assistant", "content": "◐"},
            ]
        )
        params = service._get_llm_invocation_params(context)

        self.assertEqual(params["messages"][-1]["role"], "assistant")

    def test_bedrock_gate_by_model(self):
        """Bedrock IDs match by substring; non-Claude models never inject."""
        cases = {
            "us.anthropic.claude-sonnet-4-6-v1:0": True,
            "anthropic.claude-opus-4-8-v1:0": True,
            "us.anthropic.claude-haiku-4-5-v1:0": False,
            "anthropic.claude-3-7-sonnet-20250219-v1:0": False,
            "amazon.nova-pro-v1:0": False,
        }
        for model, expected in cases.items():
            service = self._bedrock(model=model)
            self.assertEqual(service._should_inject_trailing_user_message(), expected, model)


if __name__ == "__main__":
    unittest.main()
