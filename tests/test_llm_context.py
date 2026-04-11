#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LLMContext core functionality."""

import unittest

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
)


class TestGetMessagesTruncateLargeValues(unittest.TestCase):
    """Tests for LLMContext.get_messages(truncate_large_values=True)."""

    # -- Standard messages: binary elision -----------------------------------

    def test_default_preserves_all_data(self):
        """truncate_large_values defaults to False, preserving all data."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages()

        self.assertEqual(
            result[0]["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        )

    def test_elides_base64_image_url(self):
        """Base64 data:image/ URLs are replaced with a placeholder."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result[0]["content"][0]["text"], "Describe this image")
        self.assertEqual(result[0]["content"][1]["image_url"]["url"], "data:image/...")

    def test_preserves_http_image_url(self):
        """HTTP image URLs are not elided (they aren't binary data)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(
            result[0]["content"][0]["image_url"]["url"],
            "https://example.com/image.jpg",
        )

    def test_elides_input_audio_data(self):
        """input_audio items have their data field elided."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Audio follows"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "UklGRiQA" * 1000, "format": "wav"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result[0]["content"][1]["input_audio"]["data"], "...")
        self.assertEqual(result[0]["content"][1]["input_audio"]["format"], "wav")

    def test_elides_audio_field(self):
        """Items with an 'audio' field are elided (used by some realtime adapters)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "audio": "UklGRiQA" * 1000},
                    {"type": "audio", "audio": "UklGRiQA" * 1000},
                ],
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result[0]["content"][0]["audio"], "...")
        self.assertEqual(result[0]["content"][1]["audio"], "...")

    def test_elides_top_level_mime_type_image(self):
        """Messages with top-level mime_type image/ have their data elided."""
        messages = [
            {
                "role": "user",
                "mime_type": "image/png",
                "data": "iVBORw0KGgoAAAANSU" * 1000,
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result[0]["data"], "...")
        self.assertEqual(result[0]["mime_type"], "image/png")

    def test_mixed_content_elides_only_binary(self):
        """In a message with text, image, and audio, only binary parts are elided."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is an image and audio"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw=="},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "UklGRiQA", "format": "wav"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result[0]["content"][0]["text"], "Here is an image and audio")
        self.assertEqual(result[0]["content"][1]["image_url"]["url"], "data:image/...")
        self.assertEqual(result[0]["content"][2]["input_audio"]["data"], "...")

    def test_text_only_messages_unchanged(self):
        """Plain text messages are completely unaffected."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result, messages)

    def test_does_not_mutate_original(self):
        """Returns copies; originals are untouched."""
        original_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": original_url},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        _ = context.get_messages(truncate_large_values=True)

        self.assertEqual(
            context.get_messages()[0]["content"][0]["image_url"]["url"],
            original_url,
        )

    def test_multiple_images_all_elided(self):
        """Multiple image_url items in the same message are all elided."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,BBBB"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/photo.jpg"},
                    },
                ],
            }
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result[0]["content"][0]["image_url"]["url"], "data:image/...")
        self.assertEqual(result[0]["content"][1]["image_url"]["url"], "data:image/...")
        self.assertEqual(
            result[0]["content"][2]["image_url"]["url"],
            "https://example.com/photo.jpg",
        )

    def test_works_with_llm_specific_filter(self):
        """truncate_large_values works together with llm_specific_filter."""
        adapter = OpenAILLMAdapter()
        std_msg = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                },
            ],
        }
        specific_msg = adapter.create_llm_specific_message(
            {"role": "assistant", "content": "response"}
        )
        context = LLMContext(messages=[std_msg, specific_msg])

        result = context.get_messages("openai", truncate_large_values=True)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["content"][0]["image_url"]["url"], "data:image/...")

    def test_string_content_with_no_binary(self):
        """Messages with string content (not list) pass through fine."""
        messages = [
            {"role": "user", "content": "Just a string"},
        ]
        context = LLMContext(messages=messages)
        result = context.get_messages(truncate_large_values=True)

        self.assertEqual(result[0]["content"], "Just a string")

    # -- LLMSpecificMessage: long-string truncation --------------------------

    def test_llm_specific_short_values_preserved(self):
        """Short string values in LLMSpecificMessage are kept as-is."""
        inner = {"type": "thought", "text": "brief thought"}
        specific_msg = LLMSpecificMessage(llm="anthropic", message=inner)
        context = LLMContext(messages=[specific_msg])

        result = context.get_messages(truncate_large_values=True)

        self.assertIsInstance(result[0], LLMSpecificMessage)
        self.assertEqual(result[0].message["type"], "thought")
        self.assertEqual(result[0].message["text"], "brief thought")

    def test_llm_specific_long_string_truncated(self):
        """Long string values in LLMSpecificMessage are truncated."""
        long_signature = "a" * 500
        inner = {"type": "thought", "text": "short", "signature": long_signature}
        specific_msg = LLMSpecificMessage(llm="anthropic", message=inner)
        context = LLMContext(messages=[specific_msg])

        result = context.get_messages(truncate_large_values=True)

        msg = result[0].message
        self.assertEqual(msg["type"], "thought")
        self.assertEqual(msg["text"], "short")
        # Signature should be truncated
        self.assertIn("...", msg["signature"])
        self.assertIn("500 chars", msg["signature"])
        self.assertTrue(len(msg["signature"]) < len(long_signature))

    def test_llm_specific_nested_dict_truncated(self):
        """Long strings nested in dicts within LLMSpecificMessage are truncated."""
        inner = {
            "type": "thought_signature",
            "signature": "x" * 200,
            "bookmark": {"text": "y" * 200},
        }
        specific_msg = LLMSpecificMessage(llm="google", message=inner)
        context = LLMContext(messages=[specific_msg])

        result = context.get_messages(truncate_large_values=True)

        msg = result[0].message
        self.assertEqual(msg["type"], "thought_signature")
        self.assertIn("...", msg["signature"])
        self.assertIn("...", msg["bookmark"]["text"])

    def test_llm_specific_list_values_truncated(self):
        """Long strings inside lists within LLMSpecificMessage are truncated."""
        inner = {"items": ["short", "a" * 200]}
        specific_msg = LLMSpecificMessage(llm="test", message=inner)
        context = LLMContext(messages=[specific_msg])

        result = context.get_messages(truncate_large_values=True)

        msg = result[0].message
        self.assertEqual(msg["items"][0], "short")
        self.assertIn("...", msg["items"][1])

    def test_llm_specific_non_string_values_preserved(self):
        """Non-string values (ints, bools, None) in LLMSpecificMessage are untouched."""
        inner = {"type": "test", "count": 42, "active": True, "extra": None}
        specific_msg = LLMSpecificMessage(llm="test", message=inner)
        context = LLMContext(messages=[specific_msg])

        result = context.get_messages(truncate_large_values=True)

        msg = result[0].message
        self.assertEqual(msg["count"], 42)
        self.assertEqual(msg["active"], True)
        self.assertIsNone(msg["extra"])

    def test_llm_specific_does_not_mutate_original(self):
        """Truncation returns a copy; original LLMSpecificMessage is untouched."""
        long_sig = "a" * 500
        inner = {"signature": long_sig}
        specific_msg = LLMSpecificMessage(llm="anthropic", message=inner)
        context = LLMContext(messages=[specific_msg])

        _ = context.get_messages(truncate_large_values=True)

        self.assertEqual(specific_msg.message["signature"], long_sig)


if __name__ == "__main__":
    unittest.main()
