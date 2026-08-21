#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Mistral LLM adapter message transforms."""

import unittest

from pipecat.adapters.services.mistral_adapter import MistralLLMAdapter


def _transform(messages):
    """Convenience wrapper — adapter is stateless for ``_transform_messages``."""
    adapter = MistralLLMAdapter()
    return adapter._transform_messages(messages)


class TestMistralAdapterStep1TrailingTool(unittest.TestCase):
    """Step 1: ensure mid-conversation tool messages are followed by an
    assistant; trailing tool messages are left as-is (Mistral accepts
    them and generates the next assistant turn from scratch)."""

    def test_trailing_tool_no_placeholder_inserted(self):
        # Trailing ``tool`` is accepted by Mistral on its own; no need
        # to pad with an empty assistant. Padding + step 3's prefix=True
        # tells Mistral to "continue from prefix ' '" → training-data
        # junk on current models.
        messages = [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        out = _transform(messages)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[-1]["role"], "tool")
        roles = [m["role"] for m in out]
        # No assistant added after the trailing tool.
        self.assertNotIn("assistant", roles[-1:])

    def test_mid_conversation_tool_followed_by_user_gets_placeholder(self):
        # Mistral rejects a mid-conversation ``tool`` not followed by an
        # assistant — pad with the minimal placeholder so the request
        # validates. This is the original, intended use of step 1.
        messages = [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "thanks"},
        ]
        out = _transform(messages)
        # The placeholder is inserted between the tool and the user.
        roles = [m["role"] for m in out]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant", "user"])
        inserted = out[3]
        self.assertEqual(inserted, {"role": "assistant", "content": " "})

    def test_mid_conversation_tool_followed_by_assistant_no_insert(self):
        # If the conversation already has an assistant after the tool,
        # step 1 leaves it alone (no double-insert).
        messages = [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "assistant", "content": "it's sunny today"},
            {"role": "user", "content": "thanks"},
        ]
        out = _transform(messages)
        # No new message inserted; input shape preserved.
        self.assertEqual(len(out), 5)
        roles = [m["role"] for m in out]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant", "user"])


class TestMistralAdapterStep3TrailingAssistantPrefix(unittest.TestCase):
    """Step 3: set ``prefix=True`` on a trailing assistant only when it
    has real content to continue from. Empty / whitespace-only content
    is left alone."""

    def test_trailing_assistant_with_real_content_gets_prefix(self):
        messages = [
            {"role": "user", "content": "tell me a story"},
            {"role": "assistant", "content": "Once upon a time"},
        ]
        out = _transform(messages)
        self.assertTrue(out[-1].get("prefix"))
        self.assertEqual(out[-1]["content"], "Once upon a time")

    def test_trailing_assistant_with_empty_string_no_prefix(self):
        # Empty content + prefix=True asks Mistral to continue from
        # nothing → training-data junk on current models. Skip.
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": ""},
        ]
        out = _transform(messages)
        self.assertNotIn("prefix", out[-1])

    def test_trailing_assistant_with_whitespace_only_no_prefix(self):
        # Whitespace-only is the placeholder shape inserted by step 1
        # in legacy behaviour — must not trigger prefix=True.
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "   "},
        ]
        out = _transform(messages)
        self.assertNotIn("prefix", out[-1])

    def test_trailing_assistant_with_none_content_no_prefix(self):
        # A tool-call-only assistant has ``content=None``. Setting
        # prefix=True is meaningless — there's nothing to continue.
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ],
            },
        ]
        out = _transform(messages)
        self.assertNotIn("prefix", out[-1])

    def test_existing_prefix_field_not_overwritten(self):
        # If the caller set ``prefix`` explicitly, leave it alone.
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "partial", "prefix": False},
        ]
        out = _transform(messages)
        self.assertEqual(out[-1]["prefix"], False)


class TestMistralAdapterStep2SystemConversion(unittest.TestCase):
    """Sanity: step 2 (non-initial system → user) is unchanged by this
    PR."""

    def test_initial_system_block_preserved(self):
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
        out = _transform(messages)
        self.assertEqual([m["role"] for m in out], ["system", "system", "user"])

    def test_non_initial_system_converted_to_user(self):
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "now answer in french"},
            {"role": "user", "content": "comment ça va?"},
        ]
        out = _transform(messages)
        self.assertEqual([m["role"] for m in out], ["system", "user", "user", "user"])


class TestMistralAdapterEmptyMessages(unittest.TestCase):
    def test_empty_messages_passthrough(self):
        self.assertEqual(_transform([]), [])


if __name__ == "__main__":
    unittest.main()
