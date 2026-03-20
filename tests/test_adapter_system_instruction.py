#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for system_instruction and developer message handling in LLM adapters.

Tests cover:

1. system_instruction only (no system/developer in context)
2. Initial "system" message only (no system_instruction)
3. Initial "developer" message only (no system_instruction) -> promoted to system instruction
4. Both system_instruction and initial "system" message -> warns
5. Both system_instruction and initial "developer" message -> does NOT warn; developer becomes "user"
6. Non-OpenAI adapters: subsequent "developer" messages converted to "user"
7. Non-OpenAI adapters: initial "system" discarded when system_instruction provided
8. Gemini: non-initial "system" message is converted to "user" (not extracted)
9. Single system-only message: converted to "user" instead of extracting (empty list prevention)
"""

import unittest
from unittest.mock import patch

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext, LLMStandardMessage


class TestOpenAIAdapterSystemInstruction(unittest.TestCase):
    """Tests for the OpenAI ChatCompletion adapter."""

    def setUp(self):
        self.adapter = OpenAILLMAdapter()

    def test_system_instruction_only(self):
        """system_instruction alone is prepended as a system message."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

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
        params = self.adapter.get_llm_invocation_params(context)

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
                context, system_instruction="Be concise."
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
                context, system_instruction="Be concise."
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
            self.adapter.get_llm_invocation_params(context, system_instruction="Be concise.")
            self.adapter.get_llm_invocation_params(context, system_instruction="Be concise.")
            mock_logger.warning.assert_called_once()


class TestAnthropicAdapterSystemInstruction(unittest.TestCase):
    """Tests for the Anthropic adapter."""

    def setUp(self):
        from pipecat.adapters.services.anthropic_adapter import AnthropicLLMAdapter

        self.adapter = AnthropicLLMAdapter()

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

    def test_initial_system_message_only(self):
        """Initial system message is extracted as the system parameter."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        self.assertEqual(params["system"], "You are helpful.")
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")

    def test_initial_developer_message_promoted(self):
        """Initial developer message without system_instruction is promoted to system."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        self.assertEqual(params["system"], "Extra context.")
        self.assertEqual(len(params["messages"]), 1)

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

    def test_single_system_message_becomes_user(self):
        """A lone system message is converted to user (not extracted) to prevent empty history."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        from anthropic import NOT_GIVEN

        self.assertEqual(params["system"], NOT_GIVEN)
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")


class TestBedrockAdapterSystemInstruction(unittest.TestCase):
    """Tests for the AWS Bedrock adapter."""

    def setUp(self):
        from pipecat.adapters.services.bedrock_adapter import AWSBedrockLLMAdapter

        self.adapter = AWSBedrockLLMAdapter()

    def test_system_instruction_only(self):
        """system_instruction alone becomes the system parameter."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context, system_instruction="Be helpful.")

        self.assertEqual(params["system"], [{"text": "Be helpful."}])

    def test_initial_system_message_only(self):
        """Initial system message is extracted as the system parameter."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system"], [{"text": "You are helpful."}])
        self.assertEqual(len(params["messages"]), 1)

    def test_initial_developer_message_promoted(self):
        """Initial developer message without system_instruction is promoted."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system"], [{"text": "Extra context."}])

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

    def test_single_system_message_becomes_user(self):
        """A lone system message is converted to user to prevent empty history."""
        messages: list[LLMStandardMessage] = [
            {"role": "system", "content": "You are helpful."},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertIsNone(params["system"])
        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")


class TestGeminiAdapterSystemInstruction(unittest.TestCase):
    """Tests for the Gemini adapter."""

    def setUp(self):
        from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter

        self.adapter = GeminiLLMAdapter()

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

    def test_initial_developer_message_promoted(self):
        """Initial developer message without system_instruction is promoted."""
        messages: list[LLMStandardMessage] = [
            {"role": "developer", "content": "Extra context."},
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        params = self.adapter.get_llm_invocation_params(context)

        self.assertEqual(params["system_instruction"], "Extra context.")

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
        content, role = self.adapter._extract_initial_system_or_developer(
            messages, system_instruction=None
        )

        self.assertEqual(content, "Be helpful.")
        self.assertEqual(role, "system")
        self.assertEqual(len(messages), 1)  # popped

    def test_extract_developer_without_system_instruction(self):
        """Developer message is extracted when no system_instruction."""
        messages = [
            {"role": "developer", "content": "Context."},
            {"role": "user", "content": "Hello"},
        ]
        content, role = self.adapter._extract_initial_system_or_developer(
            messages, system_instruction=None
        )

        self.assertEqual(content, "Context.")
        self.assertEqual(role, "developer")
        self.assertEqual(len(messages), 1)

    def test_developer_with_system_instruction_converts_to_user(self):
        """Developer message with system_instruction is converted to user, not extracted."""
        messages = [
            {"role": "developer", "content": "Context."},
            {"role": "user", "content": "Hello"},
        ]
        content, role = self.adapter._extract_initial_system_or_developer(
            messages, system_instruction="Be helpful."
        )

        self.assertIsNone(content)
        self.assertIsNone(role)
        self.assertEqual(len(messages), 2)  # not popped
        self.assertEqual(messages[0]["role"], "user")  # converted to user

    def test_single_system_message_becomes_user(self):
        """Single system message is converted to user instead of extracting (empty prevention)."""
        messages = [
            {"role": "system", "content": "Be helpful."},
        ]
        content, role = self.adapter._extract_initial_system_or_developer(
            messages, system_instruction=None
        )

        self.assertIsNone(content)
        self.assertIsNone(role)
        self.assertEqual(len(messages), 1)  # not popped
        self.assertEqual(messages[0]["role"], "user")

    def test_non_system_message_ignored(self):
        """Non-system/developer first message is ignored."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        content, role = self.adapter._extract_initial_system_or_developer(
            messages, system_instruction=None
        )

        self.assertIsNone(content)
        self.assertIsNone(role)
        self.assertEqual(len(messages), 1)

    def test_empty_messages(self):
        """Empty messages list returns None."""
        messages = []
        content, role = self.adapter._extract_initial_system_or_developer(
            messages, system_instruction=None
        )

        self.assertIsNone(content)
        self.assertIsNone(role)

    def test_resolve_both_system_discard(self):
        """Resolve with discard=True: system_instruction wins, warns."""
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            result = self.adapter._resolve_system_instruction(
                "from context", "system", "from settings", discard_context_system=True
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(result, "from settings")

    def test_resolve_both_system_keep(self):
        """Resolve with discard=False: warns but returns system_instruction."""
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            result = self.adapter._resolve_system_instruction(
                "from context", "system", "from settings", discard_context_system=False
            )
            mock_logger.warning.assert_called_once()

        self.assertEqual(result, "from settings")

    def test_resolve_only_system_instruction(self):
        """Only system_instruction: returns it, no warning."""
        with patch("pipecat.adapters.base_llm_adapter.logger") as mock_logger:
            result = self.adapter._resolve_system_instruction(
                None, None, "from settings", discard_context_system=True
            )
            mock_logger.warning.assert_not_called()

        self.assertEqual(result, "from settings")

    def test_resolve_only_context_system_discard(self):
        """Only context system (discard=True): returns it."""
        result = self.adapter._resolve_system_instruction(
            "from context", "system", None, discard_context_system=True
        )

        self.assertEqual(result, "from context")

    def test_resolve_only_context_system_keep(self):
        """Only context system (discard=False): returns None (already in messages)."""
        result = self.adapter._resolve_system_instruction(
            "from context", "system", None, discard_context_system=False
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
