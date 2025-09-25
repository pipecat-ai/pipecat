#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage


class TestGetMessagesForPersistentStorage(unittest.TestCase):
    """Test suite for LLMContext.get_messages_for_persistent_storage method."""

    def test_no_system_instruction_returns_messages_as_is(self):
        """Test that without system instruction, messages are returned unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = LLMContext(messages=messages)

        result = context.get_messages_for_persistent_storage()

        self.assertEqual(result, messages)
        self.assertEqual(len(result), 2)

    def test_empty_messages_with_system_instruction_adds_system_message(self):
        """Test that system instruction is added when messages list is empty."""
        context = LLMContext()
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], system_instruction)

    def test_non_system_first_message_prepends_system_instruction(self):
        """Test that system instruction is prepended when first message is not system."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = LLMContext(messages=messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], system_instruction)
        self.assertEqual(result[1], messages[0])
        self.assertEqual(result[2], messages[1])

    def test_existing_system_message_not_duplicated(self):
        """Test that system instruction is not added when first message is already system."""
        messages = [
            {"role": "system", "content": "Existing system message"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = LLMContext(messages=messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        self.assertEqual(len(result), 3)
        self.assertEqual(result, messages)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "Existing system message")

    def test_empty_system_instruction_does_not_add_message(self):
        """Test that empty system instruction does not add a system message."""
        messages = [{"role": "user", "content": "Hello"}]
        context = LLMContext(messages=messages)

        result = context.get_messages_for_persistent_storage("")

        self.assertEqual(result, messages)
        self.assertEqual(len(result), 1)

    def test_none_system_instruction_does_not_add_message(self):
        """Test that None system instruction does not add a system message."""
        messages = [{"role": "user", "content": "Hello"}]
        context = LLMContext(messages=messages)

        result = context.get_messages_for_persistent_storage(None)

        self.assertEqual(result, messages)
        self.assertEqual(len(result), 1)

    def test_whitespace_only_system_instruction_adds_message(self):
        """Test that whitespace-only system instruction still adds a system message."""
        messages = [{"role": "user", "content": "Hello"}]
        context = LLMContext(messages=messages)
        system_instruction = "   "

        result = context.get_messages_for_persistent_storage(system_instruction)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], system_instruction)

    def test_with_llm_specific_messages(self):
        """Test that method works correctly with LLMSpecificMessage objects."""
        llm_specific = LLMSpecificMessage(
            llm="test-llm", message={"role": "user", "content": "Specific"}
        )
        messages = [{"role": "user", "content": "Standard message"}, llm_specific]
        context = LLMContext(messages=messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], system_instruction)
        self.assertEqual(result[1], messages[0])
        self.assertEqual(result[2], llm_specific)

    def test_system_message_detection_case_sensitivity(self):
        """Test that system message detection is case sensitive."""
        messages = [
            {"role": "System", "content": "Mixed case system"},  # Capital S
            {"role": "user", "content": "Hello"},
        ]
        context = LLMContext(messages=messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        # Should prepend because "System" != "system"
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], system_instruction)
        self.assertEqual(result[1], messages[0])

    def test_message_without_role_key_does_not_crash(self):
        """Test that messages without 'role' key are handled gracefully."""
        messages = [{"content": "Message without role"}, {"role": "user", "content": "Hello"}]
        context = LLMContext(messages=messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        # Should prepend system instruction since first message doesn't have role="system"
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], system_instruction)

    def test_original_messages_not_modified(self):
        """Test that the original messages list is not modified."""
        original_messages = [{"role": "user", "content": "Hello"}]
        context = LLMContext(messages=original_messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        # Original messages should remain unchanged
        self.assertEqual(len(original_messages), 1)
        self.assertEqual(original_messages[0]["role"], "user")

        # Result should have system message prepended
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1], original_messages[0])

    def test_complex_message_structure_preserved(self):
        """Test that complex message structures are preserved."""
        complex_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Complex message"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            ],
        }
        messages = [complex_message]
        context = LLMContext(messages=messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1], complex_message)
        self.assertEqual(result[1]["content"], complex_message["content"])

    def test_deep_copy_prevents_nested_mutation(self):
        """Test that deep copy prevents mutation of nested message content."""
        nested_content = {"nested": {"data": "original"}}
        complex_message = {"role": "user", "content": nested_content}
        messages = [complex_message]
        context = LLMContext(messages=messages)
        system_instruction = "You are a helpful assistant."

        result = context.get_messages_for_persistent_storage(system_instruction)

        # Modify the nested content in the result
        result[1]["content"]["nested"]["data"] = "modified"

        # Original message should remain unchanged
        self.assertEqual(complex_message["content"]["nested"]["data"], "original")
        self.assertEqual(context.get_messages()[0]["content"]["nested"]["data"], "original")


if __name__ == "__main__":
    unittest.main()
