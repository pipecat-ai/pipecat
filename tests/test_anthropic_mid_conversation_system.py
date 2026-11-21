#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests for mid-conversation system messages in AnthropicLLMAdapter.

Current Claude models (Opus 5.5/5/4.8, Sonnet 5.5, Haiku 5.5, Fable 5/5.1,
Mythos 5/5.1) accept {"role": "system"} messages inside the messages array
when the message follows a user message and is last or followed by an
assistant turn. With mid_conversation_system enabled, the adapter keeps such
messages in the system role (mapping developer to system); otherwise it keeps
the historical user-role conversion.
"""

import unittest

from pipecat.adapters.llm.services.anthropic_adapter import AnthropicLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext, LLMStandardMessage


class TestAnthropicMidConversationSystem(unittest.IsolatedAsyncioTestCase):
    """Mid-conversation system message conversion in the Anthropic adapter."""

    def setUp(self):
        self.adapter = AnthropicLLMAdapter()

    async def test_system_message_after_user_kept_as_system(self):
        """A later system message after a user message (and last) keeps the system role."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Remember to be concise."},
        ]

        context = LLMContext(messages=messages)
        params = await self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, mid_conversation_system=True
        )

        # Initial system instruction is extracted as usual
        self.assertEqual(params["system"], "You are a helpful assistant.")

        # The mid-conversation system message follows a user message and is
        # last, so it is sent as an Anthropic system message instead of being
        # folded into the user turn.
        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(params["messages"][0]["content"], "Hello")
        self.assertEqual(params["messages"][1]["role"], "system")
        self.assertEqual(params["messages"][1]["content"], "Remember to be concise.")

    async def test_system_message_between_user_and_assistant_kept_as_system(self):
        """A system message following a user message and followed by an assistant turn is kept."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Remember to be concise."},
            {"role": "assistant", "content": "Hi there!"},
        ]

        context = LLMContext(messages=messages)
        params = await self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, mid_conversation_system=True
        )

        self.assertEqual([m["role"] for m in params["messages"]], ["user", "system", "assistant"])
        self.assertEqual(params["messages"][1]["content"], "Remember to be concise.")

    async def test_system_message_after_assistant_still_converted_to_user(self):
        """A system message that does not follow a user message keeps the user-role fallback."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "system", "content": "Remember to be concise."},
        ]

        context = LLMContext(messages=messages)
        params = await self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, mid_conversation_system=True
        )

        self.assertEqual([m["role"] for m in params["messages"]], ["user", "assistant", "user"])
        self.assertEqual(params["messages"][2]["content"], "Remember to be concise.")

    async def test_consecutive_system_messages_after_user_kept_and_merged(self):
        """A run of consecutive system messages after a user message is kept and merged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "First instruction."},
            {"role": "developer", "content": "Second instruction."},
        ]

        context = LLMContext(messages=messages)
        params = await self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, mid_conversation_system=True
        )

        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][0]["role"], "user")
        system_message = params["messages"][1]
        self.assertEqual(system_message["role"], "system")
        self.assertEqual(
            [block["text"] for block in system_message["content"]],
            ["First instruction.", "Second instruction."],
        )

    async def test_developer_message_after_user_kept_as_system(self):
        """A later developer message after a user message maps to the Anthropic system role."""
        messages: list[LLMStandardMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "developer", "content": "More instructions"},
        ]

        context = LLMContext(messages=messages)
        params = await self.adapter.get_llm_invocation_params(
            context, enable_prompt_caching=False, mid_conversation_system=True
        )

        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][1]["role"], "system")
        self.assertEqual(params["messages"][1]["content"], "More instructions")

    async def test_system_message_after_user_converted_without_mid_conversation_support(self):
        """Without mid-conversation support (the default), a later system message still becomes user."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Remember to be concise."},
        ]

        context = LLMContext(messages=messages)
        params = await self.adapter.get_llm_invocation_params(context, enable_prompt_caching=False)

        self.assertEqual(len(params["messages"]), 1)
        self.assertEqual(params["messages"][0]["role"], "user")
        self.assertEqual(
            params["messages"][0]["content"],
            [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "Remember to be concise."},
            ],
        )
