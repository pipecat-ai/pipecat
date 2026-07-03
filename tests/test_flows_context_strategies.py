#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test suite for context management strategies.

This module contains tests for the context management features of Pipecat Flows,
focusing on:
- Context strategy configuration
- Strategy behavior (APPEND, RESET, RESET_WITH_SUMMARY)
- Provider-specific message formatting
- Summary generation and integration
"""

import unittest
import warnings
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from pipecat.flows.exceptions import FlowError
from pipecat.flows.manager import FlowManager
from pipecat.flows.types import ContextStrategy, ContextStrategyConfig, NodeConfig
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService


class TestContextStrategies(unittest.IsolatedAsyncioTestCase):
    """Test suite for context management strategies.

    Tests functionality including:
    - Strategy configuration and validation
    - Strategy behavior and message handling
    - Provider-specific adaptations
    - Summary generation and integration
    """

    async def asyncSetUp(self):
        """Set up test fixtures before each test."""
        self.mock_task = AsyncMock()
        self.mock_task.event_handler = Mock()
        self.mock_task.set_reached_downstream_filter = Mock()

        # Set up mock LLM with client
        self.mock_llm = OpenAILLMService(api_key="test-key")
        self.mock_llm.run_inference = AsyncMock()

        self.mock_tts = AsyncMock()

        # Create mock context aggregator with messages
        self.mock_context = MagicMock()
        self.mock_context.messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        self.mock_context.get_messages.return_value = self.mock_context.messages

        self.mock_context_aggregator = MagicMock()
        self.mock_context_aggregator.user = MagicMock()
        self.mock_context_aggregator.user.return_value = MagicMock()
        self.mock_context_aggregator.user.return_value._context = self.mock_context

        # Sample node configuration
        self.sample_node: NodeConfig = {
            "task_messages": [{"role": "developer", "content": "Test task."}],
            "functions": [],
        }

    async def test_context_strategy_config_validation(self):
        """Test ContextStrategyConfig validation."""
        # Valid configurations
        ContextStrategyConfig(strategy=ContextStrategy.APPEND)
        ContextStrategyConfig(strategy=ContextStrategy.RESET)
        ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY, summary_prompt="Summarize the conversation"
        )

        # Invalid configuration - missing prompt
        with self.assertRaises(ValueError):
            ContextStrategyConfig(strategy=ContextStrategy.RESET_WITH_SUMMARY)

    async def test_reset_with_summary_deprecation_warning(self):
        """Test that RESET_WITH_SUMMARY emits a DeprecationWarning at runtime."""
        mock_summary = "Conversation summary"
        self.mock_llm.run_inference.return_value = mock_summary

        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the conversation",
            ),
        )
        await flow_manager.initialize()

        # First node using RESET_WITH_SUMMARY should trigger the deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await flow_manager._set_node("first", self.sample_node)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertTrue(len(deprecation_warnings) >= 1)
            self.assertIn("RESET_WITH_SUMMARY is deprecated", str(deprecation_warnings[0].message))

        # Second node should NOT trigger a second warning (once-only)
        self.mock_task.queue_frames.reset_mock()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await flow_manager._set_node("second", self.sample_node)

            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            self.assertEqual(len(deprecation_warnings), 0)

    async def test_default_strategy(self):
        """Test default context strategy (APPEND)."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
        )
        await flow_manager.initialize()

        # Under the default (APPEND) strategy the first node appends, keeping any
        # context already present.
        await flow_manager._set_node("first", self.sample_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesAppendFrame) for f in first_frames))
        self.assertFalse(any(isinstance(f, LLMMessagesUpdateFrame) for f in first_frames))

        # Reset mock
        self.mock_task.queue_frames.reset_mock()

        # Subsequent node should use AppendFrame with default strategy
        await flow_manager._set_node("second", self.sample_node)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesAppendFrame) for f in second_frames))

    async def test_reset_strategy(self):
        """Test RESET strategy behavior."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        )
        await flow_manager.initialize()

        # First node should use UpdateFrame under the RESET strategy
        await flow_manager._set_node("first", self.sample_node)
        first_call = self.mock_task.queue_frames.call_args_list[0]
        first_frames = first_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in first_frames))
        self.mock_task.queue_frames.reset_mock()

        # Second node should use UpdateFrame with RESET strategy
        await flow_manager._set_node("second", self.sample_node)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames))

    async def test_reset_with_summary_success(self):
        """Test successful RESET_WITH_SUMMARY strategy."""
        # Mock successful summary generation
        mock_summary = "Conversation summary"
        self.mock_llm.run_inference.return_value = mock_summary

        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the conversation",
            ),
        )
        await flow_manager.initialize()

        # Set nodes and verify summary inclusion
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager._set_node("second", self.sample_node)

        # Verify summary was included in context update
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        update_frame = next(f for f in second_frames if isinstance(f, LLMMessagesUpdateFrame))
        self.assertTrue(any(mock_summary in str(m) for m in update_frame.messages))

    async def test_reset_with_summary_timeout(self):
        """Test RESET_WITH_SUMMARY fallback to APPEND on timeout."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the conversation",
            ),
        )
        await flow_manager.initialize()

        # Mock timeout
        self.mock_llm.run_inference.side_effect = AsyncMock(side_effect=TimeoutError)

        # Set nodes and verify fallback to APPEND
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager._set_node("second", self.sample_node)

        # Verify UpdateFrame was used (APPEND behavior)
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesAppendFrame) for f in second_frames))

    async def test_provider_specific_summary_formatting(self):
        """Test summary formatting for different LLM providers."""
        summary = "Test summary"

        # Test OpenAI format
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=OpenAILLMService(api_key="test-key"),
            context_aggregator=self.mock_context_aggregator,
        )
        openai_message = flow_manager._adapter.format_summary_message(summary)
        self.assertEqual(openai_message["role"], "developer")

        # Test Anthropic format
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=AnthropicLLMService(api_key="test-key"),
            context_aggregator=self.mock_context_aggregator,
        )
        anthropic_message = flow_manager._adapter.format_summary_message(summary)
        self.assertEqual(anthropic_message["role"], "developer")

        # Test Gemini format
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=GoogleLLMService(api_key=" "),  # dummy key (GoogleLLMService rejects empty string)
            context_aggregator=self.mock_context_aggregator,
        )
        gemini_message = flow_manager._adapter.format_summary_message(summary)
        self.assertEqual(gemini_message["role"], "developer")

    async def test_node_level_strategy_override(self):
        """Test that node-level strategy overrides global strategy."""
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        )
        await flow_manager.initialize()

        # Create node with RESET strategy
        node_with_strategy = {
            **self.sample_node,
            "context_strategy": ContextStrategyConfig(strategy=ContextStrategy.RESET),
        }

        # Set nodes and verify strategy override
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        await flow_manager._set_node("second", node_with_strategy)

        # Verify UpdateFrame was used (RESET behavior) despite global APPEND
        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]
        self.assertTrue(any(isinstance(f, LLMMessagesUpdateFrame) for f in second_frames))

    async def test_summary_generation_content(self):
        """Test that summary generation uses correct prompt and context."""
        mock_summary = "Generated summary"
        self.mock_llm.run_inference.return_value = mock_summary

        summary_prompt = "Create a detailed summary"
        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY, summary_prompt=summary_prompt
            ),
        )
        await flow_manager.initialize()

        # Set nodes to trigger summary generation
        await flow_manager._set_node("first", self.sample_node)
        await flow_manager._set_node("second", self.sample_node)

        # Verify summary generation call
        run_inference_call = self.mock_llm.run_inference.call_args
        run_inference_args = run_inference_call[0]
        run_inference_kwargs = run_inference_call[1]

        # Verify summary prompt was passed as system_instruction kwarg
        self.assertEqual(run_inference_kwargs["system_instruction"], summary_prompt)

        # Verify conversation history was included in context messages
        context = run_inference_args[0]
        self.assertTrue(
            any(
                str(self.mock_context.messages[0]["content"]) in str(m)
                for m in context.get_messages()
            )
        )

    async def test_context_structure_after_summary(self):
        """Test the structure of context after summary generation."""
        mock_summary = "Generated summary"
        self.mock_llm.run_inference.return_value = mock_summary

        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY, summary_prompt="Summarize"
            ),
        )
        await flow_manager.initialize()

        # Set nodes to trigger summary generation
        await flow_manager._set_node("first", self.sample_node)
        self.mock_task.queue_frames.reset_mock()

        # Node with new task messages
        new_node = {
            "task_messages": [{"role": "developer", "content": "New task."}],
            "functions": [],
        }
        await flow_manager._set_node("second", new_node)

        # Verify context structure
        update_call = self.mock_task.queue_frames.call_args_list[0]
        update_frames = update_call[0][0]
        messages_frame = next(f for f in update_frames if isinstance(f, LLMMessagesUpdateFrame))

        # Verify order: summary message, then new task messages
        self.assertTrue(mock_summary in str(messages_frame.messages[0]))
        self.assertEqual(
            messages_frame.messages[1]["content"], new_node["task_messages"][0]["content"]
        )

    async def test_reset_with_summary_and_role_messages(self):
        """Test that LLMUpdateSettingsFrame and summary coexist correctly."""
        mock_summary = "Conversation summary"
        self.mock_llm.run_inference.return_value = mock_summary

        flow_manager = FlowManager(
            worker=self.mock_task,
            llm=self.mock_llm,
            context_aggregator=self.mock_context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.RESET_WITH_SUMMARY,
                summary_prompt="Summarize the conversation",
            ),
        )
        await flow_manager.initialize()

        # Set first node (with role_message)
        first_node = {
            "role_message": "You are a helpful assistant.",
            "task_messages": [{"role": "developer", "content": "First task."}],
            "functions": [],
        }
        await flow_manager._set_node("first", first_node)
        self.mock_task.queue_frames.reset_mock()

        # Set second node with role_message — triggers summary + settings update
        second_node = {
            "role_message": "You are now a different assistant.",
            "task_messages": [{"role": "developer", "content": "Second task."}],
            "functions": [],
        }
        await flow_manager._set_node("second", second_node)

        second_call = self.mock_task.queue_frames.call_args_list[0]
        second_frames = second_call[0][0]

        # Verify LLMUpdateSettingsFrame is present with new system instruction
        settings_frames = [f for f in second_frames if isinstance(f, LLMUpdateSettingsFrame)]
        self.assertEqual(len(settings_frames), 1)
        self.assertEqual(
            settings_frames[0].delta.system_instruction, "You are now a different assistant."
        )

        # Verify UpdateFrame contains summary + task_messages (not role_messages)
        update_frames = [f for f in second_frames if isinstance(f, LLMMessagesUpdateFrame)]
        self.assertEqual(len(update_frames), 1)
        messages = update_frames[0].messages
        self.assertTrue(mock_summary in str(messages[0]))
        self.assertEqual(messages[1]["content"], "Second task.")

        # Verify frame ordering: LLMUpdateSettingsFrame before LLMMessagesUpdateFrame
        settings_idx = second_frames.index(settings_frames[0])
        update_idx = second_frames.index(update_frames[0])
        self.assertLess(settings_idx, update_idx)
