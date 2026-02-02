#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mixin providing context summarization capabilities for LLM services.

This module provides reusable functionality for automatically compressing conversation
context when token limits are approached, enabling efficient long-running conversations.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext

DEFAULT_SUMMARIZATION_PROMPT = """You are summarizing a conversation between a user and an AI assistant.

Your task:
1. Create a concise summary that preserves:
   - Key facts, decisions, and agreements
   - Important context needed to continue the conversation
   - User preferences and requirements mentioned
   - Any unresolved questions or action items

2. Format:
   - Use clear, factual statements
   - Group related information
   - Prioritize information likely to be referenced later
   - Keep it under 200 words

3. Omit:
   - Greetings and small talk
   - Redundant information
   - Tangential discussions that were resolved

The conversation transcript follows. Generate only the summary, no other text."""


@dataclass
class ContextSummarizationConfig:
    """Configuration for context summarization.

    Attributes:
        max_tokens: Maximum tokens before compression
        summarization_threshold: Trigger at this % of max (0.5 = 50%)
        min_messages_after_summary: Recent messages to keep uncompressed
        summarization_prompt: Custom prompt (uses DEFAULT_SUMMARIZATION_PROMPT if None)
    """

    max_tokens: int = 8000
    summarization_threshold: float = 0.5
    min_messages_after_summary: int = 6
    summarization_prompt: Optional[str] = None

    @property
    def summary_prompt(self) -> str:
        """Get the summarization prompt to use."""
        return self.summarization_prompt or DEFAULT_SUMMARIZATION_PROMPT


@dataclass
class MessagesToSummarize:
    """Result of get_messages_to_summarize operation.

    Attributes:
        messages: Messages to include in the summary
        last_summarized_index: Index of the last message being summarized
    """

    messages: List[dict]
    last_summarized_index: int


class ContextSummarizationMixin:
    """Mixin providing context summarization capabilities.

    This mixin can be used by both aggregators and LLM services to provide
    consistent context summarization behavior.
    """

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using word count heuristic.

        This is a rough estimate: word_count * 1.3
        Services can override this method to use more accurate tokenizers.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        words = re.split(r"[^\w]+", text)
        return int(len(words) * 1.3)

    def estimate_context_tokens(self, context: LLMContext) -> int:
        """Estimate total token count for a context.

        Args:
            context: LLM context to estimate

        Returns:
            Estimated total token count
        """
        total = 0

        for message in context.messages:
            # Role and structure overhead
            total += 10

            # Message content
            content = message.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total += self._estimate_tokens(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Images are expensive, rough estimate
                            total += 100

            # Tool calls
            if "tool_calls" in message:
                tool_calls = message["tool_calls"]
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            func = tool_call.get("function", {})
                            if isinstance(func, dict):
                                total += self._estimate_tokens(
                                    func.get("name", "") + func.get("arguments", "")
                                )

            # Tool call ID
            if "tool_call_id" in message:
                total += 10

        return total

    def _get_function_calls_in_progress_index(self, messages: List[dict], start_idx: int) -> int:
        """Check if there are function calls without results starting from an index.

        Args:
            messages: List of messages to check
            start_idx: Index to start checking from

        Returns:
            Index of first message with function call in progress, or -1 if none
        """
        # Track tool call IDs mapped to their message index
        pending_tool_calls: dict[str, int] = {}

        for i in range(start_idx, len(messages)):
            msg = messages[i]
            role = msg.get("role")

            # Check for tool calls in assistant messages
            if role == "assistant" and "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id")
                            if tool_call_id:
                                pending_tool_calls[tool_call_id] = i

            # Check for tool results
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id in pending_tool_calls:
                    pending_tool_calls.pop(tool_call_id)

        # If we have pending tool calls, return the earliest index
        if pending_tool_calls:
            return min(pending_tool_calls.values())

        return -1

    def get_messages_to_summarize(
        self, context: LLMContext, min_messages_to_keep: int
    ) -> MessagesToSummarize:
        """Determine which messages to summarize.

        This method identifies which messages should be included in the summary while:
        - Preserving the first system message
        - Keeping the last N messages uncompressed
        - Avoiding summarization of function calls that are in progress

        Returns:
            MessagesToSummarize with messages to summarize and last index
        """
        messages = context.messages
        if len(messages) <= min_messages_to_keep:
            return MessagesToSummarize(messages=[], last_summarized_index=-1)

        # Find first system message
        first_system_index = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                first_system_index = i
                break

        # Get messages to keep (last N messages)
        keep_start_index = len(messages) - min_messages_to_keep

        # Messages to summarize are between first system and recent messages
        # We exclude the first system message itself
        if first_system_index >= 0:
            summary_start = first_system_index + 1
        else:
            summary_start = 0

        summary_end = keep_start_index

        if summary_start >= summary_end:
            return MessagesToSummarize(messages=[], last_summarized_index=-1)

        # Check for function calls in progress in the range we want to summarize
        original_summary_end = summary_end
        function_call_start = self._get_function_calls_in_progress_index(messages, summary_start)
        if function_call_start >= 0 and function_call_start < summary_end:
            # Stop summarization before the function call
            logger.debug(
                f"ContextSummarization: Found function call in progress at index {function_call_start}, "
                f"stopping summary before it (was going to summarize up to {summary_end})"
            )
            summary_end = function_call_start

            # Count how many messages we're skipping
            skipped_messages = original_summary_end - summary_end
            if skipped_messages > 0:
                logger.info(
                    f"ContextSummarization: Skipping {skipped_messages} messages with "
                    f"function calls in progress (will summarize after results are available)"
                )

        if summary_start >= summary_end:
            return MessagesToSummarize(messages=[], last_summarized_index=-1)

        messages_to_summarize = messages[summary_start:summary_end]
        last_summarized_index = summary_end - 1

        return MessagesToSummarize(
            messages=messages_to_summarize, last_summarized_index=last_summarized_index
        )

    def format_messages_for_summary(self, messages: List[dict]) -> str:
        """Format messages as a transcript for summarization.

        Args:
            messages: Messages to format

        Returns:
            Formatted transcript string
        """
        transcript_parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle different content types
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                text = " ".join(text_parts)
            else:
                text = str(content)

            if text:
                # Capitalize role for readability
                formatted_role = role.upper()
                transcript_parts.append(f"{formatted_role}: {text}")

            # Include tool calls if present
            if "tool_calls" in msg:
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            func = tool_call.get("function", {})
                            if isinstance(func, dict):
                                name = func.get("name", "unknown")
                                args = func.get("arguments", "")
                                transcript_parts.append(f"TOOL_CALL: {name}({args})")

            # Include tool results
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                transcript_parts.append(f"TOOL_RESULT[{tool_call_id}]: {text}")

        return "\n\n".join(transcript_parts)
