#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM adapter for conversation-summary generation and formatting.

This module provides the LLMAdapter class used by the flow manager to:

- Format a generated summary as a context message
- Generate a summary via out-of-band LLM inference
"""

from typing import Any

from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage


class LLMAdapter:
    """Helpers for generating and formatting conversation summaries."""

    def format_summary_message(self, summary: str) -> dict:
        """Format a summary as a developer message.

        Summary messages use the LLMContextMessage format (OpenAI-style),
        as summarization triggers an LLMMessagesUpdateFrame.

        Args:
            summary: The generated summary text.

        Returns:
            A developer message containing the summary.
        """
        return {"role": "developer", "content": f"Here's a summary of the conversation:\n{summary}"}

    async def generate_summary(
        self, llm: Any, summary_prompt: str, context: LLMContext
    ) -> str | None:
        """Generate a summary by running a direct one-shot, out-of-band inference with the LLM.

        Args:
            llm: LLM service instance containing client/credentials.
            summary_prompt: Prompt text to guide summary generation.
            context: Context object containing conversation history for the summary.

        Returns:
            Generated summary text, or None if generation fails.
        """
        try:
            messages = context.get_messages()

            prompt_messages: list[LLMContextMessage] = [
                {
                    "role": "developer",
                    "content": f"Conversation history: {messages}",
                },
            ]

            summary_context = LLMContext(messages=prompt_messages)

            return await llm.run_inference(summary_context, system_instruction=summary_prompt)

        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            return None
