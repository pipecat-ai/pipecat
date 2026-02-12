#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a summarizer for managing LLM context summarization."""

import uuid
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextSummaryRequestFrame,
    LLMContextSummaryResultFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject
from pipecat.utils.context.llm_context_summarization import (
    LLMContextSummarizationConfig,
    LLMContextSummarizationUtil,
)


class LLMContextSummarizer(BaseObject):
    """Summarizer for managing LLM context summarization.

    This class manages automatic context summarization when token or message
    limits are reached. It monitors the LLM context size, triggers
    summarization requests, and applies the results to compress conversation history.

    Event handlers available:

    - on_request_summarization: Emitted when summarization should be triggered.
        The aggregator should broadcast this frame to the LLM service.

    Example::

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame: LLMContextSummaryRequestFrame):
            await aggregator.broadcast_frame(
                LLMContextSummaryRequestFrame,
                request_id=frame.request_id,
                context=frame.context,
                ...
            )
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        config: Optional[LLMContextSummarizationConfig] = None,
    ):
        """Initialize the context summarizer.

        Args:
            context: The LLM context to monitor and summarize.
            config: Configuration for summarization behavior. If None, uses default config.
        """
        super().__init__()

        self._context = context
        self._config = config or LLMContextSummarizationConfig()

        self._task_manager: Optional[BaseTaskManager] = None

        self._summarization_in_progress = False
        self._pending_summary_request_id: Optional[str] = None

        self._register_event_handler("on_request_summarization", sync=True)

    @property
    def task_manager(self) -> BaseTaskManager:
        """Returns the configured task manager."""
        if not self._task_manager:
            raise RuntimeError(f"{self} context summarizer was not properly setup")
        return self._task_manager

    async def setup(self, task_manager: BaseTaskManager):
        """Initialize the summarizer with the given task manager.

        Args:
            task_manager: The task manager to be associated with this instance.
        """
        self._task_manager = task_manager

    async def cleanup(self):
        """Cleanup the summarizer."""
        await super().cleanup()
        await self._clear_summarization_state()

    async def process_frame(self, frame: Frame):
        """Process an incoming frame to detect when summarization is needed.

        Args:
            frame: The frame to be processed.
        """
        if isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_response_start(frame)
        elif isinstance(frame, LLMContextSummaryResultFrame):
            await self._handle_summary_result(frame)
        elif isinstance(frame, InterruptionFrame):
            await self._handle_interruption()

    async def _handle_llm_response_start(self, frame: LLMFullResponseStartFrame):
        """Handle LLM response start to check if summarization is needed.

        Args:
            frame: The LLM response start frame.
        """
        if self._should_summarize():
            await self._request_summarization()

    async def _handle_interruption(self):
        """Handle interruption by canceling summarization in progress.

        Args:
            frame: The interruption frame.
        """
        # Reset summarization state to allow new requests. This is necessary because
        # the request frame (LLMContextSummaryRequestFrame) may have been cancelled
        # during interruption. We preserve _pending_summary_request_id to handle the
        # response frame (LLMContextSummaryResultFrame), which is uninterruptible and
        # will still be delivered.
        self._summarization_in_progress = False

    async def _clear_summarization_state(self):
        """Cancel pending summarization."""
        if self._summarization_in_progress:
            logger.debug(f"{self}: Clearing pending summarization")
            self._summarization_in_progress = False
            self._pending_summary_request_id = None

    def _should_summarize(self) -> bool:
        """Determine if context summarization should be triggered.

        Evaluates whether the current context has reached either the token
        threshold or message count threshold that warrants compression.

        Returns:
            True if all conditions are met:
            - No summarization currently in progress
            - AND either:
              - Token count exceeds max_context_tokens
              - OR message count exceeds max_unsummarized_messages since last summary
        """
        logger.trace(f"{self}: Checking if context summarization is needed")

        if self._summarization_in_progress:
            logger.debug(f"{self}: Summarization already in progress")
            return False

        # Estimate tokens in context
        total_tokens = LLMContextSummarizationUtil.estimate_context_tokens(self._context)
        num_messages = len(self._context.messages)

        # Check if we've reached the token limit
        token_limit = self._config.max_context_tokens
        token_limit_exceeded = total_tokens >= token_limit

        # Check if we've exceeded max unsummarized messages
        messages_since_summary = len(self._context.messages) - 1
        message_threshold_exceeded = (
            messages_since_summary >= self._config.max_unsummarized_messages
        )

        logger.trace(
            f"{self}: Context has {num_messages} messages, "
            f"~{total_tokens} tokens (limit: {token_limit}), "
            f"{messages_since_summary} messages since last summary "
            f"(message threshold: {self._config.max_unsummarized_messages})"
        )

        # Trigger if either limit is exceeded
        if not token_limit_exceeded and not message_threshold_exceeded:
            logger.trace(
                f"{self}: Neither token limit nor message threshold exceeded, skipping summarization"
            )
            return False

        reason = []
        if token_limit_exceeded:
            reason.append(f"~{total_tokens} tokens (>={token_limit} limit)")
        if message_threshold_exceeded:
            reason.append(
                f"{messages_since_summary} messages (>={self._config.max_unsummarized_messages} threshold)"
            )

        logger.debug(f"{self}: ✓ Summarization needed - {', '.join(reason)}")
        return True

    async def _request_summarization(self):
        """Request context summarization from LLM service.

        Creates a summarization request frame and emits it via event handler.
        Tracks the request ID to match async responses and prevent race conditions.
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        min_keep = self._config.min_messages_after_summary

        # Mark summarization in progress
        self._summarization_in_progress = True
        self._pending_summary_request_id = request_id

        logger.debug(f"{self}: Sending summarization request (request_id={request_id})")

        # Create the request frame
        request_frame = LLMContextSummaryRequestFrame(
            request_id=request_id,
            context=self._context,
            min_messages_to_keep=min_keep,
            target_context_tokens=self._config.target_context_tokens,
            summarization_prompt=self._config.summary_prompt,
        )

        # Emit event for aggregator to broadcast
        await self._call_event_handler("on_request_summarization", request_frame)

    async def _handle_summary_result(self, frame: LLMContextSummaryResultFrame):
        """Handle context summarization result from LLM service.

        Processes the summary result by validating the request ID, checking for
        errors, validating context state, and applying the summary.

        Args:
            frame: The summary result frame containing the generated summary.
        """
        logger.debug(f"{self}: Received summary result (request_id={frame.request_id})")

        # Check if this is the result we're waiting for
        if frame.request_id != self._pending_summary_request_id:
            logger.debug(f"{self}: Ignoring stale summary result (request_id={frame.request_id})")
            return

        # Clear pending state
        await self._clear_summarization_state()

        # Check for errors
        if frame.error:
            logger.error(f"{self}: Context summarization failed: {frame.error}")
            return

        # Validate context state
        if not self._validate_summary_context(frame.last_summarized_index):
            logger.warning(f"{self}: Context state changed, skipping summary application")
            return

        # Apply summary
        await self._apply_summary(frame.summary, frame.last_summarized_index)

    def _validate_summary_context(self, last_summarized_index: int) -> bool:
        """Validate that context state is still valid for applying summary.

        Args:
            last_summarized_index: The index of the last summarized message.

        Returns:
            True if the context state is still consistent with the summary.
        """
        if last_summarized_index < 0:
            return False

        # Check if we still have enough messages
        if last_summarized_index >= len(self._context.messages):
            return False

        min_keep = self._config.min_messages_after_summary
        remaining = len(self._context.messages) - 1 - last_summarized_index
        if remaining < min_keep:
            return False

        return True

    async def _apply_summary(self, summary: str, last_summarized_index: int):
        """Apply summary to compress the conversation context.

        Reconstructs the context with:
        [first_system_message] + [summary_message] + [recent_messages]

        Args:
            summary: The generated summary text.
            last_summarized_index: Index of the last message that was summarized.
        """
        messages = self._context.messages

        # Find the first system message to preserve
        first_system_msg = next((msg for msg in messages if msg.get("role") == "system"), None)

        # Get recent messages to keep
        recent_messages = messages[last_summarized_index + 1 :]

        # Create summary message as an assistant message
        summary_message = {"role": "assistant", "content": f"Conversation summary: {summary}"}

        # Reconstruct context
        new_messages = []
        if first_system_msg:
            new_messages.append(first_system_msg)
        new_messages.append(summary_message)
        new_messages.extend(recent_messages)

        # Update context
        self._context.set_messages(new_messages)

        logger.info(
            f"{self}: Applied context summary, compressed {last_summarized_index + 1} messages "
            f"into summary. Context now has {len(new_messages)} messages (was {len(messages)})"
        )
