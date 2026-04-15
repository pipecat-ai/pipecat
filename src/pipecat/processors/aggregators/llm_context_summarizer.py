#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module defines a summarizer for managing LLM context summarization."""

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextSummaryRequestFrame,
    LLMContextSummaryResultFrame,
    LLMFullResponseStartFrame,
    LLMSummarizeContextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.utils.asyncio.task_manager import BaseTaskManager
from pipecat.utils.base_object import BaseObject
from pipecat.utils.context.llm_context_summarization import (
    DEFAULT_SUMMARIZATION_TIMEOUT,
    LLMAutoContextSummarizationConfig,
    LLMContextSummarizationUtil,
    LLMContextSummaryConfig,
)

if TYPE_CHECKING:
    from pipecat.services.llm_service import LLMService


@dataclass
class SummaryAppliedEvent:
    """Event data emitted when context summarization completes successfully.

    Parameters:
        original_message_count: Number of messages before summarization.
        new_message_count: Number of messages after summarization.
        summarized_message_count: Number of messages that were compressed
            into the summary.
        preserved_message_count: Number of recent messages preserved
            uncompressed.
    """

    original_message_count: int
    new_message_count: int
    summarized_message_count: int
    preserved_message_count: int


class LLMContextSummarizer(BaseObject):
    """Summarizer for managing LLM context summarization.

    This class manages context summarization, either automatically when token or
    message limits are reached, or on-demand when an ``LLMSummarizeContextFrame``
    is received. It monitors the LLM context size, triggers summarization requests,
    and applies the results to compress conversation history.

    When ``auto_trigger=True`` (the default), summarization is triggered
    automatically based on the configured thresholds in
    ``LLMAutoContextSummarizationConfig``. When ``auto_trigger=False``,
    threshold checks are skipped and summarization only happens when an
    ``LLMSummarizeContextFrame`` is explicitly pushed into the pipeline.

    Both modes can coexist: set ``auto_trigger=True`` and also push
    ``LLMSummarizeContextFrame`` at any time to force an immediate summarization
    (subject to the ``_summarization_in_progress`` guard).

    Event handlers available:

    - on_request_summarization: Emitted when summarization should be triggered.
        The aggregator should broadcast this frame to the LLM service.

    - on_summary_applied: Emitted after a summary has been successfully applied
        to the context. Receives a SummaryAppliedEvent with metrics about the
        compression.

    Example::

        @summarizer.event_handler("on_request_summarization")
        async def on_request_summarization(summarizer, frame: LLMContextSummaryRequestFrame):
            await aggregator.broadcast_frame(
                LLMContextSummaryRequestFrame,
                request_id=frame.request_id,
                context=frame.context,
                ...
            )

        @summarizer.event_handler("on_summary_applied")
        async def on_summary_applied(summarizer, event: SummaryAppliedEvent):
            logger.info(f"Compressed {event.original_message_count} -> {event.new_message_count} messages")
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        config: Optional[LLMAutoContextSummarizationConfig] = None,
        auto_trigger: bool = True,
    ):
        """Initialize the context summarizer.

        Args:
            context: The LLM context to monitor and summarize.
            config: Auto-summarization configuration controlling both trigger
                thresholds and default summary generation parameters. If None,
                uses default ``LLMAutoContextSummarizationConfig`` values.
            auto_trigger: Whether to automatically trigger summarization when
                thresholds are reached. When False, summarization only happens
                when an ``LLMSummarizeContextFrame`` is pushed into the pipeline.
                Defaults to True.
        """
        super().__init__()

        self._context = context
        self._auto_config = config or LLMAutoContextSummarizationConfig()
        self._auto_trigger = auto_trigger

        self._task_manager: Optional[BaseTaskManager] = None

        self._summarization_in_progress = False
        self._pending_summary_request_id: Optional[str] = None

        self._register_event_handler("on_request_summarization", sync=True)
        self._register_event_handler("on_summary_applied")

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
        elif isinstance(frame, LLMSummarizeContextFrame):
            await self._handle_manual_summarization_request(frame)
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

    async def _handle_manual_summarization_request(self, frame: LLMSummarizeContextFrame):
        """Handle an explicit on-demand summarization request.

        Reuses the same ``_request_summarization()`` code path as auto mode,
        so bookkeeping (``_summarization_in_progress``,
        ``_pending_summary_request_id``) is always updated correctly.

        Args:
            frame: The manual summarization request frame, optionally carrying
                a per-request :class:`~pipecat.utils.context.llm_context_summarization.LLMContextSummaryConfig`.
        """
        if self._summarization_in_progress:
            logger.debug(f"{self}: Summarization already in progress, ignoring manual request")
            return
        await self._request_summarization(config_override=frame.config)

    async def _handle_interruption(self):
        """Handle interruption by canceling summarization in progress."""
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
            - ``auto_trigger`` is enabled
            - No summarization currently in progress
            - AND either:
              - Token count exceeds ``max_context_tokens``
              - OR message count exceeds ``max_unsummarized_messages`` since last summary
        """
        logger.trace(f"{self}: Checking if context summarization is needed")

        if not self._auto_trigger:
            return False

        if self._summarization_in_progress:
            logger.debug(f"{self}: Summarization already in progress")
            return False

        # Estimate tokens in context
        total_tokens = LLMContextSummarizationUtil.estimate_context_tokens(self._context)
        num_messages = len(self._context.messages)

        # Check if we've reached the token limit
        token_limit = self._auto_config.max_context_tokens
        token_limit_exceeded = total_tokens >= token_limit

        # Check if we've exceeded max unsummarized messages
        messages_since_summary = len(self._context.messages) - 1
        message_threshold_exceeded = (
            messages_since_summary >= self._auto_config.max_unsummarized_messages
        )

        logger.trace(
            f"{self}: Context has {num_messages} messages, "
            f"~{total_tokens} tokens (limit: {token_limit}), "
            f"{messages_since_summary} messages since last summary "
            f"(message threshold: {self._auto_config.max_unsummarized_messages})"
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
                f"{messages_since_summary} messages (>={self._auto_config.max_unsummarized_messages} threshold)"
            )

        logger.debug(f"{self}: ✓ Summarization needed - {', '.join(reason)}")
        return True

    async def _request_summarization(
        self, config_override: Optional[LLMContextSummaryConfig] = None
    ):
        """Request context summarization from LLM service.

        Creates a summarization request frame and either handles it directly
        using a dedicated LLM (if configured) or emits it via event handler
        for the pipeline's primary LLM.
        Tracks the request ID to match async responses and prevent race conditions.

        Args:
            config_override: Optional per-request summary configuration. If provided,
                overrides the default summary generation settings from
                ``self._auto_config.summary_config``.
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        summary_config = config_override or self._auto_config.summary_config

        # Mark summarization in progress
        self._summarization_in_progress = True
        self._pending_summary_request_id = request_id

        logger.debug(f"{self}: Sending summarization request (request_id={request_id})")

        # Create the request frame
        request_frame = LLMContextSummaryRequestFrame(
            request_id=request_id,
            context=self._context,
            min_messages_to_keep=summary_config.min_messages_after_summary,
            target_context_tokens=summary_config.target_context_tokens,
            summarization_prompt=summary_config.summary_prompt,
            summarization_timeout=summary_config.summarization_timeout,
        )

        if summary_config.llm:
            # Use dedicated LLM directly — no need to involve the pipeline
            self.task_manager.create_task(
                self._generate_summary_with_dedicated_llm(summary_config.llm, request_frame),
                f"{self}-dedicated-llm-summary",
            )
        else:
            # Emit event for aggregator to broadcast to the pipeline LLM
            await self._call_event_handler("on_request_summarization", request_frame)

    async def _generate_summary_with_dedicated_llm(
        self, llm: "LLMService", frame: LLMContextSummaryRequestFrame
    ):
        """Generate summary using a dedicated LLM service.

        Calls the dedicated LLM's _generate_summary directly and feeds the
        result back through _handle_summary_result, bypassing the pipeline.

        Args:
            llm: The dedicated LLM service to use for summarization.
            frame: The summarization request frame.
        """
        timeout = frame.summarization_timeout or DEFAULT_SUMMARIZATION_TIMEOUT

        try:
            summary, last_index = await asyncio.wait_for(
                llm._generate_summary(frame),
                timeout=timeout,
            )
            result_frame = LLMContextSummaryResultFrame(
                request_id=frame.request_id,
                summary=summary,
                last_summarized_index=last_index,
            )
        except asyncio.TimeoutError:
            error = f"Context summarization timed out after {timeout}s"
            logger.error(f"{self}: {error}")
            result_frame = LLMContextSummaryResultFrame(
                request_id=frame.request_id,
                summary="",
                last_summarized_index=-1,
                error=error,
            )
        except Exception as e:
            error = f"Error generating context summary: {e}"
            logger.error(f"{self}: {error}")
            result_frame = LLMContextSummaryResultFrame(
                request_id=frame.request_id,
                summary="",
                last_summarized_index=-1,
                error=error,
            )

        await self._handle_summary_result(result_frame)

    async def _handle_summary_result(self, frame: LLMContextSummaryResultFrame):
        """Handle context summarization result from LLM service.

        Processes the summary result by validating the request ID, checking for
        errors, validating context state, and applying the summary.

        Args:
            frame: The summary result frame containing the generated summary.
        """
        logger.debug(f"{self}: Received summary result (request_id={frame.request_id})")

        # Check if this is the result we're waiting for. Both auto and manual
        # summarization set _pending_summary_request_id via _request_summarization(),
        # so this check always applies.
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

        min_keep = self._auto_config.summary_config.min_messages_after_summary
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
        config = self._auto_config.summary_config
        messages = self._context.messages

        # Find the first system message to preserve. LLMSpecificMessage instances are excluded
        # because they are not dict-like and never represent a system message; they hold
        # service-specific metadata (e.g. thinking blocks) that is always paired with a
        # standard message.
        first_system_msg = next(
            (
                msg
                for msg in messages
                if not isinstance(msg, LLMSpecificMessage) and msg.get("role") == "system"
            ),
            None,
        )

        # Get recent messages to keep
        recent_messages = messages[last_summarized_index + 1 :]

        # Create summary message as a user message (the summary is context
        # provided *to* the assistant, not something the assistant said)
        summary_content = config.summary_message_template.format(summary=summary)
        summary_message = {"role": "user", "content": summary_content}

        # Reconstruct context
        new_messages = []
        if first_system_msg:
            new_messages.append(first_system_msg)
        new_messages.append(summary_message)
        new_messages.extend(recent_messages)

        # Update context
        original_message_count = len(messages)
        num_system_preserved = 1 if first_system_msg else 0
        self._context.set_messages(new_messages)

        # Messages actually summarized = index range minus the preserved system message
        summarized_count = last_summarized_index + 1 - num_system_preserved

        logger.info(
            f"{self}: Applied context summary, compressed {summarized_count} messages "
            f"into summary. Context now has {len(new_messages)} messages (was {original_message_count})"
        )

        # Emit event for observability
        event = SummaryAppliedEvent(
            original_message_count=original_message_count,
            new_message_count=len(new_messages),
            summarized_message_count=summarized_count,
            preserved_message_count=len(recent_messages) + num_system_preserved,
        )
        await self._call_event_handler("on_summary_applied", event)
