#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcript processing utilities for conversation recording and analysis.

This module provides processors that convert speech and text frames into structured
transcript messages with timestamps, enabling conversation history tracking and analysis.
"""

from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    ThoughtTranscriptionMessage,
    TranscriptionFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text
from pipecat.utils.time import time_now_iso8601


class BaseTranscriptProcessor(FrameProcessor):
    """Base class for processing conversation transcripts.

    Provides common functionality for handling transcript messages and updates.
    """

    def __init__(self, **kwargs):
        """Initialize processor with empty message store.

        Args:
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._processed_messages: List[TranscriptionMessage] = []
        self._register_event_handler("on_transcript_update")

    async def _emit_update(self, messages: List[TranscriptionMessage]):
        """Emit transcript updates for new messages.

        Args:
            messages: New messages to emit in update.
        """
        if messages:
            self._processed_messages.extend(messages)
            update_frame = TranscriptionUpdateFrame(messages=messages)
            await self._call_event_handler("on_transcript_update", update_frame)
            await self.push_frame(update_frame)


class UserTranscriptProcessor(BaseTranscriptProcessor):
    """Processes user transcription frames into timestamped conversation messages."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process TranscriptionFrames into user conversation messages.

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            message = TranscriptionMessage(
                role="user", user_id=frame.user_id, content=frame.text, timestamp=frame.timestamp
            )
            await self._emit_update([message])

        await self.push_frame(frame, direction)


class AssistantTranscriptProcessor(BaseTranscriptProcessor):
    """Processes assistant TTS text frames and LLM thought frames into timestamped messages.

    This processor aggregates both TTS text frames and LLM thought frames into
    complete utterances and thoughts, emitting them as transcript messages.

    An assistant utterance is completed when:
    - The bot stops speaking (BotStoppedSpeakingFrame)
    - The bot is interrupted (InterruptionFrame)
    - The pipeline ends (EndFrame, CancelFrame)

    A thought is completed when:
    - The thought ends (LLMThoughtEndFrame)
    - The bot is interrupted (InterruptionFrame)
    - The pipeline ends (EndFrame, CancelFrame)
    """

    def __init__(self, *, process_thoughts: bool = False, **kwargs):
        """Initialize processor with aggregation state.

        Args:
            process_thoughts: Whether to process LLM thought frames. Defaults to False.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        self._process_thoughts = process_thoughts
        self._current_assistant_text_parts: List[TextPartForConcatenation] = []
        self._assistant_text_start_time: Optional[str] = None

        self._current_thought_parts: List[TextPartForConcatenation] = []
        self._thought_start_time: Optional[str] = None
        self._thought_active = False

    async def _emit_aggregated_assistant_text(self):
        """Aggregates and emits text fragments as a transcript message.

        This method aggregates text fragments that may arrive in multiple
        TTSTextFrame instances and emits them as a single TranscriptionMessage.
        """
        if self._current_assistant_text_parts and self._assistant_text_start_time:
            content = concatenate_aggregated_text(self._current_assistant_text_parts)
            if content:
                logger.trace(f"Emitting aggregated assistant message: {content}")
                message = TranscriptionMessage(
                    role="assistant",
                    content=content,
                    timestamp=self._assistant_text_start_time,
                )
                await self._emit_update([message])
            else:
                logger.trace("No content to emit after stripping whitespace")

            # Reset aggregation state
            self._current_assistant_text_parts = []
            self._assistant_text_start_time = None

    async def _emit_aggregated_thought(self):
        """Aggregates and emits thought text fragments as a thought transcript message.

        This method aggregates thought fragments that may arrive in multiple
        LLMThoughtTextFrame instances and emits them as a single ThoughtTranscriptionMessage.
        """
        if self._current_thought_parts and self._thought_start_time:
            content = concatenate_aggregated_text(self._current_thought_parts)
            if content:
                logger.trace(f"Emitting aggregated thought message: {content}")
                message = ThoughtTranscriptionMessage(
                    content=content,
                    timestamp=self._thought_start_time,
                )
                await self._emit_update([message])
            else:
                logger.trace("No thought content to emit after stripping whitespace")

            # Reset aggregation state
            self._current_thought_parts = []
            self._thought_start_time = None
            self._thought_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames into assistant conversation messages and thought messages.

        Handles different frame types:

        - TTSTextFrame: Aggregates text for current utterance
        - LLMThoughtStartFrame: Begins aggregating a new thought
        - LLMThoughtTextFrame: Aggregates text for current thought
        - LLMThoughtEndFrame: Completes current thought
        - BotStoppedSpeakingFrame: Completes current utterance
        - InterruptionFrame: Completes current utterance and thought due to interruption
        - EndFrame: Completes current utterance and thought at pipeline end
        - CancelFrame: Completes current utterance and thought due to cancellation

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            # Push frame first otherwise our emitted transcription update frame
            # might get cleaned up.
            await self.push_frame(frame, direction)
            # Emit accumulated text and thought with interruptions
            await self._emit_aggregated_assistant_text()
            if self._process_thoughts and self._thought_active:
                await self._emit_aggregated_thought()
        elif isinstance(frame, LLMThoughtStartFrame):
            # Start a new thought
            if self._process_thoughts:
                self._thought_active = True
                self._thought_start_time = time_now_iso8601()
                self._current_thought_parts = []
            # Push frame.
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMThoughtTextFrame):
            # Aggregate thought text if we have an active thought
            if self._process_thoughts and self._thought_active:
                self._current_thought_parts.append(
                    TextPartForConcatenation(
                        frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
                    )
                )
            # Push frame.
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMThoughtEndFrame):
            # Emit accumulated thought when thought ends
            if self._process_thoughts and self._thought_active:
                await self._emit_aggregated_thought()
            # Push frame.
            await self.push_frame(frame, direction)
        elif isinstance(frame, TTSTextFrame):
            # Start timestamp on first text part
            if not self._assistant_text_start_time:
                self._assistant_text_start_time = time_now_iso8601()

            self._current_assistant_text_parts.append(
                TextPartForConcatenation(
                    frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
                )
            )

            # Push frame.
            await self.push_frame(frame, direction)
        elif isinstance(frame, (BotStoppedSpeakingFrame, EndFrame)):
            # Emit accumulated text when bot finishes speaking or pipeline ends.
            await self._emit_aggregated_assistant_text()
            # Emit accumulated thought at pipeline end if still active
            if isinstance(frame, EndFrame) and self._process_thoughts and self._thought_active:
                await self._emit_aggregated_thought()
            # Push frame.
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class TranscriptProcessor:
    """Factory for creating and managing transcript processors.

    Provides unified access to user and assistant transcript processors
    with shared event handling. The assistant processor handles both TTS text
    and LLM thought frames.

    Example::

        transcript = TranscriptProcessor()

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                transcript.user(),              # User transcripts
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                transcript.assistant(),         # Assistant transcripts (including thoughts)
                context_aggregator.assistant(),
            ]
        )

        @transcript.event_handler("on_transcript_update")
        async def handle_update(processor, frame):
            print(f"New messages: {frame.messages}")

    .. deprecated:: 0.0.99
        `TranscriptProcessor` is deprecated and will be removed in a future version.
        Use `LLMUserAggregator`'s and `LLMAssistantAggregator`'s new events instead.
    """

    def __init__(self, *, process_thoughts: bool = False):
        """Initialize factory.

        Args:
            process_thoughts: Whether the assistant processor should handle LLM thought
                frames. Defaults to False.
        """
        self._process_thoughts = process_thoughts
        self._user_processor = None
        self._assistant_processor = None
        self._event_handlers = {}

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`TranscriptProcessor` is deprecated and will be removed in a future version. "
                "Use `LLMUserAggregator`'s and `LLMAssistantAggregator`'s new events instead.",
                DeprecationWarning,
            )

    def user(self, **kwargs) -> UserTranscriptProcessor:
        """Get the user transcript processor.

        Args:
            **kwargs: Arguments specific to UserTranscriptProcessor.

        Returns:
            The user transcript processor instance.
        """
        if self._user_processor is None:
            self._user_processor = UserTranscriptProcessor(**kwargs)
            # Apply any registered event handlers
            for event_name, handler in self._event_handlers.items():

                @self._user_processor.event_handler(event_name)
                async def user_handler(processor, frame):
                    return await handler(processor, frame)

        return self._user_processor

    def assistant(self, **kwargs) -> AssistantTranscriptProcessor:
        """Get the assistant transcript processor.

        Args:
            **kwargs: Arguments specific to AssistantTranscriptProcessor.

        Returns:
            The assistant transcript processor instance.
        """
        if self._assistant_processor is None:
            self._assistant_processor = AssistantTranscriptProcessor(
                process_thoughts=self._process_thoughts, **kwargs
            )
            # Apply any registered event handlers
            for event_name, handler in self._event_handlers.items():

                @self._assistant_processor.event_handler(event_name)
                async def assistant_handler(processor, frame):
                    return await handler(processor, frame)

        return self._assistant_processor

    def event_handler(self, event_name: str):
        """Register event handler for both processors.

        Args:
            event_name: Name of event to handle.

        Returns:
            Decorator function that registers handler with both processors.
        """

        def decorator(handler):
            self._event_handlers[event_name] = handler

            # Apply handler to existing processors if they exist
            if self._user_processor:

                @self._user_processor.event_handler(event_name)
                async def user_handler(processor, frame):
                    return await handler(processor, frame)

            if self._assistant_processor:

                @self._assistant_processor.event_handler(event_name)
                async def assistant_handler(processor, frame):
                    return await handler(processor, frame)

            return handler

        return decorator
