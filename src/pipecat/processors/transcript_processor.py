#
# Copyright (c) 2024–2025, Daily
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
    TranscriptionFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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
    """Processes assistant TTS text frames into timestamped conversation messages.

    This processor aggregates TTS text frames into complete utterances and emits them as
    transcript messages. Utterances are completed when:

    - The bot stops speaking (BotStoppedSpeakingFrame)
    - The bot is interrupted (InterruptionFrame)
    - The pipeline ends (EndFrame)
    """

    def __init__(self, **kwargs):
        """Initialize processor with aggregation state.

        Args:
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._current_text_parts: List[str] = []
        self._aggregation_start_time: Optional[str] = None

    async def _emit_aggregated_text(self):
        """Aggregates and emits text fragments as a transcript message.

        This method uses a heuristic to automatically detect whether text fragments
        contain embedded spacing (spaces at the beginning or end of fragments) or not,
        and applies the appropriate joining strategy. It handles fragments from different
        TTS services with different formatting patterns.

        Examples:
            Fragments with embedded spacing (concatenated)::

                TTSTextFrame: ["Hello"]
                TTSTextFrame: [" there"]  # Leading space
                TTSTextFrame: ["!"]
                TTSTextFrame: [" How"]    # Leading space
                TTSTextFrame: ["'s"]
                TTSTextFrame: [" it"]     # Leading space

                Result: "Hello there! How's it"

            Fragments with trailing spaces (concatenated)::

                TTSTextFrame: ["Hel"]
                TTSTextFrame: ["lo "]     # Trailing space
                TTSTextFrame: ["to "]     # Trailing space
                TTSTextFrame: ["you"]

                Result: "Hello to you"

            Word-by-word fragments without spacing (joined with spaces)::

                TTSTextFrame: ["Hello"]
                TTSTextFrame: ["there"]
                TTSTextFrame: ["how"]
                TTSTextFrame: ["are"]
                TTSTextFrame: ["you"]

                Result: "Hello there how are you"
        """
        if self._current_text_parts and self._aggregation_start_time:
            # Check specifically for space characters, previously isspace() was used
            # but that includes all whitespace characters (e.g. \n), not just spaces.
            has_leading_spaces = any(
                part and part[0] == " " for part in self._current_text_parts[1:]
            )
            has_trailing_spaces = any(
                part and part[-1] == " " for part in self._current_text_parts[:-1]
            )

            # If there are embedded spaces in the fragments, use direct concatenation
            contains_spacing_between_fragments = has_leading_spaces or has_trailing_spaces

            # Apply corresponding joining method
            if contains_spacing_between_fragments:
                # Fragments already have spacing - just concatenate
                content = "".join(self._current_text_parts)
            else:
                # Word-by-word fragments - join with spaces
                content = " ".join(self._current_text_parts)

            # Clean up any excessive whitespace
            content = content.strip()

            if content:
                logger.trace(f"Emitting aggregated assistant message: {content}")
                message = TranscriptionMessage(
                    role="assistant",
                    content=content,
                    timestamp=self._aggregation_start_time,
                )
                await self._emit_update([message])
            else:
                logger.trace("No content to emit after stripping whitespace")

            # Reset aggregation state
            self._current_text_parts = []
            self._aggregation_start_time = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames into assistant conversation messages.

        Handles different frame types:

        - TTSTextFrame: Aggregates text for current utterance
        - BotStoppedSpeakingFrame: Completes current utterance
        - InterruptionFrame: Completes current utterance due to interruption
        - EndFrame: Completes current utterance at pipeline end
        - CancelFrame: Completes current utterance due to cancellation

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            # Push frame first otherwise our emitted transcription update frame
            # might get cleaned up.
            await self.push_frame(frame, direction)
            # Emit accumulated text with interruptions
            await self._emit_aggregated_text()
        elif isinstance(frame, TTSTextFrame):
            # Start timestamp on first text part
            if not self._aggregation_start_time:
                self._aggregation_start_time = time_now_iso8601()

            self._current_text_parts.append(frame.text)

            # Push frame.
            await self.push_frame(frame, direction)
        elif isinstance(frame, (BotStoppedSpeakingFrame, EndFrame)):
            # Emit accumulated text when bot finishes speaking or pipeline ends.
            await self._emit_aggregated_text()
            # Push frame.
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class TranscriptProcessor:
    """Factory for creating and managing transcript processors.

    Provides unified access to user and assistant transcript processors
    with shared event handling.

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
                transcript.assistant_tts(),     # Assistant transcripts
                context_aggregator.assistant(),
            ]
        )

        @transcript.event_handler("on_transcript_update")
        async def handle_update(processor, frame):
            print(f"New messages: {frame.messages}")
    """

    def __init__(self):
        """Initialize factory."""
        self._user_processor = None
        self._assistant_processor = None
        self._event_handlers = {}

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
            self._assistant_processor = AssistantTranscriptProcessor(**kwargs)
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
