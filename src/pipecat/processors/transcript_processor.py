#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    StartInterruptionFrame,
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
        """Initialize processor with empty message store."""
        super().__init__(**kwargs)
        self._processed_messages: List[TranscriptionMessage] = []
        self._register_event_handler("on_transcript_update")

    async def _emit_update(self, messages: List[TranscriptionMessage]):
        """Emit transcript updates for new messages.

        Args:
            messages: New messages to emit in update
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
            frame: Input frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            message = TranscriptionMessage(
                role="user", content=frame.text, timestamp=frame.timestamp
            )
            await self._emit_update([message])

        await self.push_frame(frame, direction)


class AssistantTranscriptProcessor(BaseTranscriptProcessor):
    """Processes assistant TTS text frames into timestamped conversation messages.

    This processor aggregates TTS text frames into complete utterances and emits them as
    transcript messages. Utterances are completed when:
    - The bot stops speaking (BotStoppedSpeakingFrame)
    - The bot is interrupted (StartInterruptionFrame)
    - The pipeline ends (EndFrame)

    Attributes:
        _current_text_parts: List of text fragments being aggregated for current utterance
        _aggregation_start_time: Timestamp when the current utterance began
    """

    def __init__(self, **kwargs):
        """Initialize processor with aggregation state."""
        super().__init__(**kwargs)
        self._current_text_parts: List[str] = []
        self._aggregation_start_time: Optional[str] = None

    async def _emit_aggregated_text(self):
        """Emit aggregated text as a transcript message.

        This method intelligently joins text fragments to create natural spacing,
        handling both word-by-word and pre-spaced text fragments appropriately.

        The implementation handles two common patterns from TTS services:

        1. Word-by-word fragments without spacing:
        ```
        TTSTextFrame: ['Hello.']
        TTSTextFrame: ['How']
        TTSTextFrame: ['can']
        TTSTextFrame: ['I']
        TTSTextFrame: ['assist']
        TTSTextFrame: ['you']
        TTSTextFrame: ['today?']
        ```
        Result: "Hello. How can I assist you today?"

        2. Pre-spaced fragments:
        ```
        TTSTextFrame: ['Hello']
        TTSTextFrame: [' there']
        TTSTextFrame: ['!']
        TTSTextFrame: [' How']
        TTSTextFrame: ["'s"]
        TTSTextFrame: [' it']
        TTSTextFrame: [' going']
        TTSTextFrame: ['?']
        ```
        Result: "Hello there! How's it going?"
        """
        if self._current_text_parts and self._aggregation_start_time:
            # Build content with intelligent spacing
            content = ""
            for i, part in enumerate(self._current_text_parts):
                # Add a space only when the current part doesn't start with
                # whitespace or punctuation/special characters
                if i > 0 and not part.startswith((" ", ".", ",", "!", "?", ";", ":", "'", '"')):
                    content += " "
                content += part

            content = content.strip()

            if content:
                logger.debug(f"Emitting aggregated assistant message: {content}")
                message = TranscriptionMessage(
                    role="assistant",
                    content=content,
                    timestamp=self._aggregation_start_time,
                )
                await self._emit_update([message])
            else:
                logger.debug("No content to emit after stripping whitespace")

            # Reset aggregation state
            self._current_text_parts = []
            self._aggregation_start_time = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames into assistant conversation messages.

        Handles different frame types:
        - TTSTextFrame: Aggregates text for current utterance
        - BotStoppedSpeakingFrame: Completes current utterance
        - StartInterruptionFrame: Completes current utterance due to interruption
        - EndFrame: Completes current utterance at pipeline end
        - CancelFrame: Completes current utterance due to cancellation

        Args:
            frame: Input frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSTextFrame):
            # Start timestamp on first text part
            if not self._aggregation_start_time:
                self._aggregation_start_time = time_now_iso8601()

            self._current_text_parts.append(frame.text)

        elif isinstance(frame, (BotStoppedSpeakingFrame, StartInterruptionFrame, CancelFrame)):
            # Emit accumulated text when bot finishes speaking or is interrupted
            await self._emit_aggregated_text()

        elif isinstance(frame, EndFrame):
            # Emit any remaining text when pipeline ends
            await self._emit_aggregated_text()

        await self.push_frame(frame, direction)


class TranscriptProcessor:
    """Factory for creating and managing transcript processors.

    Provides unified access to user and assistant transcript processors
    with shared event handling.

    Example:
        ```python
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
        ```
    """

    def __init__(self):
        """Initialize factory."""
        self._user_processor = None
        self._assistant_processor = None
        self._event_handlers = {}

    def user(self, **kwargs) -> UserTranscriptProcessor:
        """Get the user transcript processor.

        Args:
            **kwargs: Arguments specific to UserTranscriptProcessor
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
            **kwargs: Arguments specific to AssistantTranscriptProcessor
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
            event_name: Name of event to handle

        Returns:
            Decorator function that registers handler with both processors
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
