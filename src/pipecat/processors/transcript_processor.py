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
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
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
        self._current_text_parts: List[TextPartForConcatenation] = []
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
            content = concatenate_aggregated_text(self._current_text_parts)
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

            self._current_text_parts.append(
                TextPartForConcatenation(
                    frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
                )
            )

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


class TurnAwareTranscriptProcessor(BaseTranscriptProcessor):
    """Processes transcripts with turn boundary awareness.

    This processor combines user and assistant transcript tracking with turn
    detection, emitting events when turns start and end. It correctly handles
    interruptions by only capturing what was actually spoken.

    Turn boundaries are detected based on:
    - User started speaking (UserStartedSpeakingFrame)
    - Bot stopped speaking (BotStoppedSpeakingFrame)
    - Interruptions (InterruptionFrame)

    Events:
        on_turn_started: Emitted when a new turn begins.
            Handler signature: async def handler(processor, turn_number)

        on_turn_ended: Emitted when a turn ends.
            Handler signature: async def handler(processor, turn_number,
                                                user_transcript, assistant_transcript,
                                                was_interrupted)

        on_transcript_update: Inherited from BaseTranscriptProcessor, emitted for
            individual transcript messages.

    Example::

        turn_processor = TurnAwareTranscriptProcessor()

        @turn_processor.event_handler("on_turn_started")
        async def handle_turn_started(processor, turn_number):
            print(f"Turn {turn_number} started")

        @turn_processor.event_handler("on_turn_ended")
        async def handle_turn_ended(processor, turn_number, user_text, assistant_text, interrupted):
            print(f"Turn {turn_number} ended")
            print(f"User said: {user_text}")
            print(f"Assistant said: {assistant_text}")
            print(f"Was interrupted: {interrupted}")

        pipeline = Pipeline([
            transport.input(),
            stt,
            turn_processor,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
    """

    def __init__(self, **kwargs):
        """Initialize the turn-aware transcript processor.

        Args:
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        # Turn tracking state
        self._turn_number = 0
        self._turn_active = False
        self._turn_start_time: Optional[str] = None

        # Accumulate text for current turn
        self._current_turn_user_parts: List[TextPartForConcatenation] = []
        self._current_turn_assistant_parts: List[TextPartForConcatenation] = []

        # Track bot speaking state
        self._bot_is_speaking = False

        # Register turn events
        self._register_event_handler("on_turn_started")
        self._register_event_handler("on_turn_ended")

    async def _start_turn(self):
        """Start a new turn."""
        if not self._turn_active:
            self._turn_number += 1
            self._turn_active = True
            self._turn_start_time = time_now_iso8601()
            self._current_turn_user_parts = []
            self._current_turn_assistant_parts = []

            logger.debug(f"Turn {self._turn_number} started")
            await self._call_event_handler("on_turn_started", self._turn_number)

    async def _end_turn(self, was_interrupted: bool = False):
        """End the current turn and emit aggregated transcripts.

        Args:
            was_interrupted: Whether the turn ended due to an interruption.
        """
        if not self._turn_active:
            return

        # Aggregate user text
        user_transcript = ""
        if self._current_turn_user_parts:
            user_transcript = concatenate_aggregated_text(self._current_turn_user_parts)

        # Aggregate assistant text
        assistant_transcript = ""
        if self._current_turn_assistant_parts:
            assistant_transcript = concatenate_aggregated_text(self._current_turn_assistant_parts)

        # Emit turn ended event
        logger.debug(
            f"Turn {self._turn_number} ended (interrupted={was_interrupted}). "
            f"User: '{user_transcript}', Assistant: '{assistant_transcript}'"
        )
        await self._call_event_handler(
            "on_turn_ended",
            self._turn_number,
            user_transcript,
            assistant_transcript,
            was_interrupted,
        )

        # Reset turn state
        self._turn_active = False
        self._current_turn_user_parts = []
        self._current_turn_assistant_parts = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for turn-aware transcript tracking.

        Handles:
        - UserStartedSpeakingFrame: Start new turn
        - TranscriptionFrame: Accumulate user speech and emit transcript message
        - BotStartedSpeakingFrame: Track bot speaking state
        - TTSTextFrame: Accumulate assistant speech
        - BotStoppedSpeakingFrame: End turn if no interruption pending
        - InterruptionFrame: End turn immediately as interrupted
        - EndFrame/CancelFrame: End any active turn

        Args:
            frame: Input frame to process.
            direction: Frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking
            if self._bot_is_speaking:
                # This is an interruption - end the current turn with what was spoken
                if self._current_turn_assistant_parts:
                    assistant_content = concatenate_aggregated_text(
                        self._current_turn_assistant_parts
                    )
                    if assistant_content:
                        message = TranscriptionMessage(
                            role="assistant",
                            content=assistant_content,
                            timestamp=self._turn_start_time or time_now_iso8601(),
                        )
                        await self._emit_update([message])
                await self._end_turn(was_interrupted=True)
                self._bot_is_speaking = False
            elif self._turn_active:
                # Previous turn is ending normally (bot finished speaking)
                if self._current_turn_assistant_parts:
                    assistant_content = concatenate_aggregated_text(
                        self._current_turn_assistant_parts
                    )
                    if assistant_content:
                        message = TranscriptionMessage(
                            role="assistant",
                            content=assistant_content,
                            timestamp=self._turn_start_time or time_now_iso8601(),
                        )
                        await self._emit_update([message])
                await self._end_turn(was_interrupted=False)

            # Start a new turn
            await self._start_turn()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            # Accumulate user speech for the current turn
            if self._turn_active:
                self._current_turn_user_parts.append(
                    TextPartForConcatenation(frame.text, includes_inter_part_spaces=True)
                )

            # Also emit individual transcript message
            message = TranscriptionMessage(
                role="user",
                user_id=frame.user_id,
                content=frame.text,
                timestamp=frame.timestamp,
            )
            await self._emit_update([message])
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStartedSpeakingFrame):
            # Bot started speaking
            self._bot_is_speaking = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSTextFrame):
            # Accumulate assistant speech for the current turn
            if self._turn_active:
                self._current_turn_assistant_parts.append(
                    TextPartForConcatenation(
                        frame.text, includes_inter_part_spaces=frame.includes_inter_frame_spaces
                    )
                )
            await self.push_frame(frame, direction)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Bot stopped speaking - just mark it, don't end turn yet
            # Turn will end when next user speaks or pipeline ends
            self._bot_is_speaking = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            # Handle interruption
            # Give a brief moment for any pending TTSTextFrames to process
            import asyncio

            await asyncio.sleep(0.001)

            # Emit assistant transcript message with what was spoken before interruption
            if self._current_turn_assistant_parts:
                assistant_content = concatenate_aggregated_text(self._current_turn_assistant_parts)
                if assistant_content:
                    message = TranscriptionMessage(
                        role="assistant",
                        content=assistant_content,
                        timestamp=self._turn_start_time or time_now_iso8601(),
                    )
                    await self._emit_update([message])

            # Push frame first to ensure proper cleanup
            await self.push_frame(frame, direction)

            # End turn as interrupted
            await self._end_turn(was_interrupted=True)
            self._bot_is_speaking = False

        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Pipeline ending - finalize any active turn
            if self._turn_active:
                # Emit any pending assistant transcript (allow time for TTSTextFrames to be processed)
                # Give a brief moment for any pending frames to process
                import asyncio

                await asyncio.sleep(0.001)

                if self._current_turn_assistant_parts:
                    assistant_content = concatenate_aggregated_text(
                        self._current_turn_assistant_parts
                    )
                    if assistant_content:
                        message = TranscriptionMessage(
                            role="assistant",
                            content=assistant_content,
                            timestamp=self._turn_start_time or time_now_iso8601(),
                        )
                        await self._emit_update([message])

                await self._end_turn(was_interrupted=isinstance(frame, CancelFrame))

            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
