#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from typing import List

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    OpenAILLMContextAssistantTimestampFrame,
    TranscriptionFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BaseTranscriptProcessor(FrameProcessor, ABC):
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

    @abstractmethod
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames to build conversation transcript.

        Args:
            frame: Input frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)


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
    """Processes assistant LLM context frames into timestamped conversation messages."""

    def __init__(self, **kwargs):
        """Initialize processor with empty message stores."""
        super().__init__(**kwargs)
        self._pending_assistant_messages: List[TranscriptionMessage] = []

    def _extract_messages(self, messages: List[dict]) -> List[TranscriptionMessage]:
        """Extract assistant messages from the OpenAI standard message format.

        Args:
            messages: List of messages in OpenAI format, which can be either:
                - Simple format: {"role": "user", "content": "Hello"}
                - Content list: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

        Returns:
            List[TranscriptionMessage]: Normalized conversation messages
        """
        result = []
        for msg in messages:
            if msg["role"] != "assistant":
                continue

            content = msg.get("content")
            if isinstance(content, str):
                if content:
                    result.append(TranscriptionMessage(role="assistant", content=content))
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])

                if text_parts:
                    result.append(
                        TranscriptionMessage(role="assistant", content=" ".join(text_parts))
                    )

        return result

    def _find_new_messages(self, current: List[TranscriptionMessage]) -> List[TranscriptionMessage]:
        """Find unprocessed messages from current list.

        Args:
            current: List of current messages

        Returns:
            List[TranscriptionMessage]: New messages not yet processed
        """
        if not self._processed_messages:
            return current

        processed_len = len(self._processed_messages)
        if len(current) <= processed_len:
            return []

        return current[processed_len:]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames into assistant conversation messages.

        Args:
            frame: Input frame to process
            direction: Frame processing direction
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            standard_messages = []
            for msg in frame.context.messages:
                converted = frame.context.to_standard_messages(msg)
                standard_messages.extend(converted)

            current_messages = self._extract_messages(standard_messages)
            new_messages = self._find_new_messages(current_messages)
            self._pending_assistant_messages.extend(new_messages)

        elif isinstance(frame, OpenAILLMContextAssistantTimestampFrame):
            if self._pending_assistant_messages:
                for msg in self._pending_assistant_messages:
                    msg.timestamp = frame.timestamp
                await self._emit_update(self._pending_assistant_messages)
                self._pending_assistant_messages = []

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
                context_aggregator.assistant(),
                transcript.assistant(),         # Assistant transcripts
            ]
        )

        @transcript.event_handler("on_transcript_update")
        async def handle_update(processor, frame):
            print(f"New messages: {frame.messages}")
        ```
    """

    def __init__(self, **kwargs):
        """Initialize factory with user and assistant processors."""
        self._user_processor = UserTranscriptProcessor(**kwargs)
        self._assistant_processor = AssistantTranscriptProcessor(**kwargs)
        self._event_handlers = {}

    def user(self) -> UserTranscriptProcessor:
        """Get the user transcript processor."""
        return self._user_processor

    def assistant(self) -> AssistantTranscriptProcessor:
        """Get the assistant transcript processor."""
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

            @self._user_processor.event_handler(event_name)
            async def user_handler(processor, frame):
                return await handler(processor, frame)

            @self._assistant_processor.event_handler(event_name)
            async def assistant_handler(processor, frame):
                return await handler(processor, frame)

            return handler

        return decorator
