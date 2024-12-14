#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    OpenAILLMContextAssistantTimestampFrame,
    OpenAILLMContextUserTimestampFrame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TranscriptProcessor(FrameProcessor):
    """Processes LLM context frames to generate timestamped conversation transcripts.

    This processor monitors OpenAILLMContextFrame frames and their corresponding
    timestamp frames to build a chronological conversation transcript. Messages are
    stored by role until their matching timestamp frame arrives, then emitted via
    TranscriptionUpdateFrame.

    Each LLM context (OpenAI, Anthropic, Google) provides conversion to the standard format:
    [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hi, how are you?"}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Great! And you?"}]
        }
    ]

    Events:
        on_transcript_update: Emitted when timestamped messages are available.
            Args: TranscriptionUpdateFrame containing timestamped messages.

    Example:
        ```python
        transcript_processor = TranscriptProcessor()

        @transcript_processor.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            for msg in frame.messages:
                print(f"[{msg.timestamp}] {msg.role}: {msg.content}")
        ```
    """

    def __init__(self, **kwargs):
        """Initialize the transcript processor.

        Args:
            **kwargs: Additional arguments passed to FrameProcessor
        """
        super().__init__(**kwargs)
        self._processed_messages: List[TranscriptionMessage] = []
        self._register_event_handler("on_transcript_update")
        self._pending_user_messages: List[TranscriptionMessage] = []
        self._pending_assistant_messages: List[TranscriptionMessage] = []

    def _extract_messages(self, messages: List[dict]) -> List[TranscriptionMessage]:
        """Extract conversation messages from standard format.

        Args:
            messages: List of messages in standard format with structured content

        Returns:
            List[TranscriptionMessage]: Normalized conversation messages
        """
        result = []
        for msg in messages:
            # Only process user and assistant messages
            if msg["role"] not in ("user", "assistant"):
                continue

            content = msg.get("content", [])
            if isinstance(content, list):
                # Extract text from structured content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])

                if text_parts:
                    result.append(
                        TranscriptionMessage(role=msg["role"], content=" ".join(text_parts))
                    )

        return result

    def _find_new_messages(self, current: List[TranscriptionMessage]) -> List[TranscriptionMessage]:
        """Find messages in current that aren't in self._processed_messages.

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
        """Process frames to build a timestamped conversation transcript.

        Handles three frame types in sequence:
        1. OpenAILLMContextFrame: Contains new messages to be timestamped
        2. OpenAILLMContextUserTimestampFrame: Timestamp for user messages
        3. OpenAILLMContextAssistantTimestampFrame: Timestamp for assistant messages

        Messages are stored by role until their corresponding timestamp frame arrives.
        When a timestamp frame is received, the matching messages are timestamped and
        emitted in chronological order via TranscriptionUpdateFrame.

        Args:
            frame: The frame to process
            direction: Frame processing direction

        Raises:
            ErrorFrame: If message processing fails
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            # Extract and store messages by role
            standard_messages = []
            for msg in frame.context.messages:
                converted = frame.context.to_standard_messages(msg)
                standard_messages.extend(converted)

            current_messages = self._extract_messages(standard_messages)
            new_messages = self._find_new_messages(current_messages)

            # Store new messages by role
            for msg in new_messages:
                if msg.role == "user":
                    self._pending_user_messages.append(msg)
                elif msg.role == "assistant":
                    self._pending_assistant_messages.append(msg)

        elif isinstance(frame, OpenAILLMContextUserTimestampFrame):
            # Process pending user messages with timestamp
            if self._pending_user_messages:
                for msg in self._pending_user_messages:
                    msg.timestamp = frame.timestamp
                self._processed_messages.extend(self._pending_user_messages)
                update_frame = TranscriptionUpdateFrame(messages=self._pending_user_messages)
                await self._call_event_handler("on_transcript_update", update_frame)
                await self.push_frame(update_frame)
                self._pending_user_messages = []

        elif isinstance(frame, OpenAILLMContextAssistantTimestampFrame):
            # Process pending assistant messages with timestamp
            if self._pending_assistant_messages:
                for msg in self._pending_assistant_messages:
                    msg.timestamp = frame.timestamp
                self._processed_messages.extend(self._pending_assistant_messages)
                update_frame = TranscriptionUpdateFrame(messages=self._pending_assistant_messages)
                await self._call_event_handler("on_transcript_update", update_frame)
                await self.push_frame(update_frame)
                self._pending_assistant_messages = []

        await self.push_frame(frame, direction)
