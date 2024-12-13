#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import List

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionMessage,
    TranscriptionUpdateFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TranscriptProcessor(FrameProcessor):
    """Processes LLM context frames to generate conversation transcripts.

    This processor monitors OpenAILLMContextFrame frames and extracts conversation
    content, filtering out system messages and function calls. When new messages
    are detected, it emits a TranscriptionUpdateFrame containing only the new
    messages.

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
        on_transcript_update: Emitted when new transcript messages are available.
            Args: TranscriptionUpdateFrame containing new messages.

    Example:
        ```python
        transcript_processor = TranscriptProcessor()

        @transcript_processor.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            for msg in frame.messages:
                print(f"{msg.role}: {msg.content}")
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
        """Process incoming frames, watching for OpenAILLMContextFrame.

        Args:
            frame: The frame to process
            direction: Frame processing direction

        Raises:
            ErrorFrame: If message processing fails
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            try:
                # Convert context messages to standard format
                standard_messages = []
                for msg in frame.context.messages:
                    converted = frame.context.to_standard_messages(msg)
                    standard_messages.extend(converted)

                # Extract and process messages
                current_messages = self._extract_messages(standard_messages)
                new_messages = self._find_new_messages(current_messages)

                if new_messages:
                    # Update state and notify listeners
                    self._processed_messages.extend(new_messages)
                    update_frame = TranscriptionUpdateFrame(messages=new_messages)
                    await self._call_event_handler("on_transcript_update", update_frame)
                    await self.push_frame(update_frame)

            except Exception as e:
                logger.error(f"Error processing transcript in {self}: {e}")
                await self.push_error(ErrorFrame(str(e)))

        # Always push the original frame downstream
        await self.push_frame(frame, direction)
