#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Langchain integration processor for Pipecat."""

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import Runnable
except ModuleNotFoundError as e:
    logger.error("In order to use Langchain, you need to `pip install pipecat-ai[langchain]`. ")
    raise Exception(f"Missing module: {e}")


class LangchainProcessor(FrameProcessor):
    """Processor that integrates Langchain runnables with Pipecat's frame pipeline.

    This processor takes LLM message frames, extracts the latest user message,
    and processes it through a Langchain runnable chain. The response is streamed
    back as text frames with appropriate response markers.
    """

    def __init__(self, chain: Runnable, transcript_key: str = "input"):
        """Initialize the Langchain processor.

        Args:
            chain: The Langchain runnable to use for processing messages.
            transcript_key: The key to use when passing input to the chain.
        """
        super().__init__()
        self._chain = chain
        self._transcript_key = transcript_key
        self._participant_id: str | None = None

    def set_participant_id(self, participant_id: str):
        """Set the participant ID for session tracking.

        Args:
            participant_id: The participant ID to use for session configuration.
        """
        self._participant_id = participant_id

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle LLM message frames.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            # Messages are accumulated on the context as a list of messages.
            # The last one by the human is the one we want to send to the LLM.
            logger.debug(f"Got transcription frame {frame}")
            messages = frame.context.get_messages()
            # Historically this processor has only handled plain-text user
            # messages; the guards below make that contract explicit for the
            # type checker. TODO: maybe handle other message shapes (provider-
            # specific messages, multi-modal content lists, etc.).
            last_message = messages[-1] if messages else None
            if not isinstance(last_message, dict):
                await self.push_frame(frame, direction)
                return
            content = last_message.get("content")
            if not isinstance(content, str):
                await self.push_frame(frame, direction)
                return

            await self._ainvoke(content.strip())
        else:
            await self.push_frame(frame, direction)

    @staticmethod
    def __get_token_value(text: str | AIMessageChunk) -> str:
        """Extract token value from various text types.

        Args:
            text: The text or message chunk to extract value from.

        Returns:
            The extracted string value.
        """
        match text:
            case str():
                return text
            case AIMessageChunk():
                # `content` is `str | list[...]` (multi-modal); stringify if
                # it's a list, since downstream consumers want plain text.
                content = text.content
                return content if isinstance(content, str) else str(content)
            case _:
                return ""

    async def _ainvoke(self, text: str):
        """Invoke the Langchain runnable with the provided text."""
        logger.debug(f"Invoking chain with {text}")
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for token in self._chain.astream(
                {self._transcript_key: text},
                config={"configurable": {"session_id": self._participant_id}},
            ):
                frame = TextFrame(self.__get_token_value(token))
                frame.includes_inter_frame_spaces = True
                await self.push_frame(frame)
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
