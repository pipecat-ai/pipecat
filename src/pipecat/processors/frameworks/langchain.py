#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Langchain integration processor for Pipecat."""

from typing import Optional, Union

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from langchain_core.messages import AIMessageChunk
    from langchain_core.runnables import Runnable
except ModuleNotFoundError as e:
    logger.exception("In order to use Langchain, you need to `pip install pipecat-ai[langchain]`. ")
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
        self._participant_id: Optional[str] = None

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

        if isinstance(frame, OpenAILLMContextFrame):
            # Messages are accumulated on the context as a list of messages.
            # The last one by the human is the one we want to send to the LLM.
            logger.debug(f"Got transcription frame {frame}")
            text: str = frame.context.messages[-1]["content"]

            await self._ainvoke(text.strip())
        else:
            await self.push_frame(frame, direction)

    @staticmethod
    def __get_token_value(text: Union[str, AIMessageChunk]) -> str:
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
                return text.content
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
                await self.push_frame(TextFrame(self.__get_token_value(token)))
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
