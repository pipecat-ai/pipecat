#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import ErrorFrame, Frame, LLMMessagesFrame
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from mem0 import MemoryClient  # noqa: F401
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Mem0, you need to `pip install mem0ai`. Also, set the environment variable MEM0_API_KEY."
    )
    raise Exception(f"Missing module: {e}")


class Mem0MemoryService(FrameProcessor):
    """A standalone memory service that integrates with Mem0.

    This service intercepts message frames in the pipeline, stores them in Mem0,
    and enhances context with relevant memories before passing them downstream.

    Args:
        api_key (str): The API key for accessing Mem0's API
        user_id (str): The user ID to associate with memories in Mem0
        params (InputParams, optional): Configuration parameters for memory retrieval
    """

    class InputParams(BaseModel):
        search_limit: int = Field(default=10, ge=1)
        search_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
        api_version: str = Field(default="v2")
        system_prompt: str = Field(default="Based on previous conversations, I recall: \n\n")
        add_as_system_message: bool = Field(default=True)
        position: int = Field(default=1)

    def __init__(
        self,
        *,
        api_key: str,
        user_id: str = None,
        agent_id: str = None,
        run_id: str = None,
        params: InputParams = InputParams(),
    ):
        # Important: Call the parent class __init__ first
        super().__init__()

        self.memory_client = MemoryClient(api_key=api_key)
        # At least one of user_id, agent_id, or run_id must be provided
        if not any([user_id, agent_id, run_id]):
            raise ValueError("At least one of user_id, agent_id, or run_id must be provided")

        self.user_id = user_id
        self.agent_id = agent_id
        self.run_id = run_id
        self.search_limit = params.search_limit
        self.search_threshold = params.search_threshold
        self.api_version = params.api_version
        self.system_prompt = params.system_prompt
        self.add_as_system_message = params.add_as_system_message
        self.position = params.position
        self.last_query = None
        logger.info(f"Initialized Mem0MemoryService with {user_id=}, {agent_id=}, {run_id=}")

    def _store_messages(self, messages: List[Dict[str, Any]]):
        """Store messages in Mem0.

        Args:
            messages: List of message dictionaries to store
        """
        try:
            logger.debug(f"Storing {len(messages)} messages in Mem0")
            params = {
                "messages": messages,
                "metadata": {"platform": "pipecat"},
                "output_format": "v1.1",
            }
            for id in ["user_id", "agent_id", "run_id"]:
                if getattr(self, id):
                    params[id] = getattr(self, id)
            # Note: You can run this in background to avoid blocking the conversation
            self.memory_client.add(**params)
        except Exception as e:
            logger.error(f"Error storing messages in Mem0: {e}")

    def _retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from Mem0.

        Args:
            query: The query to search for relevant memories

        Returns:
            List of relevant memory dictionaries
        """
        try:
            logger.debug(f"Retrieving memories for query: {query}")
            id_pairs = [
                ("user_id", self.user_id),
                ("agent_id", self.agent_id),
                ("run_id", self.run_id),
            ]
            clauses = [{name: value} for name, value in id_pairs if value is not None]
            filters = {"AND": clauses} if clauses else {}
            results = self.memory_client.search(
                query=query,
                filters=filters,
                version=self.api_version,
                top_k=self.search_limit,
                threshold=self.search_threshold,
            )

            logger.debug(f"Retrieved {len(results)} memories from Mem0")
            return results
        except Exception as e:
            logger.error(f"Error retrieving memories from Mem0: {e}")
            return []

    def _enhance_context_with_memories(self, context: OpenAILLMContext, query: str):
        """Enhance the LLM context with relevant memories.

        Args:
            context: The OpenAILLMContext to enhance
            query: The query to search for relevant memories
        """
        # Skip if this is the same query we just processed
        if self.last_query == query:
            return

        self.last_query = query

        memories = self._retrieve_memories(query)
        if not memories:
            return

        # Format memories as a message
        memory_text = self.system_prompt
        for i, memory in enumerate(memories, 1):
            memory_text += f"{i}. {memory.get('memory', '')}\n\n"

        # Add memories as a system message or user message based on configuration
        if self.add_as_system_message:
            context.add_message({"role": "system", "content": memory_text})
        else:
            # Add as a user message that provides context
            context.add_message({"role": "user", "content": memory_text})
        logger.debug(f"Enhanced context with {len(memories)} memories")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, intercept context frames for memory integration.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        context = None
        messages = None

        if isinstance(frame, OpenAILLMContextFrame):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            context = OpenAILLMContext.from_messages(messages)

        if context:
            try:
                # Get the latest user message to use as a query for memory retrieval
                context_messages = context.get_messages()
                latest_user_message = None

                for message in reversed(context_messages):
                    if message.get("role") == "user" and isinstance(message.get("content"), str):
                        latest_user_message = message.get("content")
                        break

                if latest_user_message:
                    # Enhance context with memories before passing it downstream
                    self._enhance_context_with_memories(context, latest_user_message)
                    # Store the conversation in Mem0. Only call this when user message is detected
                    self._store_messages(context_messages)

                # If we received an LLMMessagesFrame, create a new one with the enhanced messages
                if messages is not None:
                    await self.push_frame(LLMMessagesFrame(context.get_messages()))
                else:
                    # Otherwise, pass the enhanced context frame downstream
                    await self.push_frame(frame)
            except Exception as e:
                logger.error(f"Error processing with Mem0: {str(e)}")
                await self.push_frame(ErrorFrame(f"Error processing with Mem0: {str(e)}"))
                await self.push_frame(frame)  # Still pass the original frame through
        else:
            # For non-context frames, just pass them through
            await self.push_frame(frame, direction)
