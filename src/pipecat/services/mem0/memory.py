#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mem0 memory service integration for Pipecat.

This module provides a memory service that integrates with Mem0 to store
and retrieve conversational memories, enhancing LLM context with relevant
historical information.
"""

import asyncio
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from mem0 import Memory, MemoryClient  # noqa: F401
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
    Supports both local and cloud-based Mem0 configurations.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Mem0 memory service.

        Parameters:
            search_limit: Maximum number of memories to retrieve per query.
            search_threshold: Minimum similarity threshold for memory retrieval.
            api_version: API version to use for Mem0 client operations.
            system_prompt: Prefix text for memory context messages.
            add_as_system_message: Whether to add memories as system messages.
            position: Position to insert memory messages in context.
        """

        search_limit: int = Field(default=10, ge=1)
        search_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
        api_version: str = Field(default="v2")
        system_prompt: str = Field(default="Based on previous conversations, I recall: \n\n")
        add_as_system_message: bool = Field(default=True)
        position: int = Field(default=1)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        local_config: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        params: InputParams | None = None,
        host: str | None = None,
    ):
        """Initialize the Mem0 memory service.

        Args:
            api_key: The API key for accessing Mem0's cloud API.
            local_config: Local configuration for Mem0 client (alternative to cloud API).
            user_id: The user ID to associate with memories in Mem0.
            agent_id: The agent ID to associate with memories in Mem0.
            run_id: The run ID to associate with memories in Mem0.
            params: Configuration parameters for memory retrieval and storage.
            host: The host of the Mem0 server.

        Raises:
            ValueError: If none of user_id, agent_id, or run_id are provided.
        """
        # Important: Call the parent class __init__ first
        super().__init__()

        local_config = local_config or {}
        params = params or Mem0MemoryService.InputParams()

        if local_config:
            self.memory_client = Memory.from_config(local_config)
        else:
            self.memory_client = MemoryClient(api_key=api_key, host=host)
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

    async def get_memories(self) -> list[dict[str, Any]]:
        """Retrieve all stored memories for the configured user/agent/run IDs.

        This is a convenience method for accessing memories outside the pipeline,
        e.g. to build a personalized greeting at connection time. It wraps the
        blocking Mem0 ``get_all()`` call in a background thread.

        Returns:
            List of memory dictionaries. Each dict contains at least a
            ``"memory"`` key with the memory text. Returns an empty list on
            error.
        """
        try:
            if isinstance(self.memory_client, Memory):
                params = {
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                    "run_id": self.run_id,
                }
                params = {k: v for k, v in params.items() if v is not None}
                memories = await asyncio.to_thread(lambda: self.memory_client.get_all(**params))
            else:
                id_pairs = [
                    ("user_id", self.user_id),
                    ("agent_id", self.agent_id),
                    ("run_id", self.run_id),
                ]
                clauses = [{name: value} for name, value in id_pairs if value is not None]
                filters = {"OR": clauses} if clauses else {}
                memories = await asyncio.to_thread(
                    lambda: self.memory_client.get_all(filters=filters)
                )

            results = memories.get("results", []) if isinstance(memories, dict) else memories
            return results
        except Exception as e:
            logger.error(f"Error retrieving memories from Mem0: {e}")
            return []

    async def _store_messages(self, messages: list[dict[str, Any]]):
        """Store messages in Mem0.

        Runs the blocking Mem0 API call in a background thread to avoid
        blocking the event loop.

        Args:
            messages: List of message dictionaries to store in memory.
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

            if isinstance(self.memory_client, Memory):
                del params["output_format"]
            await asyncio.to_thread(lambda: self.memory_client.add(**params))
        except Exception as e:
            logger.error(f"Error storing messages in Mem0: {e}")

    async def _retrieve_memories(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant memories from Mem0.

        Runs the blocking Mem0 API call in a background thread to avoid
        blocking the event loop.

        Args:
            query: The query to search for relevant memories.

        Returns:
            List of relevant memory dictionaries matching the query.
        """
        try:
            logger.debug(f"Retrieving memories for query: {query}")
            if isinstance(self.memory_client, Memory):
                params = {
                    "query": query,
                    "user_id": self.user_id,
                    "agent_id": self.agent_id,
                    "run_id": self.run_id,
                    "limit": self.search_limit,
                }
                params = {k: v for k, v in params.items() if v is not None}
                results = await asyncio.to_thread(lambda: self.memory_client.search(**params))
            else:
                id_pairs = [
                    ("user_id", self.user_id),
                    ("agent_id", self.agent_id),
                    ("run_id", self.run_id),
                ]
                clauses = [{name: value} for name, value in id_pairs if value is not None]
                filters = {"OR": clauses} if clauses else {}
                results = await asyncio.to_thread(
                    lambda: self.memory_client.search(
                        query=query,
                        filters=filters,
                        version=self.api_version,
                        top_k=self.search_limit,
                        threshold=self.search_threshold,
                        output_format="v1.1",
                    )
                )

            logger.debug(f"Retrieved {len(results)} memories from Mem0")
            return results
        except Exception as e:
            logger.error(f"Error retrieving memories from Mem0: {e}")
            return []

    async def _enhance_context_with_memories(self, context: LLMContext, query: str):
        """Enhance the LLM context with relevant memories.

        Args:
            context: The LLM context to enhance with memory information.
            query: The query to search for relevant memories.
        """
        # Skip if this is the same query we just processed
        if self.last_query == query:
            return

        self.last_query = query

        memories = await self._retrieve_memories(query)
        if not memories:
            return

        # Format memories as a message
        memory_text = self.system_prompt
        for i, memory in enumerate(memories["results"], 1):
            memory_text += f"{i}. {memory.get('memory', '')}\n\n"

        # Add memories as a system message or user message based on configuration
        role = "system" if self.add_as_system_message else "user"
        memory_message = {"role": role, "content": memory_text}

        messages = context.get_messages()
        position = max(0, min(self.position, len(messages)))
        messages.insert(position, memory_message)
        context.set_messages(messages)

        logger.debug(f"Enhanced context with {len(memories)} memories")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, intercept context frames for memory integration.

        Args:
            frame: The incoming frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            context = frame.context
            try:
                # Get the latest user message to use as a query for memory retrieval
                context_messages = context.get_messages()
                latest_user_message = None

                for message in reversed(context_messages):
                    if message.get("role") == "user" and isinstance(message.get("content"), str):
                        latest_user_message = message.get("content")
                        break

                if latest_user_message:
                    # Filter to only user/assistant messages — Mem0 API
                    # doesn't accept other roles (system, developer, etc.)
                    messages_to_store = [
                        m for m in context_messages if m.get("role") in ("user", "assistant")
                    ]
                    # Enhance context with memories before passing it downstream
                    await self._enhance_context_with_memories(context, latest_user_message)
                    # Store the conversation in Mem0 as a background task
                    self.create_task(self._store_messages(messages_to_store), name="mem0_store")

                # Pass the enhanced context frame downstream
                await self.push_frame(frame)
            except Exception as e:
                await self.push_error(
                    error_msg=f"Error processing with Mem0: {str(e)}", exception=e
                )
                await self.push_frame(frame)  # Still pass the original frame through
        else:
            await self.push_frame(frame, direction)
