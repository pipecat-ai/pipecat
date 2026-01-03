#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mock LLM Service for testing purposes."""

import asyncio
import json
import time
from typing import AsyncIterator, List, Optional

from loguru import logger
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


class MockLLMService(OpenAILLMService):
    """Mock LLM service that streams predefined ChatCompletionChunk objects.

    This service is designed for testing purposes and allows you to:
    - Stream predefined chunks instead of making API calls
    - Test chunk processing logic in _process_context
    - Verify frame generation from chunks
    - Test function calling with controlled responses
    - Support multi-step responses that cycle through on each generation
    """

    def __init__(
        self,
        *,
        mock_chunks: Optional[List[ChatCompletionChunk]] = None,
        mock_steps: Optional[List[List[ChatCompletionChunk]]] = None,
        chunk_delay: float = 0.01,
        **kwargs,
    ):
        """Initialize the mock LLM service.

        Args:
            mock_chunks: List of ChatCompletionChunk objects to stream (single step)
            mock_steps: List of chunk lists for multi-step responses. Each generation
                will use the next step's chunks. Takes precedence over mock_chunks.
            chunk_delay: Delay in seconds between streaming chunks
            **kwargs: Additional arguments passed to OpenAILLMService
        """
        # Use dummy API key and model since we're not making real API calls
        kwargs["api_key"] = kwargs.get("api_key", "mock-api-key")
        kwargs["model"] = kwargs.get("model", "mock-model")
        super().__init__(**kwargs)

        self._mock_chunks = mock_chunks or []
        self._mock_steps = mock_steps or []
        self._current_step = 0
        self._chunk_delay = chunk_delay

    def _get_current_chunks(self) -> List[ChatCompletionChunk]:
        """Get the chunks for the current step."""
        if self._mock_steps:
            if self._current_step < len(self._mock_steps):
                return self._mock_steps[self._current_step]
            # If we've exhausted steps, return empty list
            return []
        return self._mock_chunks

    def _advance_step(self) -> None:
        """Advance to the next step after a generation."""
        if self._mock_steps:
            self._current_step += 1

    def get_current_step(self) -> int:
        """Get the current step index (0-based).

        Returns:
            The current step index.
        """
        return self._current_step

    async def _stream_mock_chunks(self) -> AsyncIterator[ChatCompletionChunk]:
        """Stream the mock chunks for the current step with delays."""
        chunks = self._get_current_chunks()
        for chunk in chunks:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield chunk
        # Advance to next step after streaming all chunks
        self._advance_step()

    async def _stream_chat_completions_specific_context(
        self, context: OpenAILLMContext
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Override to return mock chunks instead of API call."""
        # The base class awaits this method, so it should return an async iterator directly
        adapter = self.get_llm_adapter()
        messages_for_log = adapter.get_messages_for_logging(context)
        logger.debug(f"{self}: Generating chat from universal context {messages_for_log}")
        return self._stream_mock_chunks()

    async def _stream_chat_completions_universal_context(
        self, context: LLMContext
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Override to return mock chunks for universal context."""
        # The base class awaits this method, so it should return an async iterator directly
        adapter = self.get_llm_adapter()
        messages_for_log = adapter.get_messages_for_logging(context)
        logger.debug(f"{self}: Generating chat from universal context {messages_for_log}")
        return self._stream_mock_chunks()

    def set_mock_chunks(self, chunks: List[ChatCompletionChunk]):
        """Update the mock chunks to stream.

        Args:
            chunks: New list of chunks to stream
        """
        self._mock_chunks = chunks

    def set_mock_steps(self, steps: List[List[ChatCompletionChunk]]):
        """Update the mock steps for multi-step responses.

        Args:
            steps: List of chunk lists, one per generation step
        """
        self._mock_steps = steps
        self._current_step = 0

    def reset_steps(self):
        """Reset the step counter to start from the beginning."""
        self._current_step = 0

    # Helper methods for creating chunks
    @staticmethod
    def create_text_chunks(text: str, chunk_size: int = 10) -> List[ChatCompletionChunk]:
        """Helper to create text streaming chunks from a string.

        Args:
            text: The text to split into chunks
            chunk_size: Maximum characters per chunk

        Returns:
            List of ChatCompletionChunk objects
        """
        chunks = []
        timestamp = int(time.time())

        # Split text into chunks
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i : i + chunk_size]
            chunk = ChatCompletionChunk(
                id="chatcmpl-mock",
                object="chat.completion.chunk",
                created=timestamp,
                model="mock-model",
                choices=[
                    Choice(delta=ChoiceDelta(content=chunk_text), index=0, finish_reason=None)
                ],
            )
            chunks.append(chunk)

        # Add final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id="chatcmpl-mock",
            object="chat.completion.chunk",
            created=timestamp,
            model="mock-model",
            choices=[Choice(delta=ChoiceDelta(), index=0, finish_reason="stop")],
        )
        chunks.append(final_chunk)

        return chunks

    @staticmethod
    def create_function_call_chunks(
        function_name: str,
        arguments: dict,
        tool_call_id: str = "call_mock123",
        index: int = 0,
        chunk_arguments: bool = True,
    ) -> List[ChatCompletionChunk]:
        """Helper to create function call chunks.

        Args:
            function_name: Name of the function to call
            arguments: Dictionary of arguments to pass
            tool_call_id: ID for the tool call
            index: Index of the function call (for multiple calls)
            chunk_arguments: If True, stream arguments in chunks

        Returns:
            List of ChatCompletionChunk objects
        """
        chunks = []
        timestamp = int(time.time())

        # First chunk: function name and tool call ID
        name_chunk = ChatCompletionChunk(
            id="chatcmpl-mock",
            object="chat.completion.chunk",
            created=timestamp,
            model="mock-model",
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=index,
                                id=tool_call_id,
                                function=ChoiceDeltaToolCallFunction(
                                    name=function_name, arguments=""
                                ),
                                type="function",
                            )
                        ]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ],
        )
        chunks.append(name_chunk)

        # Stream arguments
        args_json = json.dumps(arguments)

        if chunk_arguments:
            # Stream arguments in smaller chunks
            chunk_size = 20
            for i in range(0, len(args_json), chunk_size):
                arg_chunk_text = args_json[i : i + chunk_size]
                arg_chunk = ChatCompletionChunk(
                    id="chatcmpl-mock",
                    object="chat.completion.chunk",
                    created=timestamp,
                    model="mock-model",
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                tool_calls=[
                                    ChoiceDeltaToolCall(
                                        index=index,
                                        function=ChoiceDeltaToolCallFunction(
                                            arguments=arg_chunk_text
                                        ),
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                )
                chunks.append(arg_chunk)
        else:
            # Send all arguments in one chunk
            arg_chunk = ChatCompletionChunk(
                id="chatcmpl-mock",
                object="chat.completion.chunk",
                created=timestamp,
                model="mock-model",
                choices=[
                    Choice(
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=index,
                                    function=ChoiceDeltaToolCallFunction(arguments=args_json),
                                )
                            ]
                        ),
                        index=0,
                        finish_reason=None,
                    )
                ],
            )
            chunks.append(arg_chunk)

        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id="chatcmpl-mock",
            object="chat.completion.chunk",
            created=timestamp,
            model="mock-model",
            choices=[Choice(delta=ChoiceDelta(), index=0, finish_reason="tool_calls")],
        )
        chunks.append(final_chunk)

        return chunks

    @staticmethod
    def create_mixed_chunks(
        text: str, function_name: str, arguments: dict, tool_call_id: str = "call_mock123"
    ) -> List[ChatCompletionChunk]:
        """Helper to create chunks with both text and function calls.

        Args:
            text: Text to stream before function call
            function_name: Name of the function to call
            arguments: Dictionary of arguments to pass
            tool_call_id: ID for the tool call

        Returns:
            List of ChatCompletionChunk objects with text followed by function call
        """
        chunks = []

        # First add text chunks (without the final chunk)
        text_chunks = MockLLMService.create_text_chunks(text)
        chunks.extend(text_chunks[:-1])  # Exclude the final chunk with finish_reason

        # Then add function call chunks
        func_chunks = MockLLMService.create_function_call_chunks(
            function_name, arguments, tool_call_id
        )
        chunks.extend(func_chunks)

        return chunks

    @staticmethod
    def create_multiple_function_call_chunks(functions: List[dict]) -> List[ChatCompletionChunk]:
        """Helper to create chunks with multiple function calls.

        Args:
            functions: List of dicts with 'name', 'arguments', and optional 'tool_call_id'

        Returns:
            List of ChatCompletionChunk objects with multiple function calls
        """
        chunks = []
        timestamp = int(time.time())

        for idx, func in enumerate(functions):
            func_name = func["name"]
            func_args = func["arguments"]
            tool_id = func.get("tool_call_id", f"call_mock{idx}")

            # Create chunks for this function call
            func_chunks = MockLLMService.create_function_call_chunks(
                func_name, func_args, tool_id, index=idx, chunk_arguments=False
            )

            # Add all but the last chunk (we'll add a single final chunk at the end)
            chunks.extend(func_chunks[:-1])

        # Add final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id="chatcmpl-mock",
            object="chat.completion.chunk",
            created=timestamp,
            model="mock-model",
            choices=[Choice(delta=ChoiceDelta(), index=0, finish_reason="tool_calls")],
        )
        chunks.append(final_chunk)

        return chunks

    @staticmethod
    def create_multi_step_responses(
        first_step_chunks: List[ChatCompletionChunk],
        num_text_steps: int = 1,
        step_prefix: str = "Response",
    ) -> List[List[ChatCompletionChunk]]:
        """Create a list of chunk lists for multi-step responses.

        This helper creates a sequence of responses where the first step uses
        the provided chunks, and subsequent steps use simple text responses.

        Args:
            first_step_chunks: Chunks to use for the first step (e.g., function calls)
            num_text_steps: Number of additional text response steps to generate
            step_prefix: Prefix for generated text responses

        Returns:
            List of chunk lists, one per step
        """
        steps = [first_step_chunks]

        for i in range(num_text_steps):
            text = f"{step_prefix} {i + 1}"
            text_chunks = MockLLMService.create_text_chunks(text)
            steps.append(text_chunks)

        return steps
