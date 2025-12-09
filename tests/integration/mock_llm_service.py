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
    """

    def __init__(
        self,
        *,
        mock_chunks: Optional[List[ChatCompletionChunk]] = None,
        chunk_delay: float = 0.01,
        **kwargs
    ):
        """Initialize the mock LLM service.

        Args:
            mock_chunks: List of ChatCompletionChunk objects to stream
            chunk_delay: Delay in seconds between streaming chunks
            **kwargs: Additional arguments passed to OpenAILLMService
        """
        # Use dummy API key and model since we're not making real API calls
        kwargs['api_key'] = kwargs.get('api_key', 'mock-api-key')
        kwargs['model'] = kwargs.get('model', 'mock-model')
        super().__init__(**kwargs)

        self._mock_chunks = mock_chunks or []
        self._chunk_delay = chunk_delay

    async def _stream_mock_chunks(self) -> AsyncIterator[ChatCompletionChunk]:
        """Stream the mock chunks with delays."""
        for chunk in self._mock_chunks:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield chunk

    async def _stream_chat_completions_specific_context(
        self, context: OpenAILLMContext
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Override to return mock chunks instead of API call."""
        # The base class awaits this method, so it should return an async iterator directly
        return self._stream_mock_chunks()

    async def _stream_chat_completions_universal_context(
        self, context: LLMContext
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Override to return mock chunks for universal context."""
        # The base class awaits this method, so it should return an async iterator directly
        return self._stream_mock_chunks()

    def set_mock_chunks(self, chunks: List[ChatCompletionChunk]):
        """Update the mock chunks to stream.

        Args:
            chunks: New list of chunks to stream
        """
        self._mock_chunks = chunks

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
            chunk_text = text[i:i + chunk_size]
            chunk = ChatCompletionChunk(
                id="chatcmpl-mock",
                object="chat.completion.chunk",
                created=timestamp,
                model="mock-model",
                choices=[
                    Choice(
                        delta=ChoiceDelta(content=chunk_text),
                        index=0,
                        finish_reason=None
                    )
                ]
            )
            chunks.append(chunk)

        # Add final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id="chatcmpl-mock",
            object="chat.completion.chunk",
            created=timestamp,
            model="mock-model",
            choices=[
                Choice(
                    delta=ChoiceDelta(),
                    index=0,
                    finish_reason="stop"
                )
            ]
        )
        chunks.append(final_chunk)

        return chunks

    @staticmethod
    def create_function_call_chunks(
        function_name: str,
        arguments: dict,
        tool_call_id: str = "call_mock123",
        index: int = 0,
        chunk_arguments: bool = True
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
                                    name=function_name,
                                    arguments=""
                                ),
                                type="function"
                            )
                        ]
                    ),
                    index=0,
                    finish_reason=None
                )
            ]
        )
        chunks.append(name_chunk)

        # Stream arguments
        args_json = json.dumps(arguments)

        if chunk_arguments:
            # Stream arguments in smaller chunks
            chunk_size = 20
            for i in range(0, len(args_json), chunk_size):
                arg_chunk_text = args_json[i:i + chunk_size]
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
                                        )
                                    )
                                ]
                            ),
                            index=0,
                            finish_reason=None
                        )
                    ]
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
                                    function=ChoiceDeltaToolCallFunction(
                                        arguments=args_json
                                    )
                                )
                            ]
                        ),
                        index=0,
                        finish_reason=None
                    )
                ]
            )
            chunks.append(arg_chunk)

        # Final chunk with finish_reason
        final_chunk = ChatCompletionChunk(
            id="chatcmpl-mock",
            object="chat.completion.chunk",
            created=timestamp,
            model="mock-model",
            choices=[
                Choice(
                    delta=ChoiceDelta(),
                    index=0,
                    finish_reason="tool_calls"
                )
            ]
        )
        chunks.append(final_chunk)

        return chunks

    @staticmethod
    def create_mixed_chunks(
        text: str,
        function_name: str,
        arguments: dict,
        tool_call_id: str = "call_mock123"
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
    def create_multiple_function_call_chunks(
        functions: List[dict]
    ) -> List[ChatCompletionChunk]:
        """Helper to create chunks with multiple function calls.

        Args:
            functions: List of dicts with 'name', 'arguments', and optional 'tool_call_id'

        Returns:
            List of ChatCompletionChunk objects with multiple function calls
        """
        chunks = []
        timestamp = int(time.time())

        for idx, func in enumerate(functions):
            func_name = func['name']
            func_args = func['arguments']
            tool_id = func.get('tool_call_id', f"call_mock{idx}")

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
            choices=[
                Choice(
                    delta=ChoiceDelta(),
                    index=0,
                    finish_reason="tool_calls"
                )
            ]
        )
        chunks.append(final_chunk)

        return chunks