#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mock LLM Service for testing purposes."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from loguru import logger
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.utils.types import assert_given


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
        mock_chunks: list[ChatCompletionChunk] | None = None,
        mock_steps: list[list[ChatCompletionChunk]] | None = None,
        mock_inference_responses: list[str] | None = None,
        chunk_delay: float = 0.01,
        **kwargs,
    ):
        """Initialize the mock LLM service.

        Args:
            mock_chunks: List of ChatCompletionChunk objects to stream (single step)
            mock_steps: List of chunk lists for multi-step responses. Each generation
                will use the next step's chunks. Takes precedence over mock_chunks.
            mock_inference_responses: List of response strings for run_inference, indexed
                by step. Each step's run_inference call returns the corresponding response.
            chunk_delay: Delay in seconds between streaming chunks
            **kwargs: Additional arguments passed to OpenAILLMService
        """
        # Use dummy API key and settings since we're not making real API calls
        kwargs["api_key"] = kwargs.get("api_key", "mock-api-key")
        if "settings" not in kwargs and "model" not in kwargs:
            kwargs["settings"] = OpenAILLMSettings(model="mock-model")
        super().__init__(**kwargs)

        self._mock_chunks = mock_chunks or []
        self._mock_steps = mock_steps or []
        self._current_step = 0
        self._chunk_delay = chunk_delay
        self._mock_inference_responses = mock_inference_responses or []

    def _get_current_chunks(self) -> list[ChatCompletionChunk]:
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
        try:
            chunks = self._get_current_chunks()
            for chunk in chunks:
                if self._chunk_delay > 0:
                    await asyncio.sleep(self._chunk_delay)
                yield chunk
            # Advance to next step after streaming all chunks
        except asyncio.CancelledError:
            logger.debug(f"CancelledError in {self}")
            raise
        finally:
            self._advance_step()

    async def get_chat_completions(  # type: ignore[override]
        self, context: LLMContext
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Override to return mock chunks instead of calling the OpenAI API.

        Mirrors the real ``BaseOpenAILLMService.get_chat_completions`` flow
        (adapter call, invocation params, parameter building) so adapter and
        param-building logic is exercised under test, then returns mock chunks
        in place of the network call.
        """
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating chat from context (mock) {adapter.get_messages_for_logging(context)}"
        )

        params_from_context: OpenAILLMInvocationParams = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )

        self.build_chat_completion_params(params_from_context)

        return self._stream_mock_chunks()

    def set_mock_chunks(self, chunks: list[ChatCompletionChunk]):
        """Update the mock chunks to stream.

        Args:
            chunks: New list of chunks to stream
        """
        self._mock_chunks = chunks

    def set_mock_steps(self, steps: list[list[ChatCompletionChunk]]):
        """Update the mock steps for multi-step responses.

        Args:
            steps: List of chunk lists, one per generation step
        """
        self._mock_steps = steps
        self._current_step = 0

    def reset_steps(self):
        """Reset the step counter to start from the beginning."""
        self._current_step = 0

    def set_mock_inference_responses(self, responses: list[str]):
        """Update the mock inference responses indexed by step.

        Args:
            responses: List of response strings, indexed by step number
        """
        self._mock_inference_responses = responses

    async def run_inference(
        self,
        context,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Override to return mock response for the current step.

        Uses the same step counter as streaming methods. Does NOT advance
        the step counter - only streaming completions advance the step.

        Args:
            context: The LLM context (ignored in mock).
            max_tokens: Optional maximum number of tokens (ignored in mock).
            system_instruction: Optional system instruction (ignored in mock).

        Returns:
            The mock inference response for the current step, or None if not set.
        """
        adapter = self.get_llm_adapter()
        messages_for_log = adapter.get_messages_for_logging(context)
        logger.debug(
            f"{self}: Mock run_inference called at step {self._current_step} with context {messages_for_log}"
        )

        if self._mock_inference_responses:
            if self._current_step < len(self._mock_inference_responses):
                return self._mock_inference_responses[self._current_step]
            # If we've exhausted responses, return None
            return None

        return None

    # Helper methods for creating chunks
    @staticmethod
    def create_text_chunks(text: str, chunk_size: int = 10) -> list[ChatCompletionChunk]:
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
    ) -> list[ChatCompletionChunk]:
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
    ) -> list[ChatCompletionChunk]:
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
    def create_multiple_function_call_chunks(functions: list[dict]) -> list[ChatCompletionChunk]:
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
        first_step_chunks: list[ChatCompletionChunk],
        num_text_steps: int = 1,
        step_prefix: str = "Response",
    ) -> list[list[ChatCompletionChunk]]:
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


class ContextCapturingMockLLM(MockLLMService):
    """A MockLLMService that snapshots the LLM context at each generation.

    Useful for tests that need to verify what the LLM sees on each completion
    (e.g. that a tool call result was added to the context, or that the system
    prompt was updated, before the next generation was triggered).
    """

    def __init__(self, *args, **kwargs):
        """Initialize and prepare the captured-contexts buffer.

        Args:
            *args: Positional arguments forwarded to MockLLMService.
            **kwargs: Keyword arguments forwarded to MockLLMService.
        """
        super().__init__(*args, **kwargs)
        self.captured_contexts: list[dict[str, Any]] = []

    async def get_chat_completions(  # type: ignore[override]
        self, context: LLMContext
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Capture context state before delegating to the parent mock stream."""
        messages_snapshot: list[dict[str, Any]] = []
        for msg in context.messages:
            msg_copy: dict[str, Any] = dict(cast(dict[str, Any], msg))
            if "content" in msg_copy:
                msg_copy["content"] = str(msg_copy["content"]) if msg_copy["content"] else None
            messages_snapshot.append(msg_copy)

        self.captured_contexts.append(
            {
                "step": self._current_step,
                "messages": messages_snapshot,
                "system_prompt": self._settings.system_instruction,
            }
        )

        return await super().get_chat_completions(context)

    def get_context_at_step(self, step: int) -> dict[str, Any] | None:
        """Get the captured context at a specific step (0-indexed)."""
        for ctx in self.captured_contexts:
            if ctx["step"] == step:
                return ctx
        return None

    def has_tool_call_result_at_step(self, step: int, function_name: str) -> bool:
        """Check if a tool call result for the given function exists at step."""
        ctx = self.get_context_at_step(step)
        if not ctx:
            return False

        for msg in ctx["messages"]:
            if msg.get("role") == "tool" and msg.get("name") == function_name:
                return True
            if msg.get("tool_call_id") and function_name in str(msg.get("name", "")):
                return True

        return False

    def get_system_prompt_at_step(self, step: int) -> str:
        """Get the system prompt that was active at a specific step."""
        ctx = self.get_context_at_step(step)
        if ctx:
            return ctx.get("system_prompt") or ""
        return ""
