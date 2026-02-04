#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for OpenAI LLM service."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pipecat.frames.frames import (
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService


@pytest.mark.asyncio
async def test_openai_llm_emits_error_frame_on_timeout():
    """Test that OpenAI LLM service emits ErrorFrame when a timeout occurs.

    This enables LLMSwitcher to trigger failover to backup LLMs when the
    primary LLM times out.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-4")
        service._client = AsyncMock()

        # Track pushed frames and errors
        pushed_frames = []
        pushed_errors = []
        timeout_handler_called = False

        original_push_frame = service.push_frame

        async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            pushed_frames.append(frame)
            await original_push_frame(frame, direction)

        async def mock_push_error(error_msg, exception=None):
            pushed_errors.append({"error_msg": error_msg, "exception": exception})

        async def mock_timeout_handler(event_name):
            nonlocal timeout_handler_called
            if event_name == "on_completion_timeout":
                timeout_handler_called = True

        service.push_frame = mock_push_frame
        service.push_error = mock_push_error
        service._call_event_handler = AsyncMock(side_effect=mock_timeout_handler)

        # Mock _process_context to raise TimeoutException
        service._process_context = AsyncMock(
            side_effect=httpx.TimeoutException("Connection timed out")
        )

        # Mock metrics methods
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()

        # Create a context frame to process
        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )
        frame = LLMContextFrame(context=context)

        # Process the frame
        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify timeout handler was called
        service._call_event_handler.assert_called_once_with("on_completion_timeout")
        assert timeout_handler_called

        # Verify push_error was called with correct message
        assert len(pushed_errors) == 1
        assert pushed_errors[0]["error_msg"] == "LLM completion timeout"
        assert isinstance(pushed_errors[0]["exception"], httpx.TimeoutException)

        # Verify LLMFullResponseStartFrame and LLMFullResponseEndFrame were pushed
        frame_types = [type(f) for f in pushed_frames]
        assert LLMFullResponseStartFrame in frame_types
        assert LLMFullResponseEndFrame in frame_types


@pytest.mark.asyncio
async def test_openai_llm_timeout_still_pushes_end_frame():
    """Test that LLMFullResponseEndFrame is pushed even when timeout occurs.

    The finally block should ensure proper cleanup regardless of timeout.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-4")
        service._client = AsyncMock()

        pushed_frames = []

        async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            pushed_frames.append(frame)

        service.push_frame = mock_push_frame
        service.push_error = AsyncMock()
        service._call_event_handler = AsyncMock()
        service._process_context = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )
        frame = LLMContextFrame(context=context)

        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify both start and end frames are pushed
        frame_types = [type(f) for f in pushed_frames]
        assert LLMFullResponseStartFrame in frame_types
        assert LLMFullResponseEndFrame in frame_types

        # Verify metrics were stopped
        service.stop_processing_metrics.assert_called_once()


@pytest.mark.asyncio
async def test_openai_llm_emits_error_frame_on_exception():
    """Test that OpenAI LLM service emits ErrorFrame when a general exception occurs.

    This enables proper error handling for API errors, rate limits, and other failures.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-4")
        service._client = AsyncMock()

        pushed_errors = []

        async def mock_push_error(error_msg, exception=None):
            pushed_errors.append({"error_msg": error_msg, "exception": exception})

        service.push_frame = AsyncMock()
        service.push_error = mock_push_error
        service._call_event_handler = AsyncMock()
        service._process_context = AsyncMock(side_effect=RuntimeError("API Error"))
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )
        frame = LLMContextFrame(context=context)

        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Verify push_error was called with correct message
        assert len(pushed_errors) == 1
        assert "Error during completion" in pushed_errors[0]["error_msg"]
        assert "API Error" in pushed_errors[0]["error_msg"]
        assert isinstance(pushed_errors[0]["exception"], RuntimeError)


def _create_mock_chunk(content=None, tool_call_id=None, tool_call_name=None, tool_call_args=None):
    """Create a mock ChatCompletionChunk for testing."""
    chunk = MagicMock()
    chunk.usage = None
    chunk.model = "gpt-4"
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()

    if content is not None:
        chunk.choices[0].delta.content = content
        chunk.choices[0].delta.tool_calls = None
    elif tool_call_id is not None:
        chunk.choices[0].delta.content = None
        tool_call = MagicMock()
        tool_call.index = 0
        tool_call.id = tool_call_id
        tool_call.function = MagicMock()
        tool_call.function.name = tool_call_name
        tool_call.function.arguments = tool_call_args
        chunk.choices[0].delta.tool_calls = [tool_call]
    else:
        chunk.choices[0].delta.content = None
        chunk.choices[0].delta.tool_calls = None

    return chunk


async def _mock_stream(chunks):
    """Create an async generator from a list of chunks."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_openai_llm_text_before_tool_call_frame_order():
    """Test that LLMFullResponseEndFrame is pushed BEFORE FunctionCallsStartedFrame.

    When the LLM returns both text content and tool calls, the frame order must be:
    1. LLMFullResponseStartFrame
    2. LLMTextFrame(s) - text content as it streams
    3. LLMFullResponseEndFrame - flushes text to context
    4. FunctionCallsStartedFrame - tool call processing begins

    This ensures the aggregator adds assistant text to context before tool calls
    are processed, fixing issue #3631.
    """
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-4")
        service._client = AsyncMock()

        # Track pushed frames in order
        pushed_frames = []

        async def mock_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            pushed_frames.append(frame)

        service.push_frame = mock_push_frame
        service._call_event_handler = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()

        # Track when run_function_calls is called relative to other frames
        run_function_calls_called_at_index = None
        original_run_function_calls = service.run_function_calls

        async def mock_run_function_calls(function_calls):
            nonlocal run_function_calls_called_at_index
            # Record the index when this is called (after frames already pushed)
            run_function_calls_called_at_index = len(pushed_frames)
            # Push a marker frame so we can verify ordering
            pushed_frames.append(FunctionCallsStartedFrame(function_calls=function_calls))

        service.run_function_calls = mock_run_function_calls

        # Create mock chunks simulating: text content followed by tool call
        chunks = [
            _create_mock_chunk(content="Let me "),
            _create_mock_chunk(content="check the weather."),
            _create_mock_chunk(
                tool_call_id="call_123",
                tool_call_name="get_weather",
                tool_call_args='{"location": "LA"}',
            ),
        ]

        # Mock the streaming method to return our chunks
        service._stream_chat_completions_universal_context = AsyncMock(
            return_value=_mock_stream(chunks)
        )

        # Create context and process
        context = LLMContext(messages=[{"role": "user", "content": "What's the weather?"}])
        frame = LLMContextFrame(context=context)

        await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        # Find indices of key frames
        text_indices = [i for i, f in enumerate(pushed_frames) if isinstance(f, LLMTextFrame)]
        end_frame_indices = [
            i for i, f in enumerate(pushed_frames) if isinstance(f, LLMFullResponseEndFrame)
        ]
        function_started_indices = [
            i for i, f in enumerate(pushed_frames) if isinstance(f, FunctionCallsStartedFrame)
        ]

        # Verify all expected frames are present
        assert len(text_indices) > 0, "Expected LLMTextFrame(s) to be pushed"
        assert len(end_frame_indices) == 1, "Expected exactly one LLMFullResponseEndFrame"
        assert len(function_started_indices) == 1, "Expected exactly one FunctionCallsStartedFrame"

        # Verify ordering: all text frames before end frame, end frame before function started
        last_text_index = max(text_indices)
        end_frame_index = end_frame_indices[0]
        function_started_index = function_started_indices[0]

        assert last_text_index < end_frame_index, (
            f"LLMTextFrame (index {last_text_index}) should come before "
            f"LLMFullResponseEndFrame (index {end_frame_index})"
        )
        assert end_frame_index < function_started_index, (
            f"LLMFullResponseEndFrame (index {end_frame_index}) should come before "
            f"FunctionCallsStartedFrame (index {function_started_index})"
        )
