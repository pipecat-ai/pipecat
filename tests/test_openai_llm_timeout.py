#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for OpenAI LLM error handling."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
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
async def test_openai_llm_stream_closed_on_cancellation():
    """Test that the stream is closed when CancelledError occurs during iteration.

    This prevents socket leaks when the pipeline is interrupted (e.g., user interruption).
    See issue #3589.
    """
    import asyncio

    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(model="gpt-4")
        service._client = AsyncMock()

        # Track if close was called
        stream_closed = False

        class MockAsyncStream:
            """Mock AsyncStream that tracks close() calls and raises CancelledError."""

            def __init__(self):
                self.iteration_count = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                nonlocal stream_closed
                stream_closed = True
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                self.iteration_count += 1
                if self.iteration_count > 1:
                    # Simulate cancellation during iteration
                    raise asyncio.CancelledError()
                # Return a minimal chunk for first iteration
                mock_chunk = AsyncMock()
                mock_chunk.usage = None
                mock_chunk.model = None
                mock_chunk.choices = []
                return mock_chunk

        mock_stream = MockAsyncStream()

        # Mock the stream creation methods
        service._stream_chat_completions_specific_context = AsyncMock(return_value=mock_stream)
        service._stream_chat_completions_universal_context = AsyncMock(return_value=mock_stream)
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Process context should raise CancelledError but stream should still be closed
        with pytest.raises(asyncio.CancelledError):
            await service._process_context(context)

        # Verify stream was closed despite the cancellation
        assert stream_closed, "Stream should be closed even when CancelledError occurs"


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
