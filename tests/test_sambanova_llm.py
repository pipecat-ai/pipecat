#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for SambaNova LLM service."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.sambanova.llm import SambaNovaLLMService


@pytest.mark.asyncio
async def test_sambanova_llm_stream_closed_on_cancellation():
    """Test that the stream is closed when CancelledError occurs during iteration.

    This prevents socket leaks when the pipeline is interrupted (e.g., user interruption).
    See issue #3639.
    """
    with patch.object(SambaNovaLLMService, "create_client"):
        service = SambaNovaLLMService(api_key="test-key", model="test-model")
        service._client = AsyncMock()

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
                    raise asyncio.CancelledError()
                mock_chunk = AsyncMock()
                mock_chunk.usage = None
                mock_chunk.choices = []
                return mock_chunk

        mock_stream = MockAsyncStream()

        service._stream_chat_completions_specific_context = AsyncMock(return_value=mock_stream)
        service._stream_chat_completions_universal_context = AsyncMock(return_value=mock_stream)
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        with pytest.raises(asyncio.CancelledError):
            await service._process_context(context)

        assert stream_closed, "Stream should be closed even when CancelledError occurs"
