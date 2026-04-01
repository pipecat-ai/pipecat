#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Novita LLM service."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.novita.llm import NovitaLLMService


@pytest.mark.asyncio
async def test_novita_llm_stream_closed_on_cancellation():
    """Test that the stream is closed when CancelledError occurs during iteration.

    This prevents socket leaks when the pipeline is interrupted (e.g., user interruption).
    """
    with patch.object(NovitaLLMService, "create_client"):
        service = NovitaLLMService(api_key="test-key", model="test-model")
        service._client = AsyncMock()

        stream_closed = False

        class MockAsyncStream:
            """Mock AsyncStream that tracks close() calls and raises CancelledError."""

            def __init__(self):
                self.iteration_count = 0

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

            async def close(self):
                nonlocal stream_closed
                stream_closed = True

        mock_stream = MockAsyncStream()

        service.get_chat_completions = AsyncMock(return_value=mock_stream)
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        service.start_llm_usage_metrics = AsyncMock()

        context = LLMContext(
            messages=[{"role": "user", "content": "Hello"}],
        )

        with pytest.raises(asyncio.CancelledError):
            await service._process_context(context)

        assert stream_closed, "Stream should be closed even when CancelledError occurs"
