#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Novita LLM service."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.novita.llm import NovitaLLMService


def _usage_chunk(prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0):
    """Build a stream chunk carrying a cumulative usage snapshot."""
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
        model=None,
        choices=[],
    )


@pytest.mark.asyncio
async def test_novita_llm_stream_closed_on_cancellation():
    """Test that the stream is closed when CancelledError occurs during iteration.

    This prevents socket leaks when the pipeline is interrupted (e.g., user interruption).
    """
    with patch.object(NovitaLLMService, "create_client"):
        service = NovitaLLMService(
            api_key="test-key", settings=NovitaLLMService.Settings(model="test-model")
        )
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


class TestCumulativeTokenUsage:
    """Novita repeats a cumulative usage snapshot on every streamed chunk."""

    @staticmethod
    def _service(chunks):
        """A service whose stream yields the given chunks."""
        with patch.object(NovitaLLMService, "create_client"):
            service = NovitaLLMService(
                api_key="test-key", settings=NovitaLLMService.Settings(model="test-model")
            )
        service._client = AsyncMock()

        async def stream():
            for chunk in chunks:
                yield chunk

        service.get_chat_completions = AsyncMock(return_value=stream())
        service.start_ttfb_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_the_snapshots_are_reported_once_as_a_final_total(self):
        """Three snapshots for one completion produce one report of the last."""
        service = self._service(
            [_usage_chunk(20, 5), _usage_chunk(20, 12), _usage_chunk(20, 30, reasoning_tokens=8)]
        )

        with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
            await service._process_context(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

        reported.assert_called_once()
        usage = reported.call_args.args[0]
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 50
        assert usage.reasoning_tokens == 8

    @pytest.mark.asyncio
    async def test_usage_is_reported_when_the_response_is_interrupted(self):
        """A completion cancelled mid-stream still reports what it consumed."""
        service = self._service([])

        async def interrupted_stream():
            yield _usage_chunk(20, 5)
            yield _usage_chunk(20, 12)
            raise asyncio.CancelledError()

        service.get_chat_completions = AsyncMock(return_value=interrupted_stream())

        with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
            with pytest.raises(asyncio.CancelledError):
                await service._process_context(
                    LLMContext(messages=[{"role": "user", "content": "Hi"}])
                )

        reported.assert_called_once()
        assert reported.call_args.args[0].completion_tokens == 12

    @pytest.mark.asyncio
    async def test_a_completion_without_usage_reports_nothing(self):
        """Streams that carry no usage snapshot produce no metrics."""
        service = self._service([SimpleNamespace(usage=None, model=None, choices=[])])

        with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
            await service._process_context(LLMContext(messages=[{"role": "user", "content": "Hi"}]))

        reported.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_later_completion_does_not_inherit_earlier_usage(self):
        """Each completion starts from a clean slate."""
        service = self._service([_usage_chunk(20, 30)])
        context = LLMContext(messages=[{"role": "user", "content": "Hi"}])

        with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
            await service._process_context(context)

            async def empty_stream():
                yield SimpleNamespace(usage=None, model=None, choices=[])

            service.get_chat_completions = AsyncMock(return_value=empty_stream())
            await service._process_context(context)

        reported.assert_called_once()
