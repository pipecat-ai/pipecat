#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that OpenAI-compatible services report token usage once per completion.

Providers differ in how often they send usage: some once at the end, others a
cumulative snapshot on every streamed chunk. The base streaming loop holds the
latest snapshot and reports it when the completion finishes, so a single turn
produces a single usage metric either way.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.baseten.llm import BasetenLLMService
from pipecat.services.novita.llm import NovitaLLMService
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.perplexity.llm import PerplexityLLMService
from pipecat.services.sambanova.llm import SambaNovaLLMService
from pipecat.services.xai.llm import GrokLLMService

# SambaNova keeps its own copy of the streaming loop, so it is covered here
# alongside the services that inherit the base one.
SERVICES = [
    pytest.param(OpenAILLMService, {"api_key": "test-key"}, id="openai"),
    pytest.param(BasetenLLMService, {"api_key": "test-key"}, id="baseten"),
    pytest.param(GrokLLMService, {"api_key": "test-key"}, id="grok"),
    pytest.param(NovitaLLMService, {"api_key": "test-key"}, id="novita"),
    pytest.param(PerplexityLLMService, {"api_key": "test-key"}, id="perplexity"),
    pytest.param(NvidiaLLMService, {"api_key": "test-key"}, id="nvidia"),
    pytest.param(SambaNovaLLMService, {"api_key": "test-key"}, id="sambanova"),
]


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


class _FakeStream:
    """Stands in for the provider's chat completion stream.

    Satisfies the base streaming loop, which iterates and then closes the
    stream, as well as SambaNova's copy, which enters it as an async context
    manager.
    """

    def __init__(self, chunks, raise_at_end=None):
        self._chunks = list(chunks)
        self._raise_at_end = raise_at_end

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk
        if self._raise_at_end:
            raise self._raise_at_end

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def close(self):
        pass


def _service(service_class, init_kwargs, chunks, raise_at_end=None):
    """A service whose stream yields the given chunks."""
    with patch.object(service_class, "create_client"):
        service = service_class(settings=service_class.Settings(model="test-model"), **init_kwargs)
    service._client = AsyncMock()
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks, raise_at_end))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    return service


def _context():
    return LLMContext(messages=[{"role": "user", "content": "Hi"}])


@pytest.mark.parametrize(("service_class", "init_kwargs"), SERVICES)
@pytest.mark.asyncio
async def test_the_snapshots_are_reported_once_as_a_final_total(service_class, init_kwargs):
    """Three snapshots for one completion produce one report of the last."""
    service = _service(
        service_class,
        init_kwargs,
        [_usage_chunk(20, 5), _usage_chunk(20, 12), _usage_chunk(20, 30)],
    )

    with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
        await service._process_context(_context())

    reported.assert_called_once()
    usage = reported.call_args.args[0]
    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 30
    assert usage.total_tokens == 50


@pytest.mark.parametrize(("service_class", "init_kwargs"), SERVICES)
@pytest.mark.asyncio
async def test_usage_is_reported_when_the_response_is_interrupted(service_class, init_kwargs):
    """A completion cancelled mid-stream still reports the latest snapshot once."""
    service = _service(
        service_class,
        init_kwargs,
        [_usage_chunk(20, 5), _usage_chunk(20, 12)],
        raise_at_end=asyncio.CancelledError(),
    )

    with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
        with pytest.raises(asyncio.CancelledError):
            await service._process_context(_context())

    reported.assert_called_once()
    assert reported.call_args.args[0].completion_tokens == 12


@pytest.mark.parametrize(("service_class", "init_kwargs"), SERVICES)
@pytest.mark.asyncio
async def test_cached_and_reasoning_counts_reach_the_report(service_class, init_kwargs):
    """The snapshot is reported whole, so every count the provider sent survives."""
    chunk = _usage_chunk(20, 30, reasoning_tokens=8)
    chunk.usage.prompt_tokens_details.cached_tokens = 15
    service = _service(service_class, init_kwargs, [chunk])

    with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
        await service._process_context(_context())

    usage = reported.call_args.args[0]
    assert usage.cache_read_input_tokens == 15
    assert usage.reasoning_tokens == 8


@pytest.mark.parametrize(("service_class", "init_kwargs"), SERVICES)
@pytest.mark.asyncio
async def test_a_completion_without_usage_reports_nothing(service_class, init_kwargs):
    """Streams that carry no usage snapshot produce no metrics."""
    service = _service(
        service_class, init_kwargs, [SimpleNamespace(usage=None, model=None, choices=[])]
    )

    with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
        await service._process_context(_context())

    reported.assert_not_called()


@pytest.mark.parametrize(("service_class", "init_kwargs"), SERVICES)
@pytest.mark.asyncio
async def test_a_later_completion_does_not_inherit_earlier_usage(service_class, init_kwargs):
    """Each completion starts from a clean slate."""
    service = _service(service_class, init_kwargs, [_usage_chunk(20, 30)])

    with patch.object(FrameProcessor, "start_llm_usage_metrics", AsyncMock()) as reported:
        await service._process_context(_context())

        service.get_chat_completions = AsyncMock(
            return_value=_FakeStream([SimpleNamespace(usage=None, model=None, choices=[])])
        )
        await service._process_context(_context())

    reported.assert_called_once()
