#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that ``LLMTokenUsage.total_tokens`` means the same thing across services.

Services report their input count differently: OpenAI and Google count cache reads
inside ``prompt_tokens`` and hand back a total that already includes them, while
Anthropic and Bedrock report cache reads alongside an input count that excludes
them. Computing ``prompt_tokens + completion_tokens`` on the second group therefore
drops every cached token, and a consumer summing ``total_tokens`` over a session
gets a number that is quietly too low without anything raising.

The gap is not marginal for a voice agent, where the system prompt and the
conversation history are re-read from cache on every turn, so cache reads come to
dominate the input count within a few turns.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService


def _anthropic_service() -> AnthropicLLMService:
    service = AnthropicLLMService(api_key="test-key")
    service.start_llm_usage_metrics = AsyncMock()
    return service


def _bedrock_service() -> AWSBedrockLLMService:
    service = AWSBedrockLLMService(
        aws_access_key="test-key",
        aws_secret_key="test-secret",
        aws_region="us-east-1",
        settings=AWSBedrockLLMService.Settings(model="us.anthropic.claude-sonnet-4-20250514-v1:0"),
    )
    service.start_llm_usage_metrics = AsyncMock()
    return service


def _reported(service) -> object:
    """The single LLMTokenUsage handed to start_llm_usage_metrics."""
    service.start_llm_usage_metrics.assert_called_once()
    return service.start_llm_usage_metrics.call_args.args[0]


class TestAnthropicTotals:
    @pytest.mark.asyncio
    async def test_cached_reads_are_counted_in_the_total(self):
        """The turn billed 2100 input tokens; only 100 of them were uncached."""
        service = _anthropic_service()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=2000,
        )
        usage = _reported(service)
        assert usage.total_tokens == 2150

    @pytest.mark.asyncio
    async def test_cache_creation_is_counted_in_the_total(self):
        """Writing the cache is billed too, and at a premium."""
        service = _anthropic_service()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=1500,
            cache_read_input_tokens=0,
        )
        assert _reported(service).total_tokens == 1650

    @pytest.mark.asyncio
    async def test_the_components_are_still_reported_separately(self):
        """The total is normalized, so the breakdown has to stay available."""
        service = _anthropic_service()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=2000,
        )
        usage = _reported(service)
        assert usage.prompt_tokens == 100
        assert usage.cache_read_input_tokens == 2000
        assert usage.cache_creation_input_tokens == 300
        assert usage.total_tokens == 2450

    @pytest.mark.asyncio
    async def test_an_uncached_turn_is_unaffected(self):
        """No cache in play means the old arithmetic was already right."""
        service = _anthropic_service()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        assert _reported(service).total_tokens == 150


class TestBedrockTotals:
    @pytest.mark.asyncio
    async def test_cached_reads_are_counted_in_the_total(self):
        """Bedrock fronts the same Anthropic models and reports inputTokens the same way."""
        service = _bedrock_service()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_input_tokens=2000,
            cache_creation_input_tokens=300,
        )
        assert _reported(service).total_tokens == 2450

    @pytest.mark.asyncio
    async def test_a_cache_only_report_is_not_dropped(self):
        """The guard skipped reporting unless input or output was non-zero, so a
        report carrying only cache activity vanished instead of being counted."""
        service = _bedrock_service()
        await service._report_usage_metrics(
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_input_tokens=2000,
            cache_creation_input_tokens=0,
        )
        assert _reported(service).total_tokens == 2000

    @pytest.mark.asyncio
    async def test_an_empty_report_is_still_skipped(self):
        """Widening the guard must not start emitting all-zero usage."""
        service = _bedrock_service()
        await service._report_usage_metrics(
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        service.start_llm_usage_metrics.assert_not_called()
