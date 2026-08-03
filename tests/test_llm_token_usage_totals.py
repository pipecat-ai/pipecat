#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that ``LLMTokenUsage.total_tokens`` counts cached input tokens.

Anthropic and Bedrock report cache reads and cache writes alongside an input count
that excludes them, so their totals are summed from all four component counts. On
services whose provider supplies the total, cached tokens are already inside it, so
``total_tokens`` carries the same meaning everywhere.
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
    """Anthropic totals cover the cache reads and writes reported beside the input count."""

    @pytest.mark.asyncio
    async def test_cached_reads_are_counted_in_the_total(self):
        """A turn billed for 2100 input tokens, only 100 of them uncached."""
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
    async def test_the_components_are_reported_separately(self):
        """The total is the gross figure, so the breakdown stays available beside it."""
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
    async def test_an_uncached_turn_counts_input_plus_output(self):
        """With no cache activity the total is just the input and output counts."""
        service = _anthropic_service()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        assert _reported(service).total_tokens == 150


class TestBedrockTotals:
    """Bedrock reports inputTokens net of the cache, so its totals are summed the same way."""

    @pytest.mark.asyncio
    async def test_cached_reads_are_counted_in_the_total(self):
        """A turn carrying both cache reads and cache writes."""
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
        """Cache activity alone is enough to report usage."""
        service = _bedrock_service()
        await service._report_usage_metrics(
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_input_tokens=2000,
            cache_creation_input_tokens=0,
        )
        assert _reported(service).total_tokens == 2000

    @pytest.mark.asyncio
    async def test_an_empty_report_is_skipped(self):
        """An all-zero report produces no usage metrics."""
        service = _bedrock_service()
        await service._report_usage_metrics(
            prompt_tokens=0,
            completion_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        service.start_llm_usage_metrics.assert_not_called()
