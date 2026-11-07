#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that the ``gen_ai.usage.input_tokens`` span attribute counts cached input tokens.

The OpenTelemetry GenAI conventions define ``gen_ai.usage.input_tokens`` as every
input token, cached ones included. Anthropic and Bedrock report ``prompt_tokens`` net
of the cache, so the span adds the cache counts back for them. Services that report it
gross keep their count.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.utils.tracing.service_decorators import _add_token_usage_to_span


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _span_attributes(token_usage) -> dict:
    """The attributes ``_add_token_usage_to_span`` sets for one usage report."""
    span = _FakeSpan()
    _add_token_usage_to_span(span, token_usage)
    return span.attributes


def _reported(service) -> LLMTokenUsage:
    """The single LLMTokenUsage handed to start_llm_usage_metrics."""
    service.start_llm_usage_metrics.assert_called_once()
    return service.start_llm_usage_metrics.call_args.args[0]


# A turn billed for 2400 input tokens, only 100 of them uncached.
NET_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 2450,
    "cache_read_input_tokens": 2000,
    "cache_creation_input_tokens": 300,
}


class TestNetReportingUsage:
    """Usage whose ``prompt_tokens`` leaves the cache out, as Anthropic and Bedrock report it."""

    def test_cache_reads_and_writes_are_counted_in_input_tokens(self):
        attrs = _span_attributes(LLMTokenUsage(**NET_USAGE))
        assert attrs["gen_ai.usage.input_tokens"] == 2400
        assert attrs["gen_ai.usage.output_tokens"] == 50
        assert attrs["gen_ai.usage.cache_read.input_tokens"] == 2000
        assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 300

    def test_dict_usage_is_counted_the_same_way(self):
        attrs = _span_attributes(dict(NET_USAGE))
        assert attrs["gen_ai.usage.input_tokens"] == 2400

    @pytest.mark.asyncio
    async def test_anthropic_usage(self):
        service = AnthropicLLMService(api_key="test-key")
        service.start_llm_usage_metrics = AsyncMock()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=2000,
        )
        assert _span_attributes(_reported(service))["gen_ai.usage.input_tokens"] == 2400

    @pytest.mark.asyncio
    async def test_bedrock_usage(self):
        service = AWSBedrockLLMService(
            aws_access_key="test-key",
            aws_secret_key="test-secret",
            aws_region="us-east-1",
            settings=AWSBedrockLLMService.Settings(
                model="us.anthropic.claude-sonnet-4-20250514-v1:0"
            ),
        )
        service.start_llm_usage_metrics = AsyncMock()
        await service._report_usage_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_input_tokens=2000,
            cache_creation_input_tokens=300,
        )
        assert _span_attributes(_reported(service))["gen_ai.usage.input_tokens"] == 2400


class TestGrossReportingUsage:
    """Usage whose ``prompt_tokens`` already includes the cache keeps it as the input count."""

    @pytest.mark.parametrize(
        "usage",
        [
            pytest.param(
                LLMTokenUsage(
                    prompt_tokens=2100,
                    completion_tokens=50,
                    total_tokens=2150,
                    cache_read_input_tokens=2000,
                ),
                id="cache-reads",
            ),
            pytest.param(
                LLMTokenUsage(
                    prompt_tokens=2400,
                    completion_tokens=50,
                    total_tokens=2450,
                    cache_read_input_tokens=2000,
                    cache_creation_input_tokens=300,
                ),
                id="cache-reads-and-writes",
            ),
            pytest.param(
                LLMTokenUsage(
                    prompt_tokens=2100,
                    completion_tokens=50,
                    total_tokens=0,
                    cache_read_input_tokens=2000,
                ),
                id="no-total",
            ),
        ],
    )
    def test_cached_tokens_are_not_counted_twice(self, usage):
        assert _span_attributes(usage)["gen_ai.usage.input_tokens"] == usage.prompt_tokens
