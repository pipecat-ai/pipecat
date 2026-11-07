#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that LLM services count reasoning tokens as completion tokens.

Pipecat reports every generated token in ``completion_tokens`` and the part
spent on reasoning in ``reasoning_tokens``, whichever way a provider reports
them.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from anthropic.types.beta import BetaMessageDeltaUsage, BetaUsage
from google.genai.types import LiveServerMessage, UsageMetadata
from openai.types import CompletionUsage

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.xai.llm import GrokLLMService


def _usage(**fields) -> CompletionUsage:
    return CompletionUsage.model_validate(fields)


# -- OpenAI-compatible ------------------------------------------------------


def test_openai_reasoning_is_part_of_completion_tokens():
    service = OpenAILLMService(api_key="test-key")
    tokens = service._token_usage(
        _usage(
            prompt_tokens=40,
            completion_tokens=130,
            total_tokens=170,
            completion_tokens_details={"reasoning_tokens": 120},
        )
    )
    assert tokens.completion_tokens == 130
    assert tokens.reasoning_tokens == 120


def test_grok_adds_reasoning_reported_apart_from_completion_tokens():
    service = GrokLLMService(api_key="test-key")
    tokens = service._token_usage(
        _usage(
            prompt_tokens=32,
            completion_tokens=9,
            total_tokens=135,
            completion_tokens_details={"reasoning_tokens": 94},
        )
    )
    assert tokens.completion_tokens == 103
    assert tokens.reasoning_tokens == 94
    assert tokens.total_tokens == 135


def test_grok_keeps_reasoning_already_in_completion_tokens():
    service = GrokLLMService(api_key="test-key")
    tokens = service._token_usage(
        _usage(
            prompt_tokens=32,
            completion_tokens=103,
            total_tokens=135,
            completion_tokens_details={"reasoning_tokens": 94},
        )
    )
    assert tokens.completion_tokens == 103


# -- Gemini Live ------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_live_adds_thoughts_to_completion_tokens():
    service = GeminiLiveLLMService(api_key="test-key")
    service.start_llm_usage_metrics = AsyncMock()

    await service._handle_msg_usage_metadata(
        LiveServerMessage(
            usage_metadata=UsageMetadata(
                prompt_token_count=40,
                response_token_count=10,
                thoughts_token_count=120,
                total_token_count=170,
            )
        )
    )

    tokens = service.start_llm_usage_metrics.call_args.args[0]
    assert tokens.completion_tokens == 130
    assert tokens.reasoning_tokens == 120
    assert tokens.total_tokens == 170


@pytest.mark.asyncio
async def test_gemini_live_computes_the_total_from_input_and_output():
    """Gemini Live's own total sometimes leaves the thinking tokens out."""
    service = GeminiLiveLLMService(api_key="test-key")
    service.start_llm_usage_metrics = AsyncMock()

    await service._handle_msg_usage_metadata(
        LiveServerMessage(
            usage_metadata=UsageMetadata(
                prompt_token_count=2921,
                response_token_count=41,
                thoughts_token_count=176,
                tool_use_prompt_token_count=33,
                total_token_count=2962,
            )
        )
    )

    tokens = service.start_llm_usage_metrics.call_args.args[0]
    assert tokens.prompt_tokens == 2954
    assert tokens.completion_tokens == 217
    assert tokens.total_tokens == 3171


# -- Anthropic --------------------------------------------------------------


async def _anthropic_reported_usage(*events):
    """Stream canned events through the service and return the usage it reported."""
    service = AnthropicLLMService(api_key="test-key")
    service.start_llm_usage_metrics = AsyncMock()

    async def generator():
        for event in events:
            yield event

    async def fake_stream(api_call, params):
        return generator()

    async def capture_frame(frame, direction=None):
        pass

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "_create_message_stream", fake_stream),
    ):
        await service._process_context(LLMContext())

    return service.start_llm_usage_metrics.call_args.args[0]


def _message_start(usage: BetaUsage) -> SimpleNamespace:
    return SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage))


def _message_delta(usage: BetaMessageDeltaUsage) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=usage
    )


@pytest.mark.asyncio
async def test_anthropic_reports_thinking_tokens_from_the_final_usage():
    """message_delta counts are cumulative, so they replace message_start's."""
    tokens = await _anthropic_reported_usage(
        _message_start(
            BetaUsage(
                input_tokens=40,
                output_tokens=2,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=100,
            )
        ),
        _message_delta(
            BetaMessageDeltaUsage(
                input_tokens=40,
                output_tokens=130,
                cache_read_input_tokens=100,
                output_tokens_details={"thinking_tokens": 120},
            )
        ),
    )
    assert tokens.prompt_tokens == 40
    assert tokens.completion_tokens == 130
    assert tokens.reasoning_tokens == 120
    assert tokens.cache_read_input_tokens == 100
    assert tokens.total_tokens == 270


@pytest.mark.asyncio
async def test_anthropic_keeps_input_counts_a_message_delta_leaves_out():
    tokens = await _anthropic_reported_usage(
        _message_start(BetaUsage(input_tokens=40, output_tokens=2, cache_read_input_tokens=100)),
        _message_delta(BetaMessageDeltaUsage(output_tokens=15)),
    )
    assert tokens.prompt_tokens == 40
    assert tokens.completion_tokens == 15
    assert tokens.cache_read_input_tokens == 100
    assert tokens.reasoning_tokens is None
