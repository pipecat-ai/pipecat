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

from unittest.mock import AsyncMock

import pytest
from google.genai.types import LiveServerMessage, UsageMetadata
from openai.types import CompletionUsage

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
