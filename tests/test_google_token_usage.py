#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the token usage reported by GoogleLLMService."""

from unittest.mock import AsyncMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService


async def _reported_usage(*chunks):
    """Stream canned chunks through the service and return the usage it reported."""
    service = GoogleLLMService(api_key="test-key")
    service.start_llm_usage_metrics = AsyncMock()

    async def generator():
        for chunk in chunks:
            yield chunk

    async def fake_stream(context):
        return generator()

    async def capture_frame(frame, direction=None):
        pass

    with (
        patch.object(service, "push_frame", capture_frame),
        patch.object(service, "_stream_content", fake_stream),
    ):
        await service._process_context(LLMContext())

    service.start_llm_usage_metrics.assert_called_once()
    return service.start_llm_usage_metrics.call_args.args[0]


@pytest.mark.asyncio
async def test_thinking_tokens_are_counted_as_completion_tokens():
    """Gemini counts thoughts apart from the candidates, and both are generated output."""
    usage = await _reported_usage(
        GenerateContentResponse(
            candidates=[Candidate(content=Content(role="model", parts=[Part(text="Hi!")]))],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=40,
                candidates_token_count=10,
                thoughts_token_count=120,
                total_token_count=170,
            ),
        )
    )
    assert usage.completion_tokens == 130
    assert usage.reasoning_tokens == 120
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


@pytest.mark.asyncio
async def test_tool_results_are_counted_as_prompt_tokens():
    """Code execution, URL context and search results fed back to the model are input."""
    usage = await _reported_usage(
        GenerateContentResponse(
            candidates=[Candidate(content=Content(role="model", parts=[Part(text="5117")]))],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=23,
                candidates_token_count=57,
                tool_use_prompt_token_count=79,
                total_token_count=159,
            ),
        )
    )
    assert usage.prompt_tokens == 102
    assert usage.completion_tokens == 57
    assert usage.total_tokens == 159
