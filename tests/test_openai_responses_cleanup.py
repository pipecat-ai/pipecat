#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Responses service client cleanup."""

import pytest

from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMService,
)
from pipecat.services.perplexity.llm import PerplexityAgentHttpLLMService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_class",
    [OpenAIResponsesLLMService, OpenAIResponsesHttpLLMService, PerplexityAgentHttpLLMService],
)
async def test_cleanup_closes_the_openai_client(service_class):
    service = service_class(api_key="test-key")

    await service.cleanup()

    assert service._client.is_closed()


@pytest.mark.asyncio
async def test_websocket_cleanup_disconnects_after_closing_the_client():
    service = OpenAIResponsesLLMService(api_key="test-key")

    await service.cleanup()

    assert service._client.is_closed()
    assert service._websocket is None
