#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Responses API LLM client teardown."""

from unittest.mock import AsyncMock

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
async def test_cleanup_closes_responses_client(service_class):
    """Teardown closes the real SDK client, including on repeated cleanup."""
    service = service_class(api_key="test-key")
    client = service._client
    try:
        assert not client.is_closed()
        await service.cleanup()
        assert client.is_closed()
        await service.cleanup()
        assert client.is_closed()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_websocket_cleanup_disconnects_websocket_and_closes_client():
    """The WebSocket variant closes both its socket and its HTTP client."""
    service = OpenAIResponsesLLMService(api_key="test-key")
    ws = AsyncMock()
    service._websocket = ws
    try:
        await service.cleanup()
        ws.close.assert_awaited_once()
        assert service._websocket is None
        assert service._client.is_closed()
    finally:
        await service._client.close()


@pytest.mark.asyncio
async def test_cleanup_runs_registered_tool_cleanup_before_closing_client():
    """Registered tool resources are released while the LLM client is available."""
    service = OpenAIResponsesHttpLLMService(api_key="test-key")
    client_states = []

    async def handler(params):
        pass

    async def cleanup_tool():
        client_states.append(service._client.is_closed())

    handler._pipecat_cleanup = cleanup_tool
    service.register_function("test_tool", handler)

    try:
        await service.cleanup()
        assert client_states == [False]
        assert service._client.is_closed()
    finally:
        await service._client.close()
