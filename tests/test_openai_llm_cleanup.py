#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI-compatible LLM client teardown."""

import pytest

from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@pytest.mark.asyncio
@pytest.mark.parametrize("service_class", [OpenAILLMService, GroqLLMService])
async def test_cleanup_closes_openai_compatible_client(service_class: type[BaseOpenAILLMService]):
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
async def test_cleanup_runs_registered_tool_cleanup_before_closing_client():
    """Registered tool resources are released while the LLM client is available."""
    service = OpenAILLMService(api_key="test-key")
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
