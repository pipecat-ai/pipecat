#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the MiniMax Anthropic-compatible LLM service."""

import httpx
import pytest
from anthropic import AsyncAnthropic

import pipecat.services.minimax.anthropic_llm as minimax_anthropic
from pipecat.services.minimax.anthropic_llm import MiniMaxAnthropicLLMService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "model"),
    [
        ("https://api.minimax.io/anthropic", "MiniMax-M3"),
        ("https://api.minimaxi.com/anthropic", "MiniMax-M2.7"),
    ],
)
async def test_minimax_anthropic_llm_appends_messages_path(monkeypatch, base_url, model):
    """Keep the public base URL at /anthropic while the SDK appends /v1/messages."""
    requests = []

    async def handle_request(request):
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    real_client = AsyncAnthropic

    def create_client(**kwargs):
        return real_client(**kwargs, http_client=http_client)

    monkeypatch.setattr(minimax_anthropic, "AsyncAnthropic", create_client)
    service = MiniMaxAnthropicLLMService(
        api_key="test-key",
        base_url=base_url,
        settings=MiniMaxAnthropicLLMService.Settings(model=model),
    )

    try:
        response = await service._client.messages.create(
            model=service._settings.model,
            max_tokens=1,
            messages=[{"role": "user", "content": "Hello"}],
        )
    finally:
        await service._client.close()

    assert response.content[0].text == "ok"
    assert str(requests[0].url) == f"{base_url}/v1/messages"
