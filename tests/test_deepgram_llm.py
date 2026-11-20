#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Deepgram LLM service's wire format."""

import json
import sys

import pytest

from pipecat.frames.frames import LLMContextFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.deepgram.llm import DeepgramLLMService
from pipecat.tests.utils import run_test

_CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "think-fast",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
}


def _mock_http(service: DeepgramLLMService, requests: list):
    """Answer the service's requests with one streamed chunk, recording each request."""
    http = service._client._client
    # openai < 3 is built on httpx, openai >= 3 on httpx2.
    family = next(
        module
        for module in (sys.modules.get("httpx2"), sys.modules.get("httpx"))
        if module is not None and isinstance(http, module.AsyncClient)
    )

    def handler(request):
        requests.append(request)
        body = f"data: {json.dumps(_CHUNK)}\n\ndata: [DONE]\n\n"
        return family.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode()
        )

    http._transport = family.MockTransport(handler)


@pytest.mark.asyncio
async def test_request_matches_what_think_accepts():
    service = DeepgramLLMService(
        api_key="test-key",
        settings=DeepgramLLMService.Settings(temperature=0.2, max_tokens=20),
    )
    requests = []
    _mock_http(service, requests)
    context = LLMContext(messages=[{"role": "developer", "content": "Introduce yourself."}])

    down, _ = await run_test(service, frames_to_send=[LLMContextFrame(context)])

    (request,) = requests
    body = json.loads(request.content)
    assert request.url.path == "/v1/think"
    assert request.headers["authorization"] == "Token test-key"
    # `/v1/think` rejects unknown fields, so `max_tokens` must be dropped.
    assert set(body) == {"model", "temperature", "messages", "stream", "stream_options"}
    assert body["temperature"] == 0.2
    assert body["messages"] == [{"role": "user", "content": "Introduce yourself."}]
    assert "".join(f.text for f in down if isinstance(f, LLMTextFrame)) == "hi"
