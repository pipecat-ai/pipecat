#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the OpenAI client's max_retries and request timeout on LLM services."""

import pytest
from aiohttp import web

from pipecat.frames.frames import ErrorFrame, LLMContextFrame, LLMServiceMetadataFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesHttpLLMService,
    OpenAIResponsesLLMService,
)
from pipecat.tests.utils import run_test

SERVICE_FACTORIES = {
    "openai": lambda **kwargs: OpenAILLMService(api_key="test-key", **kwargs),
    # Subclasses that override create_client() get the options too.
    "groq": lambda **kwargs: GroqLLMService(api_key="test-key", **kwargs),
    "azure": lambda **kwargs: AzureLLMService(
        api_key="test-key", endpoint="https://example.openai.azure.com/openai/v1", **kwargs
    ),
    "azure_dated": lambda **kwargs: AzureLLMService(
        api_key="test-key", endpoint="https://example.openai.azure.com", **kwargs
    ),
    "responses_http": lambda **kwargs: OpenAIResponsesHttpLLMService(api_key="test-key", **kwargs),
    "responses_websocket": lambda **kwargs: OpenAIResponsesLLMService(api_key="test-key", **kwargs),
}


@pytest.mark.parametrize("make_service", SERVICE_FACTORIES.values(), ids=SERVICE_FACTORIES.keys())
def test_client_uses_given_retries_and_timeout(make_service):
    default_client = make_service()._client
    service = make_service(max_retries=0, request_timeout_secs=3.0)

    assert service._client.max_retries == 0
    assert service._client.timeout == 3.0
    # Everything else the service configured on the client is kept.
    assert service._client.base_url == default_client.base_url
    assert service._client.api_key == default_client.api_key


@pytest.mark.parametrize("make_service", SERVICE_FACTORIES.values(), ids=SERVICE_FACTORIES.keys())
def test_client_keeps_sdk_defaults_when_not_given(make_service):
    client = make_service()._client
    assert client.max_retries == 2
    assert client.timeout.read == 600


@pytest.mark.asyncio
async def test_responses_http_max_retries_zero_surfaces_first_failure():
    """A failed request reaches the pipeline as an error without the client retrying it."""
    requests = 0

    async def unavailable(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        return web.json_response(
            {"error": {"message": "overloaded"}}, status=503, headers={"Retry-After": "30"}
        )

    app = web.Application()
    app.router.add_post("/v1/responses", unavailable)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        service = OpenAIResponsesHttpLLMService(
            api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0
        )
        context = LLMContext(messages=[{"role": "user", "content": "hi"}])
        _, up_frames = await run_test(
            service,
            frames_to_send=[LLMContextFrame(context)],
            expected_up_frames=[LLMServiceMetadataFrame, ErrorFrame],
        )
    finally:
        await runner.cleanup()

    assert requests == 1
    assert "overloaded" in up_frames[-1].error
