#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService

BASE_URL = "wss://my-resource.openai.azure.com/openai/v1/realtime"


@pytest.mark.asyncio
async def test_auth_headers_use_api_key():
    service = AzureRealtimeLLMService(base_url=BASE_URL, api_key="test-key")

    assert await service._auth_headers() == {"api-key": "test-key"}


@pytest.mark.asyncio
async def test_auth_headers_prefer_token_provider():
    async def token_provider():
        return "entra-token"

    service = AzureRealtimeLLMService(base_url=BASE_URL, token_provider=token_provider)

    assert await service._auth_headers() == {"Authorization": "Bearer entra-token"}
