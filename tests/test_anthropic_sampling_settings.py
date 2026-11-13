#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how AnthropicLLMService sends temperature, top_k and top_p.

The Messages API methods have no parameters for them, so the service carries
them in ``extra_body``, which the SDK merges into the request JSON as-is.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService


async def _request_kwargs(**settings: Any) -> dict[str, Any]:
    """Return the kwargs run_inference sends for a service with these settings."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(**settings)
    )
    service._client = AsyncMock()
    service._client.beta.messages.create.return_value = SimpleNamespace(content=[])

    await service.run_inference(LLMContext(messages=[{"role": "user", "content": "hi"}]))

    return service._client.beta.messages.create.call_args.kwargs


@pytest.mark.asyncio
async def test_sampling_settings_are_sent_in_extra_body():
    kwargs = await _request_kwargs(temperature=0.6, top_k=50, top_p=0.95)

    assert kwargs["extra_body"] == {"temperature": 0.6, "top_k": 50, "top_p": 0.95}
    for name in ("temperature", "top_k", "top_p"):
        assert name not in kwargs


@pytest.mark.asyncio
async def test_unset_sampling_settings_are_omitted():
    kwargs = await _request_kwargs(temperature=0.6)

    assert kwargs["extra_body"] == {"temperature": 0.6}


@pytest.mark.asyncio
async def test_no_extra_body_when_no_sampling_settings():
    kwargs = await _request_kwargs()

    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_explicit_none_is_sent():
    """A None reaches the API as JSON null rather than being dropped."""
    kwargs = await _request_kwargs(temperature=None)

    assert kwargs["extra_body"] == {"temperature": None}


@pytest.mark.asyncio
async def test_extra_body_from_extra_wins_per_key():
    kwargs = await _request_kwargs(
        temperature=0.6, top_k=50, extra={"extra_body": {"temperature": 0.1}}
    )

    assert kwargs["extra_body"] == {"temperature": 0.1, "top_k": 50}
