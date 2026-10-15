#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the requests and usage of PerplexityAgentHttpLLMService."""

from unittest.mock import AsyncMock

import pytest
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.perplexity.llm import PerplexityAgentHttpLLMService


async def _request(
    service: PerplexityAgentHttpLLMService, context: LLMContext | None = None
) -> dict:
    context = context or LLMContext(messages=[{"role": "user", "content": "Hi"}])
    invocation_params = await service.get_llm_adapter().get_llm_invocation_params(context)
    return service._build_response_params(invocation_params)


@pytest.mark.asyncio
async def test_default_request_uses_a_pinned_model_with_web_search():
    params = await _request(PerplexityAgentHttpLLMService(api_key="test-key"))

    assert params["model"] == "openai/gpt-5.6-luna"
    assert "extra_body" not in params
    assert params["tools"] == [{"type": "web_search"}]


@pytest.mark.asyncio
async def test_preset_alone_chooses_the_model():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(model=None, preset="fast"),
    )
    params = await _request(service)

    assert "model" not in params
    assert params["extra_body"] == {"preset": "fast"}
    assert "reasoning" not in params


def test_neither_model_nor_preset_is_rejected():
    with pytest.raises(ValueError, match="needs a `model`, a `preset`"):
        PerplexityAgentHttpLLMService(
            api_key="test-key",
            settings=PerplexityAgentHttpLLMService.Settings(model=None),
        )


@pytest.mark.asyncio
async def test_update_leaving_neither_model_nor_preset_reports_an_error():
    service = PerplexityAgentHttpLLMService(api_key="test-key")
    service.push_error = AsyncMock()

    await service._update_settings(PerplexityAgentHttpLLMService.Settings(model=None))

    service.push_error.assert_called_once()


@pytest.mark.asyncio
async def test_web_search_joins_context_tools_once():
    weather = FunctionSchema(
        name="get_weather",
        description="Get the weather.",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    context = LLMContext(
        messages=[{"role": "user", "content": "Hi"}],
        tools=ToolsSchema(standard_tools=[weather]),
    )
    params = await _request(PerplexityAgentHttpLLMService(api_key="test-key"), context)

    assert [tool["type"] for tool in params["tools"]] == ["function", "web_search"]


@pytest.mark.asyncio
async def test_web_search_can_be_turned_off():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key", settings=PerplexityAgentHttpLLMService.Settings(web_search=False)
    )

    assert "tools" not in await _request(service)


@pytest.mark.asyncio
async def test_perplexity_fields_join_extra_body():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(
            max_steps=2, extra={"extra_body": {"language_preference": "en"}}
        ),
    )

    assert (await _request(service))["extra_body"] == {
        "language_preference": "en",
        "max_steps": 2,
    }


def test_usage_reports_cache_creation_tokens():
    # construct() mirrors the SDK's lenient decode of a streamed usage payload,
    # which keeps fields it doesn't know and skips ones Perplexity leaves out.
    usage = ResponseUsage.construct(
        input_tokens=3046,
        input_tokens_details=InputTokensDetails.construct(
            cached_tokens=1780, cache_creation_input_tokens=1263, cache_read_input_tokens=1780
        ),
        output_tokens=29,
        output_tokens_details=OutputTokensDetails.construct(reasoning_tokens=0),
        total_tokens=3075,
    )
    tokens = PerplexityAgentHttpLLMService(api_key="test-key")._token_usage(usage)

    assert tokens.prompt_tokens == 3046
    assert tokens.cache_read_input_tokens == 1780
    assert tokens.cache_creation_input_tokens == 1263
    assert tokens.total_tokens == 3075


@pytest.mark.asyncio
async def test_a_preset_keeps_the_model_unless_it_is_cleared():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key", settings=PerplexityAgentHttpLLMService.Settings(preset="high")
    )
    assert (await _request(service))["model"] == "openai/gpt-5.6-luna"

    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(preset="high", model=None),
    )
    assert "model" not in await _request(service)


@pytest.mark.asyncio
async def test_an_explicit_model_with_a_preset_keeps_the_preset_reasoning():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(preset="high", model="openai/gpt-5.5"),
    )
    params = await _request(service)

    assert params["model"] == "openai/gpt-5.5"
    assert "reasoning" not in params


@pytest.mark.asyncio
async def test_explicit_reasoning_is_sent_as_configured():
    service = PerplexityAgentHttpLLMService(
        api_key="test-key",
        settings=PerplexityAgentHttpLLMService.Settings(
            reasoning=PerplexityAgentHttpLLMService.ReasoningConfig(effort="low")
        ),
    )

    assert (await _request(service))["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_a_preset_update_leaves_the_model_alone():
    service = PerplexityAgentHttpLLMService(api_key="test-key")

    await service._update_settings(PerplexityAgentHttpLLMService.Settings(preset="high"))

    assert service._settings.model == "openai/gpt-5.6-luna"
