#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the requests and usage of PerplexityAgentLLMService."""

from unittest.mock import AsyncMock

import pytest
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.perplexity.llm import PerplexityAgentLLMService


def _request(service: PerplexityAgentLLMService, context: LLMContext | None = None) -> dict:
    context = context or LLMContext(messages=[{"role": "user", "content": "Hi"}])
    invocation_params = service.get_llm_adapter().get_llm_invocation_params(context)
    return service._build_response_params(invocation_params)


def test_default_request_uses_the_fast_preset_with_a_pinned_model():
    params = _request(PerplexityAgentLLMService(api_key="test-key"))

    assert params["model"] == "openai/gpt-5.6-luna"
    assert params["extra_body"] == {"preset": "fast"}
    # The preset decides how much to reason.
    assert "reasoning" not in params
    assert params["tools"] == [{"type": "web_search"}]


def test_preset_alone_chooses_the_model():
    service = PerplexityAgentLLMService(
        api_key="test-key", settings=PerplexityAgentLLMService.Settings(model=None)
    )
    params = _request(service)

    assert "model" not in params
    assert params["extra_body"] == {"preset": "fast"}
    assert "reasoning" not in params


def test_neither_model_nor_preset_is_rejected():
    with pytest.raises(ValueError, match="needs a `model`, a `preset`"):
        PerplexityAgentLLMService(
            api_key="test-key",
            settings=PerplexityAgentLLMService.Settings(model=None, preset=None),
        )


@pytest.mark.asyncio
async def test_update_leaving_neither_model_nor_preset_reports_an_error():
    service = PerplexityAgentLLMService(api_key="test-key")
    service.push_error = AsyncMock()

    await service._update_settings(PerplexityAgentLLMService.Settings(model=None, preset=None))

    service.push_error.assert_called_once()


def test_model_without_preset():
    service = PerplexityAgentLLMService(
        api_key="test-key",
        settings=PerplexityAgentLLMService.Settings(model="openai/gpt-5.6-luna", preset=None),
    )
    params = _request(service)

    assert params["model"] == "openai/gpt-5.6-luna"
    assert "extra_body" not in params
    # OpenAI models that reason have it turned off for latency, as on OpenAI.
    assert params["reasoning"] == {"effort": "none"}


def test_other_providers_keep_their_reasoning_default():
    service = PerplexityAgentLLMService(
        api_key="test-key",
        settings=PerplexityAgentLLMService.Settings(
            model="anthropic/claude-haiku-4-5", preset=None
        ),
    )

    assert "reasoning" not in _request(service)


def test_web_search_joins_context_tools_once():
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
    params = _request(PerplexityAgentLLMService(api_key="test-key"), context)

    assert [tool["type"] for tool in params["tools"]] == ["function", "web_search"]


def test_web_search_can_be_turned_off():
    service = PerplexityAgentLLMService(
        api_key="test-key", settings=PerplexityAgentLLMService.Settings(web_search=False)
    )

    assert "tools" not in _request(service)


def test_perplexity_fields_join_extra_body():
    service = PerplexityAgentLLMService(
        api_key="test-key",
        settings=PerplexityAgentLLMService.Settings(
            max_steps=2, extra={"extra_body": {"language_preference": "en"}}
        ),
    )

    assert _request(service)["extra_body"] == {
        "language_preference": "en",
        "preset": "fast",
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
    tokens = PerplexityAgentLLMService(api_key="test-key")._token_usage(usage)

    assert tokens.prompt_tokens == 3046
    assert tokens.cache_read_input_tokens == 1780
    assert tokens.cache_creation_input_tokens == 1263
    assert tokens.total_tokens == 3075


def test_a_preset_keeps_the_model_unless_it_is_cleared():
    service = PerplexityAgentLLMService(
        api_key="test-key", settings=PerplexityAgentLLMService.Settings(preset="high")
    )
    assert _request(service)["model"] == "openai/gpt-5.6-luna"

    service = PerplexityAgentLLMService(
        api_key="test-key",
        settings=PerplexityAgentLLMService.Settings(preset="high", model=None),
    )
    assert "model" not in _request(service)


def test_an_explicit_model_with_a_preset_keeps_the_preset_reasoning():
    service = PerplexityAgentLLMService(
        api_key="test-key",
        settings=PerplexityAgentLLMService.Settings(preset="high", model="openai/gpt-5.5"),
    )
    params = _request(service)

    assert params["model"] == "openai/gpt-5.5"
    assert "reasoning" not in params


def test_explicit_reasoning_is_sent_as_configured():
    service = PerplexityAgentLLMService(
        api_key="test-key",
        settings=PerplexityAgentLLMService.Settings(
            reasoning=PerplexityAgentLLMService.ReasoningConfig(effort="low")
        ),
    )

    assert _request(service)["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_a_preset_update_leaves_the_model_alone():
    service = PerplexityAgentLLMService(api_key="test-key")

    await service._update_settings(PerplexityAgentLLMService.Settings(preset="high"))

    assert service._settings.model == "openai/gpt-5.6-luna"
