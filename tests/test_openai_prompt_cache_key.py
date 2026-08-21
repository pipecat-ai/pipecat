#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the optional ``prompt_cache_key`` passthrough on OpenAI services.

Covers request-shape only: the supplied key must appear unchanged in the
request parameters, and an unset key must be omitted entirely (not sent as a
sentinel), since ``prompt_cache_key`` requires a newer ``openai`` SDK than
this package's minimum.
"""

from unittest.mock import patch

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesHttpLLMService


def _make_chat_service(**kwargs):
    with patch.object(BaseOpenAILLMService, "create_client"):
        return BaseOpenAILLMService(api_key="test-key", **kwargs)


def _make_responses_service(**kwargs):
    with patch.object(OpenAIResponsesHttpLLMService, "_create_client"):
        return OpenAIResponsesHttpLLMService(api_key="test-key", **kwargs)


_CHAT_INVOCATION_PARAMS = {"messages": [], "tools": [], "tool_choice": None}
_RESPONSES_INVOCATION_PARAMS = {"input": []}


class TestChatCompletionsPromptCacheKey:
    def test_supplied_key_is_passed_unchanged(self):
        service = _make_chat_service(prompt_cache_key="session-abc123")
        params = service.build_chat_completion_params(dict(_CHAT_INVOCATION_PARAMS))
        assert params["prompt_cache_key"] == "session-abc123"

    def test_unset_key_is_omitted(self):
        service = _make_chat_service()
        params = service.build_chat_completion_params(dict(_CHAT_INVOCATION_PARAMS))
        assert "prompt_cache_key" not in params


class TestResponsesPromptCacheKey:
    def test_supplied_key_is_passed_unchanged(self):
        service = _make_responses_service(prompt_cache_key="session-abc123")
        params = service._build_response_params(dict(_RESPONSES_INVOCATION_PARAMS))
        assert params["prompt_cache_key"] == "session-abc123"

    def test_unset_key_is_omitted(self):
        service = _make_responses_service()
        params = service._build_response_params(dict(_RESPONSES_INVOCATION_PARAMS))
        assert "prompt_cache_key" not in params
