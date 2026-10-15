#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Provider-specific request fields that OpenAI-compatible services send in ``extra_body``."""

from unittest.mock import patch

import pytest
from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.inception.llm import InceptionLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.services.together.llm import TogetherLLMService
from pipecat.services.xai.llm import GrokLLMService
from pipecat.utils.types import NOT_GIVEN


def _build_params(service_class, settings=None):
    with patch.object(service_class, "create_client"):
        service = service_class(api_key="test-key", settings=settings)
    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )
    return service.build_chat_completion_params(invocation)


def test_merge_extra_body_omits_unset_and_none():
    params = {}
    BaseOpenAILLMService._merge_extra_body(params, {"a": 1, "b": None, "c": NOT_GIVEN})
    assert params == {"extra_body": {"a": 1}}

    params = {}
    BaseOpenAILLMService._merge_extra_body(params, {"a": None})
    assert "extra_body" not in params


def test_merge_extra_body_user_wins_and_is_not_modified():
    user_extra_body = {"a": "user", "u": 1}
    params = {"extra_body": user_extra_body}
    BaseOpenAILLMService._merge_extra_body(params, {"a": "service", "s": 2})

    assert params["extra_body"] == {"a": "user", "u": 1, "s": 2}
    assert user_extra_body == {"a": "user", "u": 1}


def test_together_reasoning_unset_by_default():
    params = _build_params(TogetherLLMService)
    assert "extra_body" not in params


def test_together_reasoning_in_extra_body():
    params = _build_params(
        TogetherLLMService,
        TogetherLLMService.Settings(
            reasoning={"enabled": False}, extra={"extra_body": {"user_field": 1}}
        ),
    )
    assert "reasoning" not in params
    assert params["extra_body"] == {"reasoning": {"enabled": False}, "user_field": 1}


def test_grok_reasoning_effort_unset_by_default():
    params = _build_params(GrokLLMService)
    assert "reasoning_effort" not in params
    assert "extra_body" not in params


def test_grok_reasoning_effort_is_top_level():
    params = _build_params(GrokLLMService, GrokLLMService.Settings(reasoning_effort="low"))
    assert params["reasoning_effort"] == "low"
    assert "extra_body" not in params


def test_grok_settings_extra_reasoning_effort_wins():
    params = _build_params(
        GrokLLMService,
        GrokLLMService.Settings(reasoning_effort="low", extra={"reasoning_effort": "high"}),
    )
    assert params["reasoning_effort"] == "high"


def test_deepseek_disables_thinking_by_default():
    params = _build_params(DeepSeekLLMService)
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "seed" not in params
    assert "max_completion_tokens" not in params


def test_deepseek_thinking_none_leaves_default_to_deepseek():
    params = _build_params(DeepSeekLLMService, DeepSeekLLMService.Settings(thinking=None))
    assert "extra_body" not in params


def test_deepseek_thinking_keeps_user_extra_body():
    params = _build_params(
        DeepSeekLLMService,
        DeepSeekLLMService.Settings(
            thinking=DeepSeekLLMService.ThinkingConfig(type="enabled"),
            extra={"extra_body": {"user_field": 1}},
        ),
    )
    assert params["extra_body"] == {"thinking": {"type": "enabled"}, "user_field": 1}


def test_inception_realtime_keeps_user_extra_body():
    params = _build_params(
        InceptionLLMService,
        InceptionLLMService.Settings(realtime=True, extra={"extra_body": {"user_field": 1}}),
    )
    assert params["extra_body"] == {"realtime": True, "user_field": 1}


def test_sarvam_wiki_grounding_does_not_modify_settings_extra():
    user_extra_body = {"user_field": 1}
    params = _build_params(
        SarvamLLMService,
        SarvamLLMService.Settings(
            model="sarvam-105b", wiki_grounding=True, extra={"extra_body": user_extra_body}
        ),
    )
    assert params["extra_body"] == {"wiki_grounding": True, "user_field": 1}
    assert user_extra_body == {"user_field": 1}


def test_sarvam_user_extra_body_wins():
    params = _build_params(
        SarvamLLMService,
        SarvamLLMService.Settings(
            model="sarvam-105b",
            wiki_grounding=True,
            extra={"extra_body": {"wiki_grounding": False}},
        ),
    )
    assert params["extra_body"] == {"wiki_grounding": False}


def test_openrouter_provider_unset_by_default():
    params = _build_params(OpenRouterLLMService)
    assert "extra_body" not in params


def test_openrouter_provider_preferences_in_extra_body():
    params = _build_params(
        OpenRouterLLMService,
        OpenRouterLLMService.Settings(
            provider=OpenRouterLLMService.ProviderPreferences(
                sort="latency", preferred_max_latency={"p90": 1.5}
            )
        ),
    )
    assert params["extra_body"] == {
        "provider": {"sort": "latency", "preferred_max_latency": {"p90": 1.5}}
    }


def test_openrouter_provider_dict_in_extra_body():
    params = _build_params(
        OpenRouterLLMService,
        OpenRouterLLMService.Settings(provider={"only": ["azure"], "allow_fallbacks": False}),
    )
    assert params["extra_body"] == {"provider": {"only": ["azure"], "allow_fallbacks": False}}


def test_openrouter_provider_preferences_keep_unknown_options():
    params = _build_params(
        OpenRouterLLMService,
        OpenRouterLLMService.Settings(
            provider=OpenRouterLLMService.ProviderPreferences(sort="price", new_option=True)
        ),
    )
    assert params["extra_body"] == {"provider": {"sort": "price", "new_option": True}}


def test_openrouter_user_extra_body_wins():
    params = _build_params(
        OpenRouterLLMService,
        OpenRouterLLMService.Settings(
            provider={"only": ["azure"]},
            extra={"extra_body": {"provider": {"only": ["groq"]}, "user_field": 1}},
        ),
    )
    assert params["extra_body"] == {"provider": {"only": ["groq"]}, "user_field": 1}


def test_openrouter_provider_dict_coerced_to_preferences():
    settings = OpenRouterLLMService.Settings(provider={"sort": "latency"})
    assert settings.provider == OpenRouterLLMService.ProviderPreferences(sort="latency")


@pytest.mark.asyncio
async def test_openrouter_provider_updates_at_runtime():
    with patch.object(OpenRouterLLMService, "create_client"):
        service = OpenRouterLLMService(
            api_key="test-key", settings=OpenRouterLLMService.Settings(provider={"only": ["azure"]})
        )
    invocation = OpenAILLMInvocationParams(
        messages=[{"role": "user", "content": "Hello"}],
        tools=OPENAI_NOT_GIVEN,
        tool_choice=OPENAI_NOT_GIVEN,
    )

    changed = await service._update_settings(
        OpenRouterLLMService.Settings.from_mapping({"provider": {"sort": "latency"}})
    )
    assert "provider" in changed
    params = service.build_chat_completion_params(invocation)
    assert params["extra_body"] == {"provider": {"sort": "latency"}}

    await service._update_settings(OpenRouterLLMService.Settings(provider=None))
    params = service.build_chat_completion_params(invocation)
    assert "extra_body" not in params
