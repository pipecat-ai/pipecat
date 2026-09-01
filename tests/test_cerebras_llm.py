#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Cerebras LLM service."""

from unittest.mock import patch

import pytest
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.cerebras.llm import CerebrasLLMService


@pytest.fixture
def service_factory():
    """Build a CerebrasLLMService without opening a client connection."""

    def _build(**settings_kwargs):
        with patch.object(CerebrasLLMService, "create_client"):
            return CerebrasLLMService(
                api_key="test-key",
                settings=CerebrasLLMService.Settings(**settings_kwargs),
            )

    return _build


def test_max_tokens_reaches_the_request(service_factory):
    """Cerebras accepts and honors ``max_tokens``, so it must not be dropped."""
    service = service_factory(model="gpt-oss-120b", max_tokens=200)

    params = service.build_chat_completion_params({})

    assert params["max_tokens"] == 200


def test_max_completion_tokens_reaches_the_request(service_factory):
    """``max_completion_tokens`` is the documented spelling and must survive too."""
    service = service_factory(model="gpt-oss-120b", max_completion_tokens=200)

    params = service.build_chat_completion_params({})

    assert params["max_completion_tokens"] == 200


def test_penalties_reach_the_request(service_factory):
    """Cerebras supports frequency and presence penalties."""
    service = service_factory(model="gpt-oss-120b", frequency_penalty=0.5, presence_penalty=-0.5)

    params = service.build_chat_completion_params({})

    assert params["frequency_penalty"] == 0.5
    assert params["presence_penalty"] == -0.5


def test_unset_params_are_omitted(service_factory):
    """Unset fields carry the OpenAI sentinel so the SDK keeps them off the wire."""
    service = service_factory(model="gpt-oss-120b")

    params = service.build_chat_completion_params({})

    assert params["max_tokens"] is OPENAI_NOT_GIVEN
    assert params["max_completion_tokens"] is OPENAI_NOT_GIVEN
    assert params["frequency_penalty"] is OPENAI_NOT_GIVEN


def test_extra_passes_provider_specific_params(service_factory):
    """``extra`` carries Cerebras-only params such as ``reasoning_effort``."""
    service = service_factory(model="gpt-oss-120b", extra={"reasoning_effort": "low"})

    params = service.build_chat_completion_params({})

    assert params["reasoning_effort"] == "low"


def test_developer_messages_are_sent_as_is(service_factory):
    """Cerebras maps the "developer" role to its developer instruction layer."""
    service = service_factory(model="gpt-oss-120b")
    context = LLMContext(
        messages=[
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ]
    )

    params = service.get_llm_adapter().get_llm_invocation_params(
        context, convert_developer_to_user=not service.supports_developer_role
    )

    assert params["messages"][0]["role"] == "developer"
