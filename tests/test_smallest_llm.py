#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Smallest AI LLM service."""

from unittest.mock import patch

import pytest
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.smallest.llm import SmallestLLMService


@pytest.fixture
def service_factory():
    """Build a SmallestLLMService without opening a client connection."""

    def _build(**settings_kwargs):
        with patch.object(SmallestLLMService, "create_client"):
            return SmallestLLMService(
                api_key="test-key",
                settings=SmallestLLMService.Settings(**settings_kwargs)
                if settings_kwargs
                else None,
            )

    return _build


def test_default_model_is_electron(service_factory):
    """Electron is Smallest AI's LLM and the service's default model."""
    service = service_factory()

    params = service.build_chat_completion_params({})

    assert params["model"] == "electron"


def test_base_url_defaults_to_waves_api(service_factory):
    """The service should point at Smallest AI's Waves API by default."""
    with patch.object(SmallestLLMService, "create_client") as mock_create_client:
        SmallestLLMService(api_key="test-key")

    assert mock_create_client.call_args.kwargs["base_url"] == "https://api.smallest.ai/waves/v1"


def test_model_can_be_overridden(service_factory):
    """Callers can select a different OpenAI-compatible model via settings."""
    service = service_factory(model="some-other-model")

    params = service.build_chat_completion_params({})

    assert params["model"] == "some-other-model"


def test_temperature_reaches_the_request(service_factory):
    """Standard OpenAI-compatible params pass through untouched."""
    service = service_factory(temperature=0.3)

    params = service.build_chat_completion_params({})

    assert params["temperature"] == 0.3


def test_unset_params_are_omitted(service_factory):
    """Unset fields carry the OpenAI sentinel so the SDK keeps them off the wire."""
    service = service_factory()

    params = service.build_chat_completion_params({})

    assert params["max_tokens"] is OPENAI_NOT_GIVEN
    assert params["frequency_penalty"] is OPENAI_NOT_GIVEN


@pytest.mark.asyncio
async def test_developer_messages_are_converted_to_user(service_factory):
    """Electron doesn't support the "developer" role, so it's sent as "user"."""
    service = service_factory()
    context = LLMContext(
        messages=[
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "Hello"},
        ]
    )

    params = await service.get_llm_adapter().get_llm_invocation_params(
        context, convert_developer_to_user=not service.supports_developer_role
    )

    assert params["messages"][0]["role"] == "user"
