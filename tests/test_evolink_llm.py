#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for EvoLink LLM service."""

from unittest.mock import patch

from pipecat.services.evolink.llm import EvoLinkLLMService


def test_evolink_llm_uses_default_endpoint_and_model():
    """Test EvoLink defaults are applied without making a network request."""
    with patch.object(EvoLinkLLMService, "create_client") as create_client:
        service = EvoLinkLLMService(api_key="test-key")

    create_client.assert_called_once()
    assert create_client.call_args.kwargs["api_key"] == "test-key"
    assert create_client.call_args.kwargs["base_url"] == "https://direct.evolink.ai/v1"
    assert service._settings.model == "gpt-5.2"


def test_evolink_llm_settings_override_default_model():
    """Test settings override the default EvoLink model."""
    with patch.object(EvoLinkLLMService, "create_client"):
        service = EvoLinkLLMService(
            api_key="test-key",
            settings=EvoLinkLLMService.Settings(model="deepseek-v4-flash"),
        )

    assert service._settings.model == "deepseek-v4-flash"
    assert not service.supports_developer_role
