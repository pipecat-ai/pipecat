#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for LLMTR LLM service."""

from unittest.mock import patch

from pipecat.services.llmtr.llm import LLMTRLLMService


def test_llmtr_default_base_url_and_model():
    """LLMTR uses the gateway base URL and a namespaced default model."""
    with patch.object(LLMTRLLMService, "create_client") as create_client:
        service = LLMTRLLMService(api_key="test-key")
    assert create_client.call_args.kwargs["base_url"] == "https://llmtr.com/v1"
    assert service._settings.model == "minimax/minimax-m3.1-flash"
