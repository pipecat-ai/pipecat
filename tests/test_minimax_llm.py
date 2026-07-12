#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the MiniMax LLM service."""

from unittest.mock import patch

from pipecat.services.minimax.llm import MiniMaxLLMService


def test_minimax_llm_defaults():
    """Use the global endpoint and current default model when not overridden."""
    with patch.object(MiniMaxLLMService, "create_client") as create_client:
        service = MiniMaxLLMService(api_key="test-key")

    assert service._settings.model == "MiniMax-M3"
    assert create_client.call_args.kwargs["base_url"] == "https://api.minimax.io/v1"


def test_minimax_llm_supports_mainland_china_endpoint():
    """Allow callers to select the Mainland China endpoint and another supported model."""
    with patch.object(MiniMaxLLMService, "create_client") as create_client:
        service = MiniMaxLLMService(
            api_key="test-key",
            base_url="https://api.minimaxi.com/v1",
            settings=MiniMaxLLMService.Settings(model="MiniMax-M2.7"),
        )

    assert service._settings.model == "MiniMax-M2.7"
    assert create_client.call_args.kwargs["base_url"] == "https://api.minimaxi.com/v1"
