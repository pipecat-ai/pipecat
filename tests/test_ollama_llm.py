#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Ollama LLM service's client configuration."""

from pipecat.services.ollama.llm import OLLamaLLMService


def test_api_key_is_sent_to_the_server():
    service = OLLamaLLMService(api_key="proxy-key")

    assert service._client.api_key == "proxy-key"


def test_local_ollama_needs_no_api_key():
    service = OLLamaLLMService()

    assert service._client.api_key == "ollama"
