#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI-compatible services never send ``OPENAI_API_KEY`` to another provider."""

import pytest

from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")


@pytest.mark.parametrize("service_cls", [GroqSTTService, OpenRouterLLMService])
def test_api_key_is_required(service_cls):
    with pytest.raises(TypeError):
        service_cls()


@pytest.mark.parametrize("service_cls", [GroqSTTService, OpenRouterLLMService, NvidiaLLMService])
def test_explicit_api_key_is_used(service_cls):
    service = service_cls(api_key="provider-key")
    assert service._client.api_key == "provider-key"


def test_nvidia_local_nim_without_key_does_not_use_openai_key():
    service = NvidiaLLMService(base_url="http://localhost:8000/v1")
    assert service._client.api_key != "openai-key"


def test_nvidia_local_nim_without_any_key_constructs(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    NvidiaLLMService(base_url="http://localhost:8000/v1")
