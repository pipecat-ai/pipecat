#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI-compatible services never send ``OPENAI_API_KEY`` to another provider."""

import importlib
from pathlib import Path

import pytest

import pipecat.services
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.baseten.llm import BasetenLLMService
from pipecat.services.cerebras.llm import CerebrasLLMService
from pipecat.services.crusoe.llm import CrusoeLLMService
from pipecat.services.deepseek.llm import DeepSeekLLMService
from pipecat.services.fireworks.llm import FireworksLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.inception.llm import InceptionLLMService
from pipecat.services.mistral.llm import MistralLLMService
from pipecat.services.nebius.llm import NebiusLLMService
from pipecat.services.novita.llm import NovitaLLMService
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.omni.llm import NvidiaOmniLLMService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.perplexity.llm import PerplexityLLMService
from pipecat.services.qwen.llm import QwenLLMService
from pipecat.services.sambanova.llm import SambaNovaLLMService
from pipecat.services.sarvam.llm import SarvamLLMService
from pipecat.services.smallest.llm import SmallestLLMService
from pipecat.services.together.llm import TogetherLLMService
from pipecat.services.whisper.base_stt import BaseWhisperSTTService
from pipecat.services.xai.llm import GrokLLMService

PROVIDER_SERVICES = [
    BasetenLLMService,
    CerebrasLLMService,
    CrusoeLLMService,
    DeepSeekLLMService,
    FireworksLLMService,
    GrokLLMService,
    GroqLLMService,
    GroqSTTService,
    InceptionLLMService,
    MistralLLMService,
    NebiusLLMService,
    NovitaLLMService,
    OpenRouterLLMService,
    PerplexityLLMService,
    QwenLLMService,
    SambaNovaLLMService,
    SarvamLLMService,
    SmallestLLMService,
    TogetherLLMService,
]

# Services that may take no key: OpenAI's own read OPENAI_API_KEY by design,
# and the rest never hand the OpenAI client a missing key.
KEY_OPTIONAL_SERVICES = {
    AzureLLMService,
    NvidiaLLMService,
    NvidiaOmniLLMService,
    OLLamaLLMService,
    OpenAILLMService,
    OpenAISTTService,
}


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")


def _subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def test_every_provider_service_requires_api_key():
    services_dir = Path(pipecat.services.__file__).parent
    for path in [*services_dir.glob("*/llm.py"), *services_dir.glob("*/stt.py")]:
        try:
            importlib.import_module(f"pipecat.services.{path.parent.name}.{path.stem}")
        except Exception:
            pass

    services = {
        cls
        for base in (BaseOpenAILLMService, BaseWhisperSTTService)
        for cls in _subclasses(base)
        if cls.__module__.startswith("pipecat.services.")
    }
    unflagged = {
        cls.__name__
        for cls in services - KEY_OPTIONAL_SERVICES
        if not cls.__dict__.get("_requires_api_key", False)
    }
    assert not unflagged, f"Set `_requires_api_key = True` on: {sorted(unflagged)}"


@pytest.mark.parametrize("service_cls", [GroqSTTService, OpenRouterLLMService])
def test_api_key_argument_is_required(service_cls):
    with pytest.raises(TypeError):
        service_cls()


@pytest.mark.parametrize("api_key", [None, ""])
@pytest.mark.parametrize("service_cls", PROVIDER_SERVICES)
def test_missing_api_key_raises(service_cls, api_key):
    with pytest.raises(ValueError, match="requires an api_key"):
        service_cls(api_key=api_key)


@pytest.mark.parametrize("service_cls", [*PROVIDER_SERVICES, NvidiaLLMService])
def test_explicit_api_key_is_used(service_cls):
    service = service_cls(api_key="provider-key")
    assert service._client.api_key == "provider-key"


def test_openai_services_read_openai_api_key():
    assert OpenAILLMService()._client.api_key == "openai-key"
    assert OpenAISTTService()._client.api_key == "openai-key"


def test_nvidia_local_nim_without_key_does_not_use_openai_key():
    service = NvidiaLLMService(base_url="http://localhost:8000/v1")
    assert service._client.api_key != "openai-key"


def test_nvidia_local_nim_without_any_key_constructs(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    NvidiaLLMService(base_url="http://localhost:8000/v1")
