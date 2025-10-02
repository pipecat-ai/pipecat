"""Tests for CometAPILLMService helper utilities.

These are pure unit tests (no network) validating:

- `recommended_models()` preserves order & uniqueness.
- `ignore_patterns()` returns a non-empty list and filters appropriately.
- `is_chat_model()` respects ignore patterns.
- `fetch_chat_models()` merges recommended when remote list is empty (mock).

Network calls are avoided by monkeypatching the underlying OpenAI client method.
"""
from __future__ import annotations

import types
from typing import List

import pytest

from pipecat.services.cometapi.llm import CometAPILLMService


def test_recommended_models_order_and_uniqueness():
    models = CometAPILLMService.recommended_models()
    assert models, "Expected non-empty recommended model list"
    # Preserve order: list should equal list of dict.fromkeys (first occurrence kept)
    assert models == list(dict.fromkeys(models)), "Recommended models contain duplicates"


def test_ignore_patterns_and_is_chat_model():
    patterns = CometAPILLMService.ignore_patterns()
    assert patterns, "Ignore patterns should not be empty"

    # A clearly text/chat model that should pass
    assert CometAPILLMService.is_chat_model("gpt-5-chat-latest") is True
    # A diffusion / image model variant should be filtered
    assert CometAPILLMService.is_chat_model("stable-diffusion-xl") is False
    # Embedding model should be filtered
    assert CometAPILLMService.is_chat_model("text-embedding-3-large") is False


@pytest.mark.asyncio
async def test_fetch_chat_models_merges_recommended(monkeypatch):
    # Provide a fake client.models.list() returning an empty list so only recommended appear
    class DummyModelsListResponse:
        data: List[types.SimpleNamespace] = []

    class DummyModelsClient:
        async def list(self):  # noqa: D401 - simple stub
            return DummyModelsListResponse()

    class DummyUnderlyingClient:
        models = DummyModelsClient()

    service = CometAPILLMService(api_key="dummy", model="gpt-5-chat-latest")
    # Monkeypatch the protected underlying client used by fetch_chat_models
    service._client = DummyUnderlyingClient()  # type: ignore[attr-defined]  # noqa: SLF001

    models = await service.fetch_chat_models()
    recommended = CometAPILLMService.recommended_models()
    # Should contain at least the recommended (and start with them in same order)
    assert models[: len(recommended)] == recommended
    assert len(models) == len(recommended)
