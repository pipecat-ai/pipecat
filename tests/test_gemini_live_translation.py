#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Gemini Live translation config reaching the Live API connection."""

from dataclasses import fields
from unittest.mock import patch

import pytest

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

# TranslationParams is imported inside each test that needs it rather than at module level, so
# this file still collects against a tree that does not have it. A module that fails to import
# reports one error for the whole file and says nothing about which behavior is missing.


def make_service(**settings_kwargs):
    """Build a service and capture the config it hands to the connection task.

    The real constructor runs, so the store-mode settings object is seeded the way production
    seeds it. ``create_client`` only builds a client object and makes no request.
    """
    service = GeminiLiveLLMService(
        api_key="dummy",
        settings=GeminiLiveLLMService.Settings(**settings_kwargs),
    )

    captured = {}
    errors = []

    def connection_task_handler(config):
        captured["config"] = config
        return "not-awaited"

    def create_task(coro, name=None):
        return coro

    async def push_error(error_msg=None, exception=None):
        errors.append(error_msg)

    service._connection_task_handler = connection_task_handler
    service.create_task = create_task
    service.push_error = push_error

    return service, captured, errors


@pytest.mark.asyncio
async def test_translation_params_reach_the_connect_config():
    from pipecat.services.google.gemini_live.llm import TranslationParams

    service, captured, errors = make_service(
        translation=TranslationParams(target_language_code="es-US", echo_target_language=True)
    )

    await service._connect()

    assert errors == []
    config = captured["config"]
    assert config.translation_config is not None
    assert config.translation_config.target_language_code == "es-US"
    assert config.translation_config.echo_target_language is True


@pytest.mark.asyncio
async def test_translation_accepts_a_dict():
    service, captured, errors = make_service(translation={"target_language_code": "fr"})

    await service._connect()

    assert errors == []
    config = captured["config"]
    assert config.translation_config is not None
    assert config.translation_config.target_language_code == "fr"
    # Left unset, so the model keeps its own behavior rather than being told False.
    assert config.translation_config.echo_target_language is None


@pytest.mark.asyncio
async def test_no_translation_leaves_the_config_unset():
    service, captured, errors = make_service()

    await service._connect()

    assert errors == []
    assert captured["config"].translation_config is None


@pytest.mark.asyncio
async def test_translation_on_an_older_google_genai_reports_the_requirement():
    from pipecat.services.google.gemini_live.llm import TranslationParams

    service, captured, errors = make_service(
        translation=TranslationParams(target_language_code="es-US")
    )

    with patch("pipecat.services.google.gemini_live.llm.TranslationConfig", None):
        await service._connect()

    assert "config" not in captured
    assert len(errors) == 1
    assert "google-genai >= 2.8.0" in errors[0]


def test_vertex_service_carries_the_translation_setting():
    """The issue asks about Vertex, which reaches this through inheritance."""
    from pipecat.services.google.gemini_live.vertex.llm import GeminiLiveVertexLLMService

    assert "translation" in {f.name for f in fields(GeminiLiveVertexLLMService.Settings)}
    assert GeminiLiveVertexLLMService._connect is GeminiLiveLLMService._connect
