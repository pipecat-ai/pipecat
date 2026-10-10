#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the low-latency thinking defaults in GoogleLLMService."""

import io
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from google.genai.types import ThinkingLevel
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.vertex.llm import GoogleVertexLLMService


def _applied_thinking_config(model: str) -> dict[str, Any] | None:
    """Return the thinking config the service applies for a model, if any."""
    service = GoogleLLMService(api_key="test-key", settings=GoogleLLMService.Settings(model=model))

    params = service._build_generation_params()

    return params.get("thinking_config")


def _warnings_from(build: Callable[[], Any]) -> str:
    """Return the WARNING-level log output produced while calling build."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        build()
    finally:
        logger.remove(handler_id)
    return sink.getvalue()


# --- default thinking config per model --------------------------------------


def test_gemini_25_flash_disables_thinking_with_a_budget():
    """The 2.5 series takes a budget, and zero turns thinking off."""
    assert _applied_thinking_config("gemini-2.5-flash") == {"thinking_budget": 0}


def test_gemini_3_flash_uses_the_minimal_level():
    """Gemini 3 flash models take a level, and minimal is the fastest."""
    assert _applied_thinking_config("gemini-3.6-flash") == {"thinking_level": "minimal"}


def test_gemini_37_flash_uses_the_lowest_level_it_accepts():
    """3.7 Flash rejects minimal outright, so it gets low instead."""
    assert _applied_thinking_config("gemini-3.7-flash") == {"thinking_level": "low"}


def test_gemini_38_flash_uses_the_lowest_level_it_accepts():
    """3.8 Flash rejects minimal outright, so it gets low instead."""
    assert _applied_thinking_config("gemini-3.8-flash") == {"thinking_level": "low"}


def test_every_id_form_of_a_listed_model_is_recognized():
    """Resource paths and version suffixes name the same model."""
    assert _applied_thinking_config("models/gemini-3.6-flash") == {"thinking_level": "minimal"}
    assert _applied_thinking_config("publishers/google/models/gemini-3.6-flash") == {
        "thinking_level": "minimal"
    }
    assert _applied_thinking_config("gemini-2.5-flash-001") == {"thinking_budget": 0}


def test_unlisted_models_run_at_googles_default():
    """Only listed models get a thinking default, so a new model is never guessed at."""
    for model in (
        "gemini-3.9-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-image",
    ):
        assert _applied_thinking_config(model) is None, model


def test_thinking_config_in_extra_is_left_alone():
    """A thinking config passed through extra wins too."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(
            model="gemini-3.6-flash", extra={"thinking_config": {"thinking_level": "high"}}
        ),
    )

    params = service._build_generation_params()

    assert params["thinking_config"] == {"thinking_level": "high"}


def test_the_applied_default_is_logged_once_per_model():
    """The developer learns what Pipecat sent, without a log line per request."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-3.6-flash")
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{message}")
    try:
        service._build_generation_params()
        service._build_generation_params()
        service._settings.model = "gemini-3.8-flash"
        service._build_generation_params()
    finally:
        logger.remove(handler_id)

    lines = [line for line in sink.getvalue().splitlines() if "thinking_config=" in line]
    assert len(lines) == 2
    assert "gemini-3.6-flash" in lines[0]
    assert "Set `thinking` in GoogleLLMService.Settings" in lines[0]
    assert "gemini-3.8-flash" in lines[1]


def test_nothing_is_logged_for_a_model_without_a_default():
    """A model running at Google's default needs no explanation."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model="gemini-3.5-flash-lite")
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{message}")
    try:
        service._build_generation_params()
    finally:
        logger.remove(handler_id)

    assert "thinking_config=" not in sink.getvalue()


def test_a_configured_thinking_config_is_left_alone():
    """An explicit thinking config wins over the low-latency default."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(
            model="gemini-3.7-flash",
            thinking=GoogleLLMService.ThinkingConfig(thinking_level="high"),
        ),
    )

    params = service._build_generation_params()

    assert params["thinking_config"] == {"thinking_level": "high"}


# --- every inference path ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_inference_applies_the_thinking_default():
    """Out-of-band inference gets the same default as the in-pipeline path."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(
            model="gemini-3.6-flash", system_instruction="You are helpful."
        ),
    )
    response = SimpleNamespace(candidates=[])

    with patch.object(
        service._client.aio.models, "generate_content", return_value=response
    ) as generate:
        await service.run_inference(LLMContext(messages=[{"role": "user", "content": "hi"}]))

    config = generate.call_args.kwargs["config"]
    assert config.thinking_config.thinking_level == ThinkingLevel.MINIMAL


# --- warning on a budget that may not control thinking ----------------------


def test_thinking_budget_on_a_gemini_3_model_warns():
    """Gemini 3 takes a level, so a budget set on one may not apply."""
    output = _warnings_from(
        lambda: GoogleLLMService(
            api_key="test-key",
            settings=GoogleLLMService.Settings(
                model="gemini-3.6-flash",
                thinking=GoogleLLMService.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    assert "thinking_budget" in output
    assert "gemini-3.6-flash" in output
    assert "thinking_level" in output


def test_thinking_budget_on_a_gemini_25_model_does_not_warn():
    """The 2.5 series honors a budget, so there is nothing to warn about."""
    output = _warnings_from(
        lambda: GoogleLLMService(
            api_key="test-key",
            settings=GoogleLLMService.Settings(
                model="gemini-2.5-flash",
                thinking=GoogleLLMService.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    assert "thinking_budget" not in output


def test_thinking_level_on_a_gemini_3_model_does_not_warn():
    """A level is the right control for Gemini 3."""
    output = _warnings_from(
        lambda: GoogleLLMService(
            api_key="test-key",
            settings=GoogleLLMService.Settings(
                model="gemini-3.6-flash",
                thinking=GoogleLLMService.ThinkingConfig(thinking_level="low"),
            ),
        )
    )

    assert "thinking_budget" not in output


@pytest.mark.asyncio
async def test_switching_to_a_gemini_3_model_at_runtime_warns():
    """Changing the model re-checks the thinking configuration against it."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(
            model="gemini-2.5-flash",
            thinking=GoogleLLMService.ThinkingConfig(thinking_budget=0),
        ),
    )

    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")
    try:
        await service._update_settings(GoogleLLMService.Settings(model="gemini-3.6-flash"))
    finally:
        logger.remove(handler_id)

    assert "thinking_budget" in sink.getvalue()


# --- Vertex defaults --------------------------------------------------


def _vertex_service(**kwargs) -> GoogleVertexLLMService:
    with (
        patch.object(GoogleVertexLLMService, "_get_credentials", return_value=None),
        patch.object(GoogleVertexLLMService, "create_client"),
    ):
        return GoogleVertexLLMService(project_id="test-project", **kwargs)


def test_vertex_defaults_to_gemini_3_on_the_global_endpoint():
    """Vertex serves the Gemini 3 series only from global, so both defaults pair."""
    service = _vertex_service()

    assert service._settings.model == "gemini-3.6-flash"
    assert service._location == "global"


def test_vertex_shares_the_thinking_defaults():
    """The Vertex service picks its thinking default from the same per-model table."""
    service = _vertex_service(settings=GoogleVertexLLMService.Settings(model="gemini-3.7-flash"))

    params = service._build_generation_params()

    assert params["thinking_config"] == {"thinking_level": "low"}


def test_vertex_warns_on_a_thinking_budget_for_gemini_3():
    """The warning covers the Vertex service too."""
    output = _warnings_from(
        lambda: _vertex_service(
            settings=GoogleVertexLLMService.Settings(
                model="gemini-3.6-flash",
                thinking=GoogleVertexLLMService.ThinkingConfig(thinking_budget=0),
            )
        )
    )

    assert "thinking_budget" in output
