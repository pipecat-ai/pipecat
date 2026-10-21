#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the lowest-thinking defaults in GoogleLLMService."""

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


def _log_lines(*models: str) -> list[str]:
    """Return what building params logs, at INFO and above, for each model in turn."""
    service = GoogleLLMService(
        api_key="test-key", settings=GoogleLLMService.Settings(model=models[0])
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{level} {message}")
    try:
        for model in models:
            service._settings.model = model
            service._build_generation_params()
            service._build_generation_params()
    finally:
        logger.remove(handler_id)
    return sink.getvalue().splitlines()


# --- default thinking config per model --------------------------------------


def test_gemini_25_flash_disables_thinking_with_a_budget():
    """The 2.5 series takes a budget, and zero turns thinking off."""
    assert _applied_thinking_config("gemini-2.5-flash") == {"thinking_budget": 0}
    assert _applied_thinking_config("gemini-2.5-flash-lite") == {"thinking_budget": 0}


def test_gemini_25_pro_gets_its_minimum_budget():
    """2.5 Pro can't turn thinking off, and 128 tokens is the least it accepts."""
    assert _applied_thinking_config("gemini-2.5-pro") == {"thinking_budget": 128}


@pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemma-4-31b-it"])
def test_models_that_accept_it_use_the_minimal_level(model):
    """Gemini 3 and Gemma 4 models take a level, and minimal is the lowest."""
    assert _applied_thinking_config(model) == {"thinking_level": "minimal"}


def test_gemini_31_pro_uses_the_lowest_level_it_accepts():
    """3.1 Pro doesn't support minimal, so it gets low instead."""
    assert _applied_thinking_config("gemini-3.1-pro-preview") == {"thinking_level": "low"}


def test_gemini_37_flash_uses_the_lowest_level_it_accepts():
    """3.7 Flash rejects minimal outright, so it gets low instead."""
    assert _applied_thinking_config("gemini-3.7-flash") == {"thinking_level": "low"}


def test_gemini_38_flash_uses_the_lowest_level_it_accepts():
    """3.8 Flash rejects minimal outright, so it gets low instead."""
    assert _applied_thinking_config("gemini-3.8-flash") == {"thinking_level": "low"}


def test_gemini_4_argon_uses_the_lowest_level_it_accepts():
    """4 Argon rejects minimal outright, so it gets low instead."""
    assert _applied_thinking_config("gemini-4-argon") == {"thinking_level": "low"}


def test_every_id_form_of_a_listed_model_is_recognized():
    """Resource paths and version suffixes name the same model."""
    assert _applied_thinking_config("models/gemini-3.6-flash") == {"thinking_level": "minimal"}
    assert _applied_thinking_config("publishers/google/models/gemini-3.6-flash") == {
        "thinking_level": "minimal"
    }
    assert _applied_thinking_config("gemini-2.5-flash-001") == {"thinking_budget": 0}


def test_unlisted_models_run_at_googles_default():
    """Only listed models get a thinking default, so a new model is never guessed at."""
    for model in ("gemini-3.9-flash", "gemini-flash-latest", "gemini-3.1-flash-image"):
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
    """The developer learns what Pipecat sent, and when to change it, once per model."""
    lines = _log_lines("gemini-3.6-flash", "gemini-3.8-flash")

    assert len(lines) == 2
    assert all(line.startswith("INFO") for line in lines)
    assert "thinking_config={'thinking_level': 'minimal'} to gemini-3.6-flash," in lines[0]
    assert "multistep reasoning" in lines[0]
    assert "set `thinking` in GoogleLLMService.Settings" in lines[0]
    assert "to gemini-3.8-flash," in lines[1]


@pytest.mark.parametrize("model", ["gemini-3.9-flash", "gemini-flash-latest", "gemma-4-e4b-it"])
def test_an_unlisted_model_that_can_think_is_warned_about_once(model):
    """A model without a default may think before replying, so the developer hears."""
    lines = _log_lines(model)

    assert len(lines) == 1
    assert lines[0].startswith("WARNING")
    assert f"no thinking default for {model}" in lines[0]


@pytest.mark.parametrize("model", ["gemini-2.0-flash", "gemini-3.1-flash-image"])
def test_nothing_is_logged_for_a_model_that_cant_think(model):
    """Models without thinking have no latency to recover."""
    assert _log_lines(model) == []


def test_nothing_is_logged_when_thinking_is_configured():
    """An explicit setting is the developer's choice, and needs no comment."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(
            model="gemini-3.9-flash", thinking=GoogleLLMService.ThinkingConfig(thinking_level="low")
        ),
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{message}")
    try:
        service._build_generation_params()
    finally:
        logger.remove(handler_id)

    assert sink.getvalue() == ""


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


def test_thinking_budget_on_a_gemma_4_model_warns():
    """Gemma 4 takes a level too."""
    output = _warnings_from(
        lambda: GoogleLLMService(
            api_key="test-key",
            settings=GoogleLLMService.Settings(
                model="models/gemma-4-31b-it",
                thinking=GoogleLLMService.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    assert "thinking_budget" in output


def test_thinking_budget_on_a_gemini_4_model_warns():
    """Gemini 4 takes a level, and rejects a budget."""
    output = _warnings_from(
        lambda: GoogleLLMService(
            api_key="test-key",
            settings=GoogleLLMService.Settings(
                model="gemini-4-argon",
                thinking=GoogleLLMService.ThinkingConfig(thinking_budget=0),
            ),
        )
    )

    assert "thinking_budget" in output


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

    assert service._settings.model == "gemini-3.8-flash"
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


# --- use_provider_thinking_default -------------------------------------------


def test_use_provider_thinking_default_runs_at_googles_default():
    """The setting sends no default and logs the reason at DEBUG."""
    service = GoogleLLMService(
        api_key="test-key",
        settings=GoogleLLMService.Settings(
            model="gemini-3.6-flash", use_provider_thinking_default=True
        ),
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{level} {message}")
    try:
        params = service._build_generation_params()
        service._build_generation_params()
    finally:
        logger.remove(handler_id)
    lines = sink.getvalue().splitlines()

    assert "thinking_config" not in params
    assert len(lines) == 1
    assert lines[0].startswith("DEBUG")
    assert "use_provider_thinking_default is set, so gemini-3.6-flash runs" in lines[0]
