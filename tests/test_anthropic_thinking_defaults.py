#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the lowest-thinking default and effort setting in AnthropicLLMService."""

import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService


def _applied_default(model: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the request params after the service applies its default for a model."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model=model)
    )
    params = params if params is not None else {}

    service._maybe_apply_thinking_default(params)

    return params


def _applied_thinking(model: str) -> dict[str, Any] | None:
    """Return the thinking config the service applies for a model, if any."""
    return _applied_default(model).get("thinking")


def _log_lines(*models: str) -> list[str]:
    """Return what applying the default logs, at INFO and above, for each model in turn."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{level} {message}")
    try:
        service = AnthropicLLMService(
            api_key="test-key",
            settings=AnthropicLLMService.Settings(model=models[0]),
        )
        for model in models:
            service._settings.model = model
            service._maybe_apply_thinking_default({})
            service._maybe_apply_thinking_default({})
    finally:
        logger.remove(handler_id)
    return sink.getvalue().splitlines()


async def _request(service: AnthropicLLMService, **kwargs) -> dict[str, Any]:
    """Return the request params run_inference sends for a service."""
    service._client = AsyncMock()
    service._client.beta.messages.create.return_value = SimpleNamespace(content=[])

    await service.run_inference(LLMContext(messages=[{"role": "user", "content": "hi"}]), **kwargs)

    return service._client.beta.messages.create.call_args.kwargs


# --- default thinking config per model --------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
)
def test_models_that_allow_it_get_thinking_disabled(model):
    """Turning thinking off is the lowest setting wherever a model allows it."""
    assert _applied_default(model) == {"thinking": {"type": "disabled"}}


def test_sonnet_5_5_gets_between_tools():
    """Sonnet 5.5 rejects "disabled"; "between_tools" is its lowest setting."""
    assert _applied_default("claude-sonnet-5-5") == {"thinking": {"type": "between_tools"}}


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1", "claude-fable-5"])
def test_models_that_always_think_get_low_effort(model):
    """Models whose thinking can't be turned off get the lowest effort instead."""
    assert _applied_default(model) == {"output_config": {"effort": "low"}}


def test_low_effort_joins_a_response_schema_in_output_config():
    """The effort default doesn't displace a response schema already in output_config."""
    output_format = {"type": "json_schema", "schema": {"type": "object"}}
    params = {"output_config": {"format": output_format}}

    _applied_default("claude-opus-5-5", params)

    assert params["output_config"] == {"format": output_format, "effort": "low"}


def test_every_id_form_of_a_model_is_recognized():
    """Bedrock prefixes, Vertex versions and dated snapshots name the same model."""
    for model in (
        "anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
        "claude-sonnet-5@20260630",
        "claude-sonnet-5-20260630",
        "claude-haiku-4-5-20251001",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ):
        assert _applied_thinking(model) == {"type": "disabled"}, model


def test_unlisted_models_run_at_anthropics_default():
    """Only listed models get a thinking default, so a new model is never guessed at."""
    for model in ("claude-sonnet-6", "claude-mythos-5", "claude-3-5-sonnet-20241022"):
        assert _applied_default(model) == {}, model


# --- explicit configuration wins --------------------------------------------


@pytest.mark.asyncio
async def test_a_configured_thinking_config_is_left_alone():
    """An explicit thinking config wins over the low-latency default."""
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-5",
            thinking=AnthropicLLMService.ThinkingConfig(type="adaptive", display="summarized"),
        ),
    )

    request = await _request(service)

    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}


@pytest.mark.asyncio
async def test_thinking_passed_through_extra_is_left_alone():
    """A thinking config in extra wins too."""
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-5", extra={"thinking": {"type": "adaptive"}}
        ),
    )

    request = await _request(service)

    assert request["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_a_configured_effort_replaces_the_thinking_default():
    """Effort and thinking together decide how much the model thinks, so either one wins."""
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(model="claude-sonnet-5", effort="low"),
    )

    request = await _request(service)

    assert "thinking" not in request
    assert request["output_config"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_effort_passed_through_extra_replaces_the_thinking_default():
    """An effort in extra wins too."""
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-5", extra={"output_config": {"effort": "low"}}
        ),
    )

    request = await _request(service)

    assert "thinking" not in request
    assert request["output_config"] == {"effort": "low"}


# --- effort setting ------------------------------------------------------------


@pytest.mark.asyncio
async def test_effort_is_omitted_when_unset():
    """Without an effort setting, only a default can add one."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5")
    )

    request = await _request(service)

    assert "output_config" not in request


@pytest.mark.asyncio
async def test_effort_and_response_schema_share_output_config():
    """A response schema doesn't displace the effort setting, nor the reverse."""
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(model="claude-opus-5-5", effort="medium"),
    )

    request = await _request(service, response_schema=schema)

    assert request["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": schema},
    }


# --- logging -------------------------------------------------------------------


def test_the_applied_default_is_logged_once_per_model():
    """The developer learns what Pipecat sent, and when to change it, once per model."""
    lines = _log_lines("claude-sonnet-5", "claude-sonnet-5-20260630", "claude-opus-5-5")

    assert len(lines) == 3
    assert all(line.startswith("INFO") for line in lines)
    assert "thinking={'type': 'disabled'} to claude-sonnet-5," in lines[0]
    assert "multistep reasoning" in lines[0]
    assert "set `thinking` or `effort`" in lines[0]
    assert "to claude-sonnet-5-20260630," in lines[1]
    assert "output_config={'effort': 'low'} to claude-opus-5-5," in lines[2]


def test_an_unlisted_claude_model_is_warned_about_once():
    """A Claude model without a default may think before replying, so the developer hears."""
    lines = _log_lines("claude-sonnet-6")

    assert len(lines) == 1
    assert lines[0].startswith("WARNING")
    assert "no thinking default for claude-sonnet-6" in lines[0]


def test_nothing_is_logged_for_a_model_id_that_names_no_claude_model():
    """A Bedrock inference profile ARN doesn't say which model it runs."""
    profile = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    assert _log_lines(profile) == []


def test_nothing_is_logged_when_reasoning_is_configured():
    """An explicit setting is the developer's choice, and needs no comment."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-6")
    )
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{message}")
    try:
        service._maybe_apply_thinking_default({"thinking": {"type": "adaptive"}})
        service._maybe_apply_thinking_default({"output_config": {"effort": "low"}})
    finally:
        logger.remove(handler_id)

    assert sink.getvalue() == ""


# --- use_provider_thinking_default -------------------------------------------


def test_use_provider_thinking_default_runs_at_anthropics_default():
    """The setting sends no default and logs the reason at DEBUG."""
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(
            model="claude-opus-5-5", use_provider_thinking_default=True
        ),
    )
    params: dict[str, Any] = {}
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{level} {message}")
    try:
        service._maybe_apply_thinking_default(params)
        service._maybe_apply_thinking_default(params)
    finally:
        logger.remove(handler_id)
    lines = sink.getvalue().splitlines()

    assert params == {}
    assert len(lines) == 1
    assert lines[0].startswith("DEBUG")
    assert "use_provider_thinking_default is set, so claude-opus-5-5 runs" in lines[0]


# --- every inference path ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_inference_applies_the_thinking_default():
    """Out-of-band inference gets the same default as the in-pipeline path."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5")
    )

    request = await _request(service)

    assert request["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_streaming_applies_the_thinking_default_and_effort():
    """The in-pipeline request carries the default, and the effort setting when set."""
    requests: list[dict[str, Any]] = []

    async def fake_stream(api_call, params):
        requests.append(params)

        async def no_events():
            return
            yield

        return no_events()

    async def drop_frame(frame, direction=None):
        pass

    for settings in (
        AnthropicLLMService.Settings(model="claude-sonnet-5"),
        AnthropicLLMService.Settings(model="claude-sonnet-5-5", effort="low"),
    ):
        service = AnthropicLLMService(api_key="test-key", settings=settings)
        with (
            patch.object(service, "push_frame", drop_frame),
            patch.object(service, "run_function_calls", AsyncMock()),
            patch.object(service, "_create_message_stream", fake_stream),
        ):
            await service._process_context(LLMContext())

    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "output_config" not in requests[0]
    assert "thinking" not in requests[1]
    assert requests[1]["output_config"] == {"effort": "low"}
