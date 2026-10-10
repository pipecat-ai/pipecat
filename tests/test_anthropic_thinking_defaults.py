#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the low-latency thinking default and effort setting in AnthropicLLMService."""

import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService


def _applied_thinking(model: str) -> dict[str, Any] | None:
    """Return the thinking config the service applies for a model, if any."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model=model)
    )
    params: dict[str, Any] = {}

    service._maybe_apply_thinking_default(params)

    return params.get("thinking")


async def _request(service: AnthropicLLMService, **kwargs) -> dict[str, Any]:
    """Return the request params run_inference sends for a service."""
    service._client = AsyncMock()
    service._client.beta.messages.create.return_value = SimpleNamespace(content=[])

    await service.run_inference(LLMContext(messages=[{"role": "user", "content": "hi"}]), **kwargs)

    return service._client.beta.messages.create.call_args.kwargs


# --- default thinking config per model --------------------------------------


def test_sonnet_5_disables_thinking():
    """Sonnet 5 thinks before most tool calls unless told not to, so it gets told not to."""
    assert _applied_thinking("claude-sonnet-5") == {"type": "disabled"}


def test_every_id_form_of_sonnet_5_is_recognized():
    """Bedrock prefixes, Vertex versions and dated snapshots name the same model."""
    assert _applied_thinking("anthropic.claude-sonnet-5") == {"type": "disabled"}
    assert _applied_thinking("us.anthropic.claude-sonnet-5") == {"type": "disabled"}
    assert _applied_thinking("claude-sonnet-5@20260630") == {"type": "disabled"}
    assert _applied_thinking("claude-sonnet-5-20260630") == {"type": "disabled"}


def test_sonnet_5_5_runs_at_anthropics_default():
    """Sonnet 5.5 rejects "disabled", and its default is already the fastest setting."""
    assert _applied_thinking("claude-sonnet-5-5") is None
    assert _applied_thinking("anthropic.claude-sonnet-5-5") is None


def test_unlisted_models_run_at_anthropics_default():
    """Only listed models get a thinking default, so a new model is never guessed at."""
    for model in (
        "claude-sonnet-6",
        "claude-haiku-5-5",
        "claude-opus-5",
        "claude-fable-5-1",
        "claude-sonnet-4-6",
        "claude-3-5-sonnet-20241022",
    ):
        assert _applied_thinking(model) is None, model


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
    """Without an effort setting the model's default effort applies."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-opus-5-5")
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
    """The developer learns what Pipecat sent, without a log line per request."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{message}")
    try:
        service = AnthropicLLMService(
            api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5")
        )
        service._maybe_apply_thinking_default({})
        service._maybe_apply_thinking_default({})
        service._settings.model = "claude-sonnet-5-20260630"
        service._maybe_apply_thinking_default({})
    finally:
        logger.remove(handler_id)

    lines = [line for line in sink.getvalue().splitlines() if "thinking=" in line]
    assert len(lines) == 2
    assert "claude-sonnet-5 " in lines[0]
    assert "Set `thinking` or `effort`" in lines[0]
    assert "claude-sonnet-5-20260630" in lines[1]


def test_nothing_is_logged_for_a_model_without_a_default():
    """A model running at Anthropic's default needs no explanation."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="INFO", format="{message}")
    try:
        service = AnthropicLLMService(
            api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5-5")
        )
        service._maybe_apply_thinking_default({})
    finally:
        logger.remove(handler_id)

    assert "thinking=" not in sink.getvalue()


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
