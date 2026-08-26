#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the low-latency thinking default in AnthropicLLMService."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.anthropic.llm import AnthropicLLMService


def _applied_thinking(model: str) -> dict[str, Any] | None:
    """Return the thinking config the service applies for a model, if any."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model=model)
    )
    params: dict[str, Any] = {}

    service._maybe_disable_thinking(params)

    return params.get("thinking")


async def _requested_thinking(service: AnthropicLLMService) -> dict[str, Any] | None:
    """Return the thinking config run_inference sends for a service."""
    service._client = AsyncMock()
    service._client.beta.messages.create.return_value = SimpleNamespace(content=[])

    await service.run_inference(LLMContext(messages=[{"role": "user", "content": "hi"}]))

    return service._client.beta.messages.create.call_args.kwargs.get("thinking")


# --- default thinking config per model --------------------------------------


def test_sonnet_5_disables_thinking():
    """Sonnet 5 thinks unless told not to, so it gets told not to."""
    assert _applied_thinking("claude-sonnet-5") == {"type": "disabled"}


def test_later_sonnet_generations_disable_thinking():
    """A Sonnet newer than 5 is assumed to think by default too."""
    assert _applied_thinking("claude-sonnet-6") == {"type": "disabled"}


def test_every_id_form_of_sonnet_5_is_recognized():
    """Bedrock prefixes and dated snapshots name the same model."""
    assert _applied_thinking("anthropic.claude-sonnet-5") == {"type": "disabled"}
    assert _applied_thinking("claude-sonnet-5-20260630") == {"type": "disabled"}


def test_sonnet_4_6_gets_no_thinking_default():
    """Earlier Sonnets think only when asked, so there is nothing to turn off."""
    assert _applied_thinking("claude-sonnet-4-6") is None


def test_pre_4_sonnet_ids_get_no_thinking_default():
    """Ids that put the generation before the name are not mistaken for Sonnet 20."""
    assert _applied_thinking("claude-3-5-sonnet-20241022") is None


def test_opus_and_fable_get_no_thinking_default():
    """Only the Sonnet line trades reasoning for latency by default."""
    assert _applied_thinking("claude-opus-5") is None
    assert _applied_thinking("claude-fable-5") is None


def test_haiku_gets_no_thinking_default():
    """Haiku thinks only when asked."""
    assert _applied_thinking("claude-haiku-4-5") is None


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

    assert await _requested_thinking(service) == {"type": "adaptive", "display": "summarized"}


@pytest.mark.asyncio
async def test_thinking_passed_through_extra_is_left_alone():
    """A thinking config in extra wins too."""
    service = AnthropicLLMService(
        api_key="test-key",
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-5", extra={"thinking": {"type": "adaptive"}}
        ),
    )

    assert await _requested_thinking(service) == {"type": "adaptive"}


# --- every inference path ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_inference_applies_the_thinking_default():
    """Out-of-band inference gets the same default as the in-pipeline path."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5")
    )

    assert await _requested_thinking(service) == {"type": "disabled"}


@pytest.mark.asyncio
async def test_streaming_applies_the_thinking_default():
    """The in-pipeline request carries the default too."""
    service = AnthropicLLMService(
        api_key="test-key", settings=AnthropicLLMService.Settings(model="claude-sonnet-5")
    )
    requests: list[dict[str, Any]] = []

    async def fake_stream(api_call, params):
        requests.append(params)

        async def no_events():
            return
            yield

        return no_events()

    async def drop_frame(frame, direction=None):
        pass

    with (
        patch.object(service, "push_frame", drop_frame),
        patch.object(service, "run_function_calls", AsyncMock()),
        patch.object(service, "_create_message_stream", fake_stream),
    ):
        await service._process_context(LLMContext())

    assert requests[0]["thinking"] == {"type": "disabled"}
