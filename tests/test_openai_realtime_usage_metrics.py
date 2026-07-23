#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenAI Realtime token usage reporting.

Covers:
- ``response.done`` usage details (audio/cached breakdown) land on the
  ``LLMTokenUsage`` passed to ``start_llm_usage_metrics``.
- Missing detail objects degrade to ``None`` fields rather than errors.
- ``_add_token_usage_to_span`` emits the audio token span attributes for
  both ``LLMTokenUsage`` objects and plain dicts.
"""

from unittest.mock import AsyncMock

import pytest

from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.utils.tracing.service_decorators import _add_token_usage_to_span

# ---------------------------------------------------------------------------
# response.done -> LLMTokenUsage
# ---------------------------------------------------------------------------


def _response_done_evt(usage: dict) -> events.ResponseDone:
    return events.ResponseDone.model_validate(
        {
            "event_id": "ev_1",
            "type": "response.done",
            "response": {
                "id": "resp_1",
                "object": "realtime.response",
                "status": "completed",
                "status_details": None,
                "output": [],
                "usage": usage,
            },
        }
    )


def _service_for_usage_capture() -> OpenAIRealtimeLLMService:
    service = OpenAIRealtimeLLMService(
        api_key="test-key",
        settings=OpenAIRealtimeLLMService.Settings(model="gpt-realtime"),
    )
    service.start_llm_usage_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.push_frame = AsyncMock()
    service._call_event_handler = AsyncMock()
    service._current_audio_response = None
    return service


@pytest.mark.asyncio
async def test_response_done_reports_audio_and_cached_audio_tokens():
    service = _service_for_usage_capture()
    evt = _response_done_evt(
        {
            "total_tokens": 100,
            "input_tokens": 60,
            "output_tokens": 40,
            "input_token_details": {
                "cached_tokens": 30,
                "text_tokens": 20,
                "audio_tokens": 40,
                "cached_tokens_details": {"text_tokens": 10, "audio_tokens": 20},
            },
            "output_token_details": {"text_tokens": 15, "audio_tokens": 25},
        }
    )

    await service._handle_evt_response_done(evt)

    tokens: LLMTokenUsage = service.start_llm_usage_metrics.call_args.args[0]
    assert tokens.prompt_tokens == 60
    assert tokens.completion_tokens == 40
    assert tokens.total_tokens == 100
    assert tokens.cache_read_input_tokens == 30
    assert tokens.input_audio_tokens == 40
    assert tokens.output_audio_tokens == 25
    assert tokens.cache_read_input_audio_tokens == 20


@pytest.mark.asyncio
async def test_response_done_without_cached_details_reports_none():
    service = _service_for_usage_capture()
    evt = _response_done_evt(
        {
            "total_tokens": 10,
            "input_tokens": 6,
            "output_tokens": 4,
            "input_token_details": {"cached_tokens": 0, "text_tokens": 6, "audio_tokens": 0},
            "output_token_details": {"text_tokens": 4, "audio_tokens": 0},
        }
    )

    await service._handle_evt_response_done(evt)

    tokens: LLMTokenUsage = service.start_llm_usage_metrics.call_args.args[0]
    assert tokens.cache_read_input_audio_tokens is None
    assert tokens.input_audio_tokens == 0
    assert tokens.output_audio_tokens == 0


# ---------------------------------------------------------------------------
# _add_token_usage_to_span
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_span_attributes_from_llm_token_usage_object():
    span = _FakeSpan()
    _add_token_usage_to_span(
        span,
        LLMTokenUsage(
            prompt_tokens=60,
            completion_tokens=40,
            total_tokens=100,
            cache_read_input_tokens=30,
            input_audio_tokens=40,
            output_audio_tokens=25,
            cache_read_input_audio_tokens=20,
        ),
    )
    assert span.attributes["gen_ai.usage.input_tokens"] == 60
    assert span.attributes["gen_ai.usage.output_tokens"] == 40
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 30
    assert span.attributes["gen_ai.usage.audio.input_tokens"] == 40
    assert span.attributes["gen_ai.usage.audio.output_tokens"] == 25
    assert span.attributes["gen_ai.usage.audio.cache_read.input_tokens"] == 20


def test_span_attributes_omitted_when_audio_fields_unset():
    span = _FakeSpan()
    _add_token_usage_to_span(
        span,
        LLMTokenUsage(prompt_tokens=6, completion_tokens=4, total_tokens=10),
    )
    assert "gen_ai.usage.audio.input_tokens" not in span.attributes
    assert "gen_ai.usage.audio.output_tokens" not in span.attributes
    assert "gen_ai.usage.audio.cache_read.input_tokens" not in span.attributes


def test_span_attributes_from_dict():
    span = _FakeSpan()
    _add_token_usage_to_span(
        span,
        {
            "prompt_tokens": 60,
            "completion_tokens": 40,
            "input_audio_tokens": 40,
            "output_audio_tokens": 25,
            "cache_read_input_audio_tokens": 20,
        },
    )
    assert span.attributes["gen_ai.usage.audio.input_tokens"] == 40
    assert span.attributes["gen_ai.usage.audio.output_tokens"] == 25
    assert span.attributes["gen_ai.usage.audio.cache_read.input_tokens"] == 20
