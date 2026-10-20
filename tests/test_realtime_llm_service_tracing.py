#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service-level tests for ``traced_realtime_llm``.

The real services' handlers are driven with fake server messages, after
``setup()`` with a ``TaskManager``, and the exported ``llm_response`` spans
are checked. The services have no pipeline neighbours, so the frames they push
go nowhere; the tracing wrapper still sees them.
"""

from types import SimpleNamespace

import pytest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:  # pragma: no cover
    HAS_OPENTELEMETRY = False

from google.genai.types import MediaModality

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import LLMFullResponseStartFrame, TTSStartedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.tracing.tracing_context import TracingContext

pytestmark = pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")

_exporter: "InMemorySpanExporter | None" = None


def _install_exporter():
    global _exporter
    if _exporter is not None:
        return _exporter
    _exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(_exporter)
    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        existing.add_span_processor(processor)
    else:
        provider = TracerProvider()
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    return _exporter


@pytest.fixture
def exporter():
    exp = _install_exporter()
    exp.clear()
    return exp


def _reply_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.name == "llm_response"]


async def _setup(service):
    await service.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
            enable_tracing=True,
            tracing_context=TracingContext(),
        )
    )


# ---------------------------------------------------------------------------
# Gemini Live
# ---------------------------------------------------------------------------


class _FakeServerContent:
    def __init__(self, *, turn_complete=False, interaction_status=None):
        self.model_turn = None
        self.input_transcription = None
        self.output_transcription = None
        self.grounding_metadata = None
        self.turn_complete = turn_complete
        self.interrupted = False
        self.interaction_status = interaction_status


def _usage_metadata(prompt, response, thoughts=0, audio_prompt=None):
    details = None
    if audio_prompt is not None:
        details = [SimpleNamespace(modality=MediaModality.AUDIO, token_count=audio_prompt)]
    return SimpleNamespace(
        prompt_token_count=prompt,
        tool_use_prompt_token_count=0,
        response_token_count=response,
        thoughts_token_count=thoughts,
        cached_content_token_count=None,
        prompt_tokens_details=details,
        response_tokens_details=None,
        cache_tokens_details=None,
    )


class _FakeServerMessage:
    def __init__(self, server_content=None, usage_metadata=None):
        self.server_content = server_content
        self.tool_call = None
        self.usage_metadata = usage_metadata
        self.session_resumption_update = None


async def _gemini_mid_reply(model="models/gemini-3.8-live"):
    service = GeminiLiveLLMService(
        api_key="test-key", settings=GeminiLiveLLMService.Settings(model=model)
    )
    await _setup(service)
    await service._set_bot_is_responding(True)
    await service.push_frame(TTSStartedFrame())
    await service.push_frame(LLMFullResponseStartFrame())
    return service


@pytest.mark.asyncio
async def test_gemini_turn_complete_produces_one_span_with_usage(exporter):
    service = await _gemini_mid_reply()

    await service._handle_server_message(
        _FakeServerMessage(
            _FakeServerContent(turn_complete=True),
            usage_metadata=_usage_metadata(prompt=300, response=40, audio_prompt=250),
        )
    )

    (span,) = _reply_spans(exporter)
    attrs = span.attributes
    assert attrs["gen_ai.provider.name"] == "gcp.gemini"
    assert attrs["gen_ai.request.model"] == "models/gemini-3.8-live"
    assert attrs["gen_ai.usage.input_tokens"] == 300
    assert attrs["gen_ai.usage.output_tokens"] == 40
    assert attrs["gen_ai.usage.audio.input_tokens"] == 250
    assert attrs["llm.interrupted"] is False
    assert [s.name for s in exporter.get_finished_spans() if s.name == "stt"] == []


@pytest.mark.asyncio
async def test_gemini_held_thinking_turn_sums_usage_of_every_chunk(exporter):
    """A thinking model reports usage per model call; the reply's span adds them up.

    This needs usage to be reported before the turn_complete frames: the
    second report lands on the reply only while it is still open.
    """
    service = await _gemini_mid_reply(model="models/gemini-3.8-live-extended-thinking")

    await service._handle_server_message(
        _FakeServerMessage(
            _FakeServerContent(turn_complete=True, interaction_status="IN_PROGRESS"),
            usage_metadata=_usage_metadata(prompt=3228, response=61, thoughts=104),
        )
    )
    assert _reply_spans(exporter) == []

    await service._handle_server_message(
        _FakeServerMessage(
            _FakeServerContent(turn_complete=True, interaction_status="IDLE"),
            usage_metadata=_usage_metadata(prompt=3368, response=20, thoughts=10),
        )
    )

    (span,) = _reply_spans(exporter)
    attrs = span.attributes
    assert attrs["gen_ai.usage.input_tokens"] == 3228 + 3368
    assert attrs["gen_ai.usage.output_tokens"] == (61 + 104) + (20 + 10)
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 114


# ---------------------------------------------------------------------------
# OpenAI Realtime and Azure Realtime
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


_USAGE = {
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


async def _openai_mid_reply(service):
    await _setup(service)
    service._current_assistant_response = SimpleNamespace(id="item_1")
    service._current_audio_response = None
    await service.push_frame(LLMFullResponseStartFrame())
    return service


@pytest.mark.asyncio
async def test_openai_response_done_produces_one_span_with_audio_breakdown(exporter):
    service = await _openai_mid_reply(
        OpenAIRealtimeLLMService(
            api_key="test-key", settings=OpenAIRealtimeLLMService.Settings(model="gpt-realtime")
        )
    )

    await service._handle_evt_response_done(_response_done_evt(_USAGE))

    (span,) = _reply_spans(exporter)
    attrs = span.attributes
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-realtime"
    assert attrs["gen_ai.usage.input_tokens"] == 60
    assert attrs["gen_ai.usage.output_tokens"] == 40
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 30
    assert attrs["gen_ai.usage.audio.input_tokens"] == 40
    assert attrs["gen_ai.usage.audio.output_tokens"] == 25
    assert attrs["gen_ai.usage.audio.cache_read.input_tokens"] == 20


@pytest.mark.asyncio
async def test_azure_realtime_reports_its_provider_name(exporter):
    service = await _openai_mid_reply(
        AzureRealtimeLLMService(
            api_key="test-key",
            base_url="wss://example.openai.azure.com/openai/realtime",
            settings=AzureRealtimeLLMService.Settings(model="gpt-realtime"),
        )
    )

    await service._handle_evt_response_done(_response_done_evt(_USAGE))

    (span,) = _reply_spans(exporter)
    assert span.attributes["gen_ai.provider.name"] == "azure.ai.openai"
