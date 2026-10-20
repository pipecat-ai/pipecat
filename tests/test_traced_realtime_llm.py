#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ``traced_realtime_llm``.

The decorator is driven through a stub class that defines only what it wraps
and reads: ``push_frame``, ``start_llm_usage_metrics``, ``cleanup``,
``create_task``, ``name``, ``_tracing_enabled`` and ``_tracing_context``.
Frames are pushed as a realtime service would push them, and the exported
spans are checked.
"""

import asyncio
import time

import pytest

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    HAS_OPENTELEMETRY = True
except ImportError:  # pragma: no cover
    HAS_OPENTELEMETRY = False

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    TextType,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.tracing import service_decorators
from pipecat.utils.tracing.service_decorators import traced_realtime_llm
from pipecat.utils.tracing.tracing_context import TracingContext

pytestmark = pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="opentelemetry not installed")

UP = FrameDirection.UPSTREAM


class _Stub:
    """A realtime service reduced to what the decorator touches."""

    name = "stub-realtime"

    def __init__(self):
        self._tracing_enabled = True
        self._tracing_context = TracingContext()
        self._model_name = "rt-model"
        self.pushed = []
        self.tasks = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))

    async def start_llm_usage_metrics(self, tokens):
        pass

    async def cleanup(self):
        pass

    def create_task(self, coroutine, name=None):
        task = asyncio.get_running_loop().create_task(coroutine, name=name)
        self.tasks.append(task)
        return task


Stub = traced_realtime_llm(_Stub)  # type: ignore[type-var]


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


@pytest.fixture
def svc():
    return Stub()


_GRACE_SECS = 0.05


@pytest.fixture(autouse=True)
def short_grace(monkeypatch):
    """Replies that answer speech export after the grace period; keep it short."""
    monkeypatch.setattr(service_decorators, "_REALTIME_REPLY_EXPORT_GRACE_SECS", _GRACE_SECS)


async def _settle():
    """Let the grace period run out so pending replies export."""
    await asyncio.sleep(_GRACE_SECS * 3)


def _spans(exporter, name="llm_response"):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _usage(prompt=10, completion=5, **kw):
    return LLMTokenUsage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion, **kw
    )


def _ttfb(processor, value):
    return MetricsFrame(data=[TTFBMetricsData(processor=processor, value=value)])


def _audio():
    return TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)


def _turn_context(ctx: TracingContext, name: str):
    """Open a turn span, make it current on ``ctx``, and return it."""
    span = trace.get_tracer("pipecat").start_span(name)
    ctx.set_turn_context(span.get_span_context())
    return span


async def _normal_reply(svc, *, speech_end=None, text="Hi there.", usage=True):
    """User stops speaking, the model replies, End, then usage."""
    if speech_end is not None:
        await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=speech_end + 0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame(text))
    await svc.push_frame(_audio())
    await svc.push_frame(LLMFullResponseEndFrame())
    if usage:
        await svc.start_llm_usage_metrics(_usage())
    await _settle()


# ---------------------------------------------------------------------------
# Reply spans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_reply_starts_at_speech_end_with_attributes(exporter, svc):
    speech_end = time.time() - 1.0
    await svc.push_frame(TranscriptionFrame("What time is it?", "", "", finalized=False), UP)
    before_end = time.time_ns()
    await _normal_reply(svc, speech_end=speech_end)

    (span,) = _spans(exporter)
    assert span.start_time == int(speech_end * 1e9)
    assert span.end_time >= before_end
    attrs = span.attributes
    assert attrs["gen_ai.provider.name"] == "_stub"
    assert attrs["gen_ai.request.model"] == "rt-model"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.output.type"] == "speech"
    assert attrs["input"] == "What time is it?"
    assert attrs["input_modality"] == "AUDIO"
    assert attrs["output"] == "Hi there."
    assert attrs["output_modality"] == "AUDIO"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["llm.interrupted"] is False


@pytest.mark.asyncio
async def test_usage_and_transcript_after_end_are_waited_for(exporter, svc):
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Sure."))
    end_before = time.time_ns()
    await svc.push_frame(LLMFullResponseEndFrame())
    end_after = time.time_ns()
    assert _spans(exporter) == []

    await svc.start_llm_usage_metrics(_usage())
    assert _spans(exporter) == [], "still waiting for the transcript"

    await svc.push_frame(TranscriptionFrame("Hello.", "", "", finalized=False), UP)
    assert _spans(exporter) == [], "the transcript may come in several pieces"

    await _settle()
    (span,) = _spans(exporter)
    assert span.attributes["input"] == "Hello."
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    # The span ends when the End frame passed, not when the late data arrived.
    assert end_before <= span.end_time <= end_after


@pytest.mark.asyncio
async def test_transcript_sentences_after_usage_all_belong_to_the_reply(exporter, svc):
    # Gemini Live pushes one TranscriptionFrame per sentence, often after the
    # reply has ended and its usage has arrived.
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Sunny, 21 degrees."))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    await svc.push_frame(
        TranscriptionFrame("What's the weather in Rome?", "", "", finalized=False), UP
    )
    await svc.push_frame(TranscriptionFrame(" Use Celsius.", "", "", finalized=False), UP)
    assert _spans(exporter) == []

    # The next reply closes the pending one without waiting for the grace period.
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(TranscriptionFrame("Thanks.", "", "", finalized=False), UP)
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    await _settle()

    first, second = _spans(exporter)
    assert first.attributes["input"] == "What's the weather in Rome? Use Celsius."
    assert second.attributes["input"] == "Thanks."


@pytest.mark.asyncio
async def test_reply_without_speech_exports_as_soon_as_usage_arrives(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Hello! How can I help?"))
    await svc.push_frame(LLMFullResponseEndFrame())
    assert _spans(exporter) == []
    await svc.start_llm_usage_metrics(_usage())
    (span,) = _spans(exporter)
    assert "input" not in span.attributes


@pytest.mark.asyncio
async def test_grace_period_exports_an_incomplete_reply(exporter, svc):
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Hi there."))
    await svc.push_frame(LLMFullResponseEndFrame())
    assert _spans(exporter) == []

    await _settle()
    (span,) = _spans(exporter)
    assert "gen_ai.usage.input_tokens" not in span.attributes
    assert span.attributes["output"] == "Hi there."


@pytest.mark.asyncio
async def test_interruption_with_late_usage_and_duplicate_end(exporter, svc):
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(TranscriptionFrame("Tell me a story", "", "", finalized=False), UP)
    await svc.push_frame(LLMTextFrame("Let me tell you a long"))
    onset = time.time()
    await svc.push_frame(VADUserStartedSpeakingFrame(start_secs=0.1, timestamp=onset + 0.1))
    await svc.push_frame(InterruptionFrame())
    # The cancelled response finishes: End again, then its usage.
    await svc.push_frame(LLMFullResponseEndFrame())
    assert _spans(exporter) == []
    await svc.start_llm_usage_metrics(_usage(prompt=3, completion=1))

    await _settle()
    (span,) = _spans(exporter)
    assert span.attributes["input"] == "Tell me a story"
    assert span.attributes["llm.interrupted"] is True
    assert span.attributes["gen_ai.usage.input_tokens"] == 3
    assert span.end_time == int(onset * 1e9)


@pytest.mark.asyncio
async def test_barge_in_transcript_goes_to_the_next_reply(exporter, svc):
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(TranscriptionFrame("First question", "", "", finalized=False), UP)
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Answer one"))
    await svc.push_frame(VADUserStartedSpeakingFrame(start_secs=0.1))
    # The user is speaking again, so this transcript is theirs, not the first reply's.
    await svc.push_frame(TranscriptionFrame("Second question", "", "", finalized=False), UP)
    await svc.push_frame(InterruptionFrame())
    await svc.start_llm_usage_metrics(_usage())
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Answer two"))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    await _settle()
    first, second = _spans(exporter)
    assert first.attributes["input"] == "First question"
    assert first.attributes["output"] == "Answer one"
    assert second.attributes["input"] == "Second question"
    assert second.attributes["output"] == "Answer two"


@pytest.mark.asyncio
async def test_duplicate_starts_make_one_span_and_a_bare_end_makes_none(exporter, svc):
    await svc.push_frame(LLMFullResponseEndFrame())
    assert _spans(exporter) == []

    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("One reply"))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    await svc.push_frame(LLMFullResponseEndFrame())

    (span,) = _spans(exporter)
    assert span.attributes["output"] == "One reply"


@pytest.mark.asyncio
async def test_upstream_copies_of_broadcast_frames_are_ignored(exporter, svc):
    long_ago = time.time() - 60
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=long_ago), UP)
    before = time.time_ns()
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert span.start_time >= before


@pytest.mark.asyncio
async def test_user_stopped_speaking_is_the_fallback_start_and_a_late_one_is_ignored(exporter, svc):
    before = time.time_ns()
    await svc.push_frame(UserStoppedSpeakingFrame())
    after = time.time_ns()
    await svc.push_frame(TranscriptionFrame("Hi", "", "", finalized=False), UP)
    await asyncio.sleep(0.01)
    await svc.push_frame(LLMFullResponseStartFrame())
    # A turn strategy ends the user turn after the service already started replying.
    await svc.push_frame(UserStoppedSpeakingFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    await _settle()
    (span,) = _spans(exporter)
    assert before <= span.start_time <= after
    assert span.attributes["input"] == "Hi"

    # The late stop must not become the next reply's start either.
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    await _settle()
    _, second = _spans(exporter)
    assert second.start_time > after


@pytest.mark.asyncio
async def test_vad_speech_end_wins_over_user_stopped_speaking(exporter, svc):
    speech_end = time.time() - 2.0
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=speech_end + 0.2))
    await svc.push_frame(UserStoppedSpeakingFrame())
    await svc.push_frame(TranscriptionFrame("Hi", "", "", finalized=False), UP)
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    await _settle()
    (span,) = _spans(exporter)
    assert span.start_time == int(speech_end * 1e9)


@pytest.mark.asyncio
async def test_reply_with_no_speech_before_it_has_no_input(exporter, svc):
    before = time.time_ns()
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Hello! I'm your assistant."))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert span.start_time >= before
    assert "input" not in span.attributes
    assert "input_modality" not in span.attributes


@pytest.mark.asyncio
async def test_reply_is_parented_to_the_turn_of_its_speech_end(exporter, svc):
    turn_a = _turn_context(svc._tracing_context, "turn")
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(TranscriptionFrame("Hi", "", "", finalized=False), UP)
    turn_a.end()
    turn_b = _turn_context(svc._tracing_context, "turn")
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    turn_b.end()

    await _settle()
    (span,) = _spans(exporter)
    assert span.parent.span_id == turn_a.get_span_context().span_id


@pytest.mark.asyncio
async def test_reply_without_speech_end_is_parented_to_the_current_turn(exporter, svc):
    turn = _turn_context(svc._tracing_context, "turn")
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    turn.end()

    (span,) = _spans(exporter)
    assert span.parent.span_id == turn.get_span_context().span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("closer", ["end_frame", "cancel_frame", "cleanup"])
async def test_pipeline_end_closes_an_open_reply(exporter, svc, closer):
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Bye"))
    if closer == "end_frame":
        await svc.push_frame(EndFrame())
    elif closer == "cancel_frame":
        await svc.push_frame(CancelFrame())
    else:
        await svc.cleanup()

    (span,) = _spans(exporter)
    assert span.attributes["output"] == "Bye"
    assert span.attributes["llm.interrupted"] is True


@pytest.mark.asyncio
async def test_pipeline_end_exports_a_pending_reply(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    assert _spans(exporter) == []
    await svc.push_frame(EndFrame())
    (span,) = _spans(exporter)
    assert span.attributes["llm.interrupted"] is False


@pytest.mark.asyncio
async def test_tts_text_is_the_fallback_output_including_text_before_start(exporter, svc):
    tts = TTSTextFrame("Weather", text_type=TextType.SENTENCE)
    await svc.push_frame(tts)
    await svc.push_frame(LLMFullResponseStartFrame())
    tts2 = TTSTextFrame("today", text_type=TextType.SENTENCE)
    await svc.push_frame(tts2)
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert span.attributes["output"] == "Weather today"
    assert span.attributes["output_modality"] == "TEXT"
    assert span.attributes["gen_ai.output.type"] == "text"


@pytest.mark.asyncio
async def test_llm_text_is_preferred_over_tts_text(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Hello "))
    await svc.push_frame(TTSTextFrame("Hello", text_type=TextType.SENTENCE))
    await svc.push_frame(LLMTextFrame("world."))
    await svc.push_frame(TTSTextFrame("world.", text_type=TextType.SENTENCE))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert span.attributes["output"] == "Hello world."


@pytest.mark.asyncio
async def test_ttfb_before_start_belongs_to_the_next_reply(exporter, svc):
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(_ttfb("stub-realtime", 0.42))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(_ttfb("stub-realtime", 0.99))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    await svc.push_frame(TranscriptionFrame("Hi", "", "", finalized=False), UP)

    await _settle()
    (span,) = _spans(exporter)
    assert span.attributes["metrics.ttfb"] == 0.42
    assert span.attributes["gen_ai.response.time_to_first_chunk"] == 0.42


@pytest.mark.asyncio
async def test_metrics_from_other_processors_are_ignored(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(_ttfb("some-stt", 0.31))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert "metrics.ttfb" not in span.attributes


@pytest.mark.asyncio
async def test_partial_usage_reports_are_summed(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.start_llm_usage_metrics(_usage(prompt=100, completion=10, reasoning_tokens=4))
    await svc.start_llm_usage_metrics(_usage(prompt=20, completion=5, input_audio_tokens=7))
    await svc.push_frame(LLMFullResponseEndFrame())

    (span,) = _spans(exporter)
    attrs = span.attributes
    assert attrs["gen_ai.usage.input_tokens"] == 120
    assert attrs["gen_ai.usage.output_tokens"] == 15
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 4
    assert attrs["gen_ai.usage.audio.input_tokens"] == 7


@pytest.mark.asyncio
async def test_tracing_disabled_records_nothing(exporter, svc):
    svc._tracing_enabled = False
    await _normal_reply(svc, speech_end=time.time())
    await svc.push_frame(
        FunctionCallInProgressFrame(function_name="f", tool_call_id="c1", arguments={})
    )
    assert exporter.get_finished_spans() == ()


# ---------------------------------------------------------------------------
# Interrupted reply end times
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupted_reply_ends_at_the_vad_onset(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await asyncio.sleep(0.01)
    onset = time.time()
    await svc.push_frame(VADUserStartedSpeakingFrame(start_secs=0.15, timestamp=onset + 0.15))
    await asyncio.sleep(0.01)
    await svc.push_frame(InterruptionFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert span.end_time == int(onset * 1e9)


@pytest.mark.asyncio
async def test_interrupted_reply_ends_at_the_services_proposed_start(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await asyncio.sleep(0.01)
    before = time.time_ns()
    await svc.push_frame(ProposedUserStartedSpeakingFrame())
    after = time.time_ns()
    await asyncio.sleep(0.01)
    await svc.push_frame(InterruptionFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert before <= span.end_time <= after


@pytest.mark.asyncio
async def test_end_pushed_by_the_interruption_handler_still_counts_as_interrupted(exporter, svc):
    """OpenAI Realtime pushes End from _handle_interruption, before the InterruptionFrame."""
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Once upon a time"))
    await asyncio.sleep(0.01)
    onset = time.time()
    await svc.push_frame(ProposedUserStartedSpeakingFrame())
    await asyncio.sleep(0.01)
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.push_frame(InterruptionFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert span.attributes["llm.interrupted"] is True
    assert abs(span.end_time - int(onset * 1e9)) < 5_000_000


@pytest.mark.asyncio
async def test_interruption_after_a_normally_ended_reply_leaves_it_alone(exporter, svc):
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMFullResponseEndFrame())
    end_ns = time.time_ns()
    await asyncio.sleep(0.01)
    # The user speaks later, during silence; the aggregator still broadcasts an interruption.
    await svc.push_frame(VADUserStartedSpeakingFrame(start_secs=0.0))
    await svc.push_frame(InterruptionFrame())
    await svc.start_llm_usage_metrics(_usage())

    (span,) = _spans(exporter)
    assert span.attributes["llm.interrupted"] is False
    assert span.end_time <= end_ns


@pytest.mark.asyncio
async def test_interrupted_reply_never_ends_before_its_start(exporter, svc):
    # The user was already speaking when the reply started (e.g. a tool
    # follow-up colliding with the next utterance).
    await svc.push_frame(VADUserStartedSpeakingFrame(start_secs=0.1))
    await asyncio.sleep(0.01)
    await svc.push_frame(LLMFullResponseStartFrame())
    # The utterance under way is not the follow-up's input, even though it
    # arrives while the follow-up is open.
    await svc.push_frame(TranscriptionFrame("Actually, wait.", "", "", finalized=False), UP)
    await svc.push_frame(InterruptionFrame())
    await svc.start_llm_usage_metrics(_usage())
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Sure, go ahead."))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage())
    await _settle()

    follow_up, answer = _spans(exporter)
    assert follow_up.end_time == follow_up.start_time
    assert "input" not in follow_up.attributes
    assert answer.attributes["input"] == "Actually, wait."


# ---------------------------------------------------------------------------
# Tool call spans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_span_covers_in_progress_to_final_result(exporter, svc):
    turn = _turn_context(svc._tracing_context, "turn")
    await svc.push_frame(
        FunctionCallInProgressFrame(
            function_name="get_weather", tool_call_id="c1", arguments={"city": "Rome"}
        )
    )
    assert _spans(exporter, "llm_tool_call") == []
    await svc.push_frame(
        FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="c1",
            arguments={"city": "Rome"},
            result={"temp_c": 21},
        )
    )
    turn.end()

    (span,) = _spans(exporter, "llm_tool_call")
    attrs = span.attributes
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "get_weather"
    assert attrs["gen_ai.tool.call.id"] == "c1"
    assert attrs["gen_ai.tool.call.arguments"] == '{"city": "Rome"}'
    assert attrs["gen_ai.tool.call.result"] == '{"temp_c": 21}'
    assert span.parent.span_id == turn.get_span_context().span_id
    assert span.status.status_code == trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_tool_only_response_without_start_frame_opens_a_reply(exporter, svc):
    # OpenAI Realtime pushes no LLMFullResponseStartFrame for a response that
    # only calls a tool: the function call frames, then usage and End from
    # response.done, then a Start for the follow-up response.
    speech_end = time.time() - 0.5
    await svc.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=speech_end + 0.2))
    await svc.push_frame(TranscriptionFrame("Weather in Rome?", "", "", finalized=False), UP)
    await svc.push_frame(
        FunctionCallInProgressFrame(
            function_name="get_weather", tool_call_id="c1", arguments={"city": "Rome"}
        )
    )
    await svc.start_llm_usage_metrics(_usage(prompt=7, completion=2))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.push_frame(
        FunctionCallResultFrame(
            function_name="get_weather",
            tool_call_id="c1",
            arguments={"city": "Rome"},
            result={"temp_c": 21},
        )
    )
    await svc.push_frame(LLMFullResponseStartFrame())
    await svc.push_frame(LLMTextFrame("Sunny, 21 degrees."))
    await svc.push_frame(LLMFullResponseEndFrame())
    await svc.start_llm_usage_metrics(_usage(prompt=9, completion=4))

    tool_reply, follow_up = _spans(exporter)
    assert tool_reply.start_time == int(speech_end * 1e9)
    assert tool_reply.attributes["input"] == "Weather in Rome?"
    assert tool_reply.attributes["gen_ai.usage.input_tokens"] == 7
    assert "output" not in tool_reply.attributes
    assert follow_up.attributes["output"] == "Sunny, 21 degrees."
    assert follow_up.attributes["gen_ai.usage.input_tokens"] == 9
    (tool,) = _spans(exporter, "llm_tool_call")
    assert tool.attributes["gen_ai.tool.name"] == "get_weather"


@pytest.mark.asyncio
async def test_tool_call_error_sets_error_status(exporter, svc):
    await svc.push_frame(
        FunctionCallInProgressFrame(function_name="boom", tool_call_id="c2", arguments={})
    )
    await svc.push_frame(
        FunctionCallResultFrame(
            function_name="boom",
            tool_call_id="c2",
            arguments={},
            result="The tool failed.",
            error="ValueError: nope",
        )
    )

    (span,) = _spans(exporter, "llm_tool_call")
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.attributes["error.type"] == "tool_error"


@pytest.mark.asyncio
async def test_intermediate_result_keeps_the_tool_span_open(exporter, svc):
    await svc.push_frame(
        FunctionCallInProgressFrame(function_name="slow", tool_call_id="c3", arguments={})
    )
    await svc.push_frame(
        FunctionCallResultFrame(
            function_name="slow",
            tool_call_id="c3",
            arguments={},
            result="working...",
            properties=FunctionCallResultProperties(is_final=False),
        )
    )
    assert _spans(exporter, "llm_tool_call") == []
    await svc.push_frame(
        FunctionCallResultFrame(
            function_name="slow",
            tool_call_id="c3",
            arguments={},
            result="done",
            properties=FunctionCallResultProperties(is_final=True),
        )
    )
    (span,) = _spans(exporter, "llm_tool_call")
    assert span.attributes["gen_ai.tool.call.result"] == '"done"'


@pytest.mark.asyncio
async def test_cancelled_tool_call_is_marked(exporter, svc):
    await svc.push_frame(
        FunctionCallInProgressFrame(function_name="slow", tool_call_id="c4", arguments={})
    )
    await svc.push_frame(FunctionCallCancelFrame(function_name="slow", tool_call_id="c4"))

    (span,) = _spans(exporter, "llm_tool_call")
    assert span.attributes["tool.cancelled"] is True


@pytest.mark.asyncio
async def test_tool_attributes_are_truncated(exporter, svc):
    big = {"blob": "x" * 5000}
    await svc.push_frame(
        FunctionCallInProgressFrame(function_name="big", tool_call_id="c5", arguments=big)
    )
    await svc.push_frame(
        FunctionCallResultFrame(function_name="big", tool_call_id="c5", arguments=big, result=big)
    )

    (span,) = _spans(exporter, "llm_tool_call")
    assert len(span.attributes["gen_ai.tool.call.arguments"]) == 2003
    assert span.attributes["gen_ai.tool.call.arguments"].endswith("...")
    assert len(span.attributes["gen_ai.tool.call.result"]) == 2003


@pytest.mark.asyncio
async def test_pipeline_end_closes_open_tool_calls(exporter, svc):
    await svc.push_frame(
        FunctionCallInProgressFrame(function_name="slow", tool_call_id="c6", arguments={})
    )
    await svc.push_frame(EndFrame())

    (span,) = _spans(exporter, "llm_tool_call")
    assert span.attributes["tool.cancelled"] is True


@pytest.mark.asyncio
async def test_upstream_tool_frames_are_ignored(exporter, svc):
    await svc.push_frame(
        FunctionCallInProgressFrame(function_name="f", tool_call_id="c7", arguments={}), UP
    )
    await svc.push_frame(EndFrame())
    assert _spans(exporter, "llm_tool_call") == []


# ---------------------------------------------------------------------------
# Decorator mechanics
# ---------------------------------------------------------------------------


def test_decorating_a_subclass_wraps_once():
    class Child(Stub):
        pass

    assert traced_realtime_llm(Child) is Child  # type: ignore[type-var]
    assert Child.push_frame is Stub.push_frame
