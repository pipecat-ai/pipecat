#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Run a response through a setup's processors and record what happened to it.

:func:`run_textflow` builds the processors a :class:`TextFlowSetup` describes,
streams the response through them with the stand-ins from :mod:`.mocks`, and
returns a :class:`TextFlowTrace`. The processors are the real ones; tracing
subclasses only bracket the methods that rewrite text so each rewrite can be
attributed to the segment it happened to.
"""

import asyncio
import contextlib
import dataclasses
import itertools
import json
import time
import warnings
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMContextAssistantTurnFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.models import (
    PROTOCOL_VERSION,
    BotOutputTransformResult,
    BotTranscriptionMessage,
)
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text
from pipecat.utils.text.base_text_aggregator import Aggregation, BaseTextAggregator, TextType
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.transforms.text_transform import TextTransform
from pipecat.workers.runner import WorkerRunner

from .capture import BotSummary, CapturedBot, callable_label, capture_bot, mirror_setup
from .mocks import SAMPLE_RATE, ProviderRequest, TextFlowTTSService, llm_response_frames
from .setup import (
    TextFlowSetup,
    TransformSpec,
    build_aggregator,
    build_transforms,
)

Stage = Literal["llm_text_processor", "tts", "rtvi"]


START_TIMEOUT_S = 30.0
"""How long a live TTS service may take to connect."""
RUN_TIMEOUT_S = 120.0
LIVE_RUN_TIMEOUT_S = 900.0
"""How long a run may take; a live run plays its audio at real pace."""


class TextFlowRunError(Exception):
    """The run could not be made; the message says why."""


class StageEvent(BaseModel):
    """Something a stage did to a segment's text.

    Parameters:
        stage: The stage that acted.
        kind: ``transform`` or ``filter`` rewrote the text, ``skip`` left it unspoken,
            ``trim`` or ``drop`` removed it before or after transforms, ``error``
            reports a transform that failed.
        name: The transform or filter, when one acted.
        before: The text before the action.
        after: The text after it, or None when it was removed.
        detail: Anything else worth showing, such as an error message.
        seq: The timeline position of the action, when it was placed in sequence.
    """

    stage: Stage
    kind: Literal["transform", "filter", "skip", "trim", "drop", "error"]
    name: str = ""
    before: str | None = None
    after: str | None = None
    detail: str = ""
    seq: int | None = None


class WordEvent(BaseModel):
    """One ``TTSTextFrame`` the service emitted for a segment, with its progress.

    Parameters:
        text: The word as the service reported it.
        llm_text: The span of the LLM's text the word stands for.
        accumulated: The segment text spoken so far, from the progress frame.
        remaining: The segment text still to speak.
    """

    text: str
    llm_text: str | None = None
    accumulated: str | None = None
    remaining: str | None = None


class TextFlowSegment(BaseModel):
    """One segment of the response and everything that happened to it.

    Parameters:
        segment_id: The ``AggregatedTextFrame``'s id, which words and client messages
            refer to. None for a segment dropped before it became a frame.
        produced_by: The stage whose aggregator cut the segment.
        text_type: The segment's text type.
        llm_text: What the conversation context records: the frame's ``raw_text``.
        segment_text: What downstream consumers see: the frame's ``text``.
        tts_text: What the provider was asked to speak, or None when nothing was.
        outcome: ``spoken``, ``skipped`` (a skipped text type), ``dropped`` (removed
            by a transform, a filter or whitespace trimming) or ``unspoken`` (the
            LLM marked it skip_tts).
        events: What each stage did to the segment, in order.
        words: The words the service reported while speaking it.
        context_text: What this segment contributed to the context, or None.
        rtvi_messages: The ``bot-output`` messages the client received for it.
    """

    segment_id: int | None
    produced_by: Literal["llm_text_processor", "tts"]
    text_type: str
    llm_text: str
    segment_text: str
    tts_text: str | None = None
    outcome: Literal["spoken", "skipped", "dropped", "unspoken"] = "spoken"
    events: list[StageEvent] = Field(default_factory=list)
    words: list[WordEvent] = Field(default_factory=list)
    context_text: str | None = None
    rtvi_messages: list[dict[str, Any]] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    """One thing that happened, in pipeline order.

    Parameters:
        seq: The entry's position in the run.
        t: Seconds since the run started.
        kind: ``frame`` for a frame moving between processors, ``rtvi`` for a message
            sent to the client, ``provider`` for a synthesis request, ``aggregate``
            for text entering or leaving a processor's aggregator, ``stage`` for
            something a processor did to a segment's text.
        source: Where it came from. The processor itself for ``aggregate`` and ``stage``.
        destination: Where it went. The processor itself for ``aggregate`` and ``stage``.
        label: The frame class, message type, request or action.
        summary: The text it carried.
        segment_id: The segment it belongs to, when known.
        data: What the entry carried, by field. A frame's text, type, raw text and
            progress; an aggregator's chunk and buffer; a stage action's before
            and after; a message's status; a request's tokens.
    """

    seq: int
    t: float = 0.0
    kind: Literal["frame", "rtvi", "provider", "aggregate", "stage"]
    source: str = ""
    destination: str = ""
    label: str
    summary: str = ""
    segment_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class TextFlowTrace(BaseModel):
    """What a run produced.

    Parameters:
        setup: The setup the run used.
        text: The response that was run.
        chunks: How the response streamed from the LLM.
        segments: Every segment, in the order it was produced.
        context_messages: The conversation context after the run.
        rtvi_messages: Every message the client received, in order.
        provider_requests: Every synthesis request, in order.
        timeline: Everything that happened, in order.
        errors: Errors pushed during the run.
        bot: What was found in the bot file, when the setup names one.
    """

    setup: TextFlowSetup
    bot: BotSummary | None = None
    text: str
    chunks: list[str]
    segments: list[TextFlowSegment]
    context_messages: list[dict[str, Any]]
    rtvi_messages: list[dict[str, Any]]
    provider_requests: list[dict[str, Any]]
    timeline: list[TimelineEntry]
    errors: list[str] = Field(default_factory=list)


_TIMELINE_FRAMES = (
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    LLMContextFrame,
    LLMContextAssistantTurnFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
)


def _summary(frame: Frame) -> str:
    if isinstance(frame, AggregatedTextProgressFrame):
        return f"{frame.accumulated_text!r} | {frame.remaining_text!r}"
    if isinstance(frame, TTSTextFrame):
        extra = (
            f" for {frame.raw_text!r}" if frame.raw_text and frame.raw_text != frame.text else ""
        )
        return f"{frame.text!r}{extra}"
    if isinstance(frame, AggregatedTextFrame):
        return f"{frame.text!r} ({frame.text_type})"
    if isinstance(frame, TextFrame):
        return repr(frame.text)
    if isinstance(frame, LLMContextAssistantTurnFrame):
        return repr(frame.text)
    if isinstance(frame, ErrorFrame):
        return frame.error
    return ""


def _jsonable(value: Any) -> Any:
    """A JSON-friendly stand-in for a frame field's value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return repr(value)


def _frame_fields(frame: Frame) -> dict[str, Any]:
    """Every field of a frame, for the step's detail view."""
    return {f.name: _jsonable(getattr(frame, f.name)) for f in dataclasses.fields(frame)}


def _frame_data(frame: Frame) -> dict[str, Any]:
    """What a frame carries, for the step-through view."""
    data: dict[str, Any] = {"frame": _frame_fields(frame)}
    if isinstance(frame, TextFrame):
        data["text"] = frame.text
        data["append_to_context"] = frame.append_to_context
        data["includes_inter_frame_spaces"] = frame.includes_inter_frame_spaces
        if frame.skip_tts:
            data["skip_tts"] = True
    if isinstance(frame, AggregatedTextFrame):
        data["text_type"] = str(frame.text_type)
        if frame.raw_text is not None and frame.raw_text != frame.text:
            data["raw_text"] = frame.raw_text
    if isinstance(frame, TTSTextFrame) and frame.pts is not None:
        data["pts_s"] = frame.pts / 1_000_000_000
    if isinstance(frame, AggregatedTextProgressFrame):
        data["text_type"] = str(frame.text_type)
        data["accumulated"] = frame.accumulated_text
        data["remaining"] = frame.remaining_text
    if isinstance(frame, TTSAudioRawFrame):
        data["duration_s"] = len(frame.audio) / (2 * frame.sample_rate * frame.num_channels)
    if isinstance(frame, LLMContextAssistantTurnFrame):
        data["text"] = frame.text
    if isinstance(frame, ErrorFrame):
        data["error"] = frame.error
    return data


def _segment_of(frame: Frame) -> int | None:
    if isinstance(frame, TTSTextFrame):
        return frame.segment_id
    if isinstance(frame, AggregatedTextProgressFrame):
        return frame.segment_id
    if isinstance(frame, AggregatedTextFrame):
        return frame.id
    return None


class _Tracer:
    """Everything the run records, shared by the tracing processors and observers."""

    def __init__(self):
        self.seq = itertools.count(1)
        self.t0 = time.monotonic()
        self.timeline: list[TimelineEntry] = []
        # Segment frames in the order they were first seen, by frame id.
        self.frames: dict[int, AggregatedTextFrame] = {}
        self.produced_by: dict[int, str] = {}
        self.events: dict[int, list[StageEvent]] = {}
        # Segments a transform removed before they became frames, with the
        # position they would have had.
        self.dropped: list[tuple[int, TextFlowSegment]] = []
        self.order: dict[int, int] = {}
        self.words: dict[int, list[TTSTextFrame]] = {}
        self.progress: dict[int, list[AggregatedTextProgressFrame]] = {}
        self.context_parts: dict[int | None, list[TextPartForConcatenation]] = {}
        self.all_parts: list[TextPartForConcatenation] = []
        self.requests: dict[int, ProviderRequest] = {}
        self.rtvi_messages: list[dict[str, Any]] = []
        self.errors: list[str] = []
        # A provider's audio comes in many frames; the timeline shows one entry
        # per request, grown as the frames arrive.
        self.audio_entry: TimelineEntry | None = None

    def see_frame(self, frame: AggregatedTextFrame, produced_by: str):
        if frame.id not in self.frames:
            self.frames[frame.id] = frame
            self.produced_by[frame.id] = produced_by
            self.order[frame.id] = next(self.seq)

    def drop(self, segment: TextFlowSegment):
        self.dropped.append((next(self.seq), segment))

    def add_event(self, frame_id: int, event: StageEvent):
        self.events.setdefault(frame_id, []).append(event)

    def record(self, entry: TimelineEntry):
        entry.t = round(time.monotonic() - self.t0, 4)
        self.timeline.append(entry)

    def record_stage(self, node: Stage, event: StageEvent) -> TimelineEntry:
        """Place a stage action in sequence and return its entry."""
        event.seq = next(self.seq)
        entry = TimelineEntry(
            seq=event.seq,
            kind="stage",
            source=node,
            destination=node,
            label=f"{event.kind} {event.name}".strip(),
            summary=(
                event.detail
                if event.kind == "error"
                else f"{event.before!r} → {event.after!r}"
                if event.after is not None
                else f"{event.before!r} → (removed)"
            ),
            data=event.model_dump(exclude={"seq"}),
        )
        self.record(entry)
        return entry


class _RecordingAggregator(BaseTextAggregator):
    """Runs a text aggregator and records what goes in and what comes out.

    Each chunk appended and each aggregation cut is an ``aggregate`` entry
    with the buffer as it stands after, so the buffer can be watched filling
    up and emptying.
    """

    def __init__(self, inner: BaseTextAggregator, tracer: _Tracer, node: str):
        super().__init__(aggregation_type=inner.aggregation_type, language=inner.language)
        self._inner = inner
        self._tracer = tracer
        self._node = node

    def _buffer(self) -> str:
        # The built-in aggregators keep their raw buffer in ``_text``; the
        # ``text`` property strips it, which hides the spaces between chunks.
        raw = getattr(self._inner, "_text", None)
        return raw if isinstance(raw, str) else self._inner.text.text

    def _record(self, action: str, buffer: str, chunk: str = "", aggregation=None):
        data: dict[str, Any] = {"action": action, "buffer": buffer}
        summary = f"+{chunk!r} → buffer {buffer!r}"
        if chunk:
            data["chunk"] = chunk
        if aggregation is not None:
            data["text"] = aggregation.text
            data["text_type"] = str(aggregation.type)
            summary = f"{aggregation.text!r} ({aggregation.type}), buffer {buffer!r}"
        self._tracer.record(
            TimelineEntry(
                seq=next(self._tracer.seq),
                kind="aggregate",
                source=self._node,
                destination=self._node,
                label=action,
                summary=summary,
                data=data,
            )
        )

    @property
    def language(self) -> str:
        """The inner aggregator's language."""
        return self._inner.language

    def set_language(self, language: str | None) -> None:
        """Select the inner aggregator's sentence tokenizer."""
        self._inner.set_language(language)

    @property
    def text(self) -> Aggregation:
        """The inner aggregator's buffered text."""
        return self._inner.text

    async def aggregate(self, text: str):
        """Append text to the inner aggregator, recording the buffer and each cut.

        The aggregators work through a chunk character by character, so the
        buffer at a cut holds only what was consumed so far. A ``settle`` entry
        shows the buffer once the whole chunk is in.
        """
        last = self._buffer() + text
        self._record("append", last, chunk=text)
        async for aggregation in self._inner.aggregate(text):
            last = self._buffer()
            self._record("cut", last, aggregation=aggregation)
            yield aggregation
        if self._buffer() != last:
            self._record("settle", self._buffer())

    async def flush(self) -> Aggregation | None:
        """Flush the inner aggregator, recording what it held."""
        result = await self._inner.flush()
        if result:
            self._record("flush", self._buffer(), aggregation=result)
        return result

    async def handle_interruption(self):
        """Pass the interruption to the inner aggregator."""
        await self._inner.handle_interruption()

    async def reset(self):
        """Reset the inner aggregator."""
        await self._inner.reset()


class _RecordingFilter(BaseTextFilter):
    """Runs a text filter and records what it did."""

    def __init__(self, inner: BaseTextFilter, record):
        self._inner = inner
        self._record = record

    async def update_settings(self, settings: Mapping[str, Any]):
        await self._inner.update_settings(settings)

    async def filter(self, text: str) -> str:
        result = await self._inner.filter(text)
        self._record(type(self._inner).__name__, text, result)
        return result

    async def handle_interruption(self):
        await self._inner.handle_interruption()

    async def reset_interruption(self):
        await self._inner.reset_interruption()


Transforms = list[tuple[str, TextTransform, str]]
"""``(text_type, transform, label)`` triples: the transforms a stage runs, with
the name each is reported under."""


def _spec_transforms(specs: list[TransformSpec]) -> Transforms:
    """The registered transforms the specs name."""
    return [
        (text_type, transform, spec.name)
        for spec, (text_type, transform) in zip(specs, build_transforms(specs))
    ]


def _bot_transforms(pairs: list[tuple[Any, Any]]) -> Transforms:
    """A bot's own ``(text_type, transform)`` pairs, labelled by callable name."""
    return [
        (str(text_type), transform, callable_label(transform)) for text_type, transform in pairs
    ]


def _recording_transforms(transforms: Transforms, record) -> list[tuple[str, TextTransform]]:
    """Wrap each transform to record what it did."""
    wrapped: list[tuple[str, TextTransform]] = []
    for text_type, transform, label in transforms:

        async def recording(text: str, tt: TextType | str, *, _t=transform, _n=label) -> str:
            result = await _t(text, tt)
            record(_n, text, result)
            return result

        wrapped.append((text_type, recording))
    return wrapped


class _TracingLLMTextProcessor(LLMTextProcessor):
    """An ``LLMTextProcessor`` that attributes each transform to the frame it produced."""

    def __init__(
        self,
        tracer: _Tracer,
        transforms: Transforms,
        text_aggregator: BaseTextAggregator,
        **kwargs,
    ):
        self._tracer = tracer
        self._pending: list[tuple[StageEvent, TimelineEntry]] = []
        super().__init__(
            text_transforms=_recording_transforms(transforms, self._step),
            text_aggregator=_RecordingAggregator(text_aggregator, tracer, "llm_text_processor"),
            **kwargs,
        )

    def _add(self, **fields):
        event = StageEvent(stage="llm_text_processor", **fields)
        self._pending.append((event, self._tracer.record_stage("llm_text_processor", event)))

    def _step(self, name: str, before: str, after: str):
        self._add(kind="transform", name=name, before=before, after=after)

    async def _aggregated_text_frame(self, aggregation: Aggregation) -> AggregatedTextFrame | None:
        self._pending = []
        frame = await super()._aggregated_text_frame(aggregation)
        pending, self._pending = self._pending, []
        events = [event for event, _ in pending]
        if frame is None:
            self._tracer.drop(
                TextFlowSegment(
                    segment_id=None,
                    produced_by="llm_text_processor",
                    text_type=str(aggregation.type),
                    llm_text=aggregation.text,
                    segment_text=aggregation.text,
                    outcome="dropped",
                    events=events,
                )
            )
            return None
        self._tracer.see_frame(frame, "llm_text_processor")
        for event, entry in pending:
            self._tracer.add_event(frame.id, event)
            entry.segment_id = frame.id
        return frame

    async def push_error(self, error_msg: str, *args, **kwargs):
        self._add(kind="error", detail=error_msg, after=None)
        await super().push_error(error_msg, *args, **kwargs)


class _TTSTracing:
    """Attaches to a ``TTSService`` and attributes what it does to segments.

    Every service, the stand-in included, goes through the base class's text
    hooks, so wrapping them on the instance traces a real service and the
    stand-in alike. The instance's transforms, filters and aggregator are
    replaced with recording ones.
    """

    def __init__(self, tts: TTSService, tracer: _Tracer, transforms: Transforms):
        self._tracer = tracer
        self._current: int | None = None
        self.requests: list[ProviderRequest] = []

        tts._text_transforms = _recording_transforms(transforms, self._step)
        tts._text_filters = [
            _RecordingFilter(f, self._filter_step) for f in list(tts._text_filters)
        ]
        tts._text_aggregator = _RecordingAggregator(tts._text_aggregator, tracer, "tts")  # type: ignore[assignment]

        handle = tts._handle_aggregated_text
        skipped = tts._push_skipped_text
        trim = tts._trim_text
        transform = tts._transform_text
        push_error = tts.push_error
        run_tts = tts.run_tts
        add_word_timestamps = tts.add_word_timestamps

        async def _handle_aggregated_text(src_frame: AggregatedTextFrame, context_id=None):
            tracer.see_frame(src_frame, "tts")
            self._current = src_frame.id
            try:
                await handle(src_frame, context_id)
            finally:
                self._current = None

        async def _push_skipped_text(frame: AggregatedTextFrame, context_id: str):
            self._event(kind="skip", before=frame.text, after=None, detail=str(frame.text_type))
            await skipped(frame, context_id)

        def _trim_text(text: str) -> str | None:
            trimmed = trim(text)
            if trimmed is None:
                self._event(kind="trim", before=text, after=None)
            return trimmed

        async def _transform_text(text: str, text_type: str) -> str | None:
            result = await transform(text, text_type)
            if result is None and self._current is not None:
                events = tracer.events.get(self._current, [])
                if not any(e.kind == "error" for e in events):
                    self._event(kind="drop", before=text, after=None)
            return result

        async def _push_error(error_msg: str, *args, **kwargs):
            self._event(kind="error", detail=error_msg, after=None)
            await push_error(error_msg, *args, **kwargs)

        async def _run_tts(text: str, context_id: str):
            request = ProviderRequest(context_id=context_id, text=text)
            self.requests.append(request)
            if self._current is not None and self._current not in tracer.requests:
                tracer.requests[self._current] = request
            tracer.audio_entry = None
            # The request's tokens list is shared with the entry, so the words
            # the provider reports later show up on it.
            tracer.record(
                TimelineEntry(
                    seq=next(tracer.seq),
                    kind="provider",
                    source="tts",
                    destination="provider",
                    label="run_tts",
                    summary=repr(text),
                    segment_id=self._current,
                    data={"text": text, "context_id": context_id, "tokens": request.tokens},
                )
            )
            async for frame in run_tts(text, context_id):
                yield frame

        async def _add_word_timestamps(word_times, context_id=None, *args, **kwargs):
            request = next((r for r in reversed(self.requests) if r.context_id == context_id), None)
            if request is not None:
                request.tokens.extend((word, t) for word, t in word_times)
            await add_word_timestamps(word_times, context_id, *args, **kwargs)

        # Set on the instance, so they shadow the class's methods for this one service.
        hooks = {
            "_handle_aggregated_text": _handle_aggregated_text,
            "_push_skipped_text": _push_skipped_text,
            "_trim_text": _trim_text,
            "_transform_text": _transform_text,
            "push_error": _push_error,
            "run_tts": _run_tts,
            "add_word_timestamps": _add_word_timestamps,
        }
        for name, hook in hooks.items():
            setattr(tts, name, hook)

    def _event(self, **fields):
        if self._current is not None:
            event = StageEvent(stage="tts", **fields)
            self._tracer.add_event(self._current, event)
            self._tracer.record_stage("tts", event).segment_id = self._current

    def _step(self, name: str, before: str, after: str):
        self._event(kind="transform", name=name, before=before, after=after)

    def _filter_step(self, name: str, before: str, after: str):
        self._event(kind="filter", name=name, before=before, after=after)


class _SilentOutputTransport(BaseOutputTransport):
    """An output transport that plays audio to nowhere.

    It still paces the audio it receives, releases each segment's text frames
    as their audio plays, and reports when the bot starts and stops speaking,
    which is what the RTVI observer and the context aggregator wait for.
    """

    def __init__(self):
        super().__init__(
            TransportParams(
                audio_out_enabled=True,
                audio_out_sample_rate=SAMPLE_RATE,
                audio_out_end_silence_secs=0,
            )
        )

    async def start(self, frame: StartFrame):
        """Start the transport: there is nothing to connect to, so it is ready at once."""
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame) -> bool:
        """Take the audio frame, taking as long as playing it would."""
        await asyncio.sleep(len(frame.audio) / (2 * frame.sample_rate * frame.num_channels))
        return True


class _Recorder(BaseObserver):
    """Records frames as they move between processors."""

    def __init__(self, tracer: _Tracer, names: dict[int, str], assistant: FrameProcessor):
        super().__init__()
        self._tracer = tracer
        self._names = names
        self._assistant = assistant
        self._seen: set[int] = set()

    def _label(self, processor: FrameProcessor) -> str:
        return self._names.get(id(processor), processor.__class__.__name__)

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        tracer = self._tracer
        entry: TimelineEntry | None = None
        # Only hops into one of the setup's processors are worth showing; the
        # frames reach them from the harness's own source, standing in for the LLM.
        # The transport broadcasts its speaking frames both ways and the
        # context aggregator's turn frame travels all the way upstream; one hop
        # of each is enough to show them. Audio matters when it reaches the
        # transport, not when the transport passes it on.
        source = self._names.get(id(data.source), "llm")
        destination = self._names.get(id(data.destination))
        once = (
            (
                isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame))
                and data.direction == FrameDirection.UPSTREAM
            )
            or (isinstance(frame, LLMContextAssistantTurnFrame) and source != "context")
            or (isinstance(frame, TTSAudioRawFrame) and destination != "transport")
        )
        if isinstance(frame, TTSAudioRawFrame) and tracer.audio_entry is not None and not once:
            grown = tracer.audio_entry
            grown.data["frames"] = grown.data.get("frames", 1) + 1
            grown.data["duration_s"] += _frame_data(frame)["duration_s"]
            grown.summary = f"{grown.data['frames']} frames, {grown.data['duration_s']:.2f}s"
        elif isinstance(frame, _TIMELINE_FRAMES) and destination is not None and not once:
            entry = TimelineEntry(
                seq=next(tracer.seq),
                kind="frame",
                source=source,
                destination=self._label(data.destination),
                label=frame.__class__.__name__,
                summary=_summary(frame),
                segment_id=_segment_of(frame),
                data=_frame_data(frame),
            )
            tracer.record(entry)
            if isinstance(frame, TTSAudioRawFrame):
                tracer.audio_entry = entry

        if isinstance(frame, TextFrame) and data.destination is self._assistant:
            if frame.append_to_context and frame.text:
                llm_text = (
                    frame.raw_text
                    if isinstance(frame, AggregatedTextFrame) and frame.raw_text
                    else frame.text
                )
                part = TextPartForConcatenation(llm_text, frame.includes_inter_frame_spaces)
                tracer.context_parts.setdefault(_segment_of(frame), []).append(part)
                tracer.all_parts.append(part)
                if entry is not None:
                    entry.data["context_part"] = llm_text
                    entry.data["context_so_far"] = concatenate_aggregated_text(tracer.all_parts)

        if frame.id in self._seen:
            return
        self._seen.add(frame.id)

        if isinstance(frame, LLMContextAssistantTurnFrame):
            tracer.all_parts = []
        elif isinstance(frame, TTSTextFrame):
            if frame.segment_id is not None:
                tracer.words.setdefault(frame.segment_id, []).append(frame)
        elif isinstance(frame, AggregatedTextFrame):
            tracer.see_frame(frame, self._label(data.source))
        elif isinstance(frame, AggregatedTextProgressFrame):
            tracer.progress.setdefault(frame.segment_id, []).append(frame)
        elif isinstance(frame, ErrorFrame):
            tracer.errors.append(frame.error)


class _ClientRTVIProcessor(RTVIProcessor):
    """An ``RTVIProcessor`` whose client speaks the current protocol.

    No client connects during a run, so the negotiated version stays at its
    default otherwise and the observer would withhold the messages a current
    client gets.
    """

    @property
    def client_version(self) -> list[int]:
        """The current protocol version, as a connected client would negotiate it."""
        return [int(part) for part in PROTOCOL_VERSION.split(".")]


class _RecordingRTVIObserver(RTVIObserver):
    """An ``RTVIObserver`` that keeps the messages it would send to the client."""

    def __init__(self, tracer: _Tracer, **kwargs):
        super().__init__(**kwargs)
        self._tracer = tracer

    async def send_rtvi_message(self, model: BaseModel, exclude_none: bool = True):
        # The deprecated sentence-level transcription duplicates the bot-llm-text
        # tokens and the bot-output segments, so it is left out of the trace.
        if isinstance(model, BotTranscriptionMessage):
            return
        message = model.model_dump(exclude_none=exclude_none)
        self._tracer.rtvi_messages.append(message)
        data = message.get("data") or {}
        summary = data.get("text", "")
        if "spoken_status" in data:
            summary = f"{data['spoken_status']}: {summary!r}"
        self._tracer.record(
            TimelineEntry(
                seq=next(self._tracer.seq),
                kind="rtvi",
                source="rtvi",
                destination="client",
                label=str(message.get("type", "")),
                summary=summary,
                segment_id=data.get("segment_id"),
                data=data,
            )
        )


def _rtvi_transforms(specs: list[TransformSpec]) -> list[tuple[str, Any]] | None:
    """Build the specs' transforms in the observer's progress-aware shape.

    A registered transform rewrites a whole segment. The observer also asks for
    the spoken and unspoken halves, which are split from the rewritten text in
    the proportion the original was split.
    """
    built: list[tuple[str, Any]] = []
    for text_type, transform in build_transforms(specs):

        async def progress_aware(
            text: str,
            tt: TextType | str,
            accumulated_text: str | None = None,
            remaining_text: str | None = None,
            *,
            _t=transform,
        ) -> BotOutputTransformResult:
            transformed = await _t(text, tt)
            if accumulated_text is None or remaining_text is None:
                return BotOutputTransformResult(text=transformed)
            ratio = len(accumulated_text) / max(len(text), 1)
            split = round(len(transformed) * ratio)
            return BotOutputTransformResult(
                text=transformed,
                accumulated_text=transformed[:split],
                remaining_text=transformed[split:],
            )

        built.append((text_type, progress_aware))
    return built or None


def _build_tts(
    setup: TextFlowSetup, tracer: _Tracer, bot_tts: TTSService | None = None
) -> tuple[TTSService, _TTSTracing]:
    """The TTS service to run, traced.

    The bot's own service when the setup says to run it live; otherwise the
    stand-in with the setup's knobs, and the bot's processing parts when given.
    """
    tts = setup.tts
    if tts.live:
        if bot_tts is None:
            raise TextFlowRunError("tts.live runs the bot's TTS service, so the setup needs a bot")
        return bot_tts, _TTSTracing(
            bot_tts, tracer, _bot_transforms(list(bot_tts._text_transforms))
        )
    kwargs: dict[str, Any] = dict(
        word_timestamps=tts.word_timestamps,
        delivery=tts.delivery,
        tokenization=tts.tokenization,
        settings=TTSSettings(model=None, voice=None, language=None),
    )
    aggregator: BaseTextAggregator | None = None
    if bot_tts is not None:
        aggregator = bot_tts._text_aggregator
        kwargs.update(
            text_aggregation_mode=bot_tts._text_aggregation_mode,
            skip_text_types=list(bot_tts._skip_text_types),
            text_filters=list(bot_tts._text_filters),
            append_trailing_space=bot_tts._append_trailing_space,
        )
        transforms = _bot_transforms(list(bot_tts._text_transforms))
    else:
        kwargs.update(
            text_aggregation_mode=TextAggregationMode(tts.aggregation_mode),
            skip_text_types=list(tts.skip_text_types),
            append_trailing_space=tts.append_trailing_space,
        )
        if tts.aggregator is not None:
            aggregator = build_aggregator(tts.aggregator)
        if tts.markdown_filter:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

                kwargs["text_filters"] = [MarkdownTextFilter()]
        transforms = _spec_transforms(tts.transforms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        stand_in = TextFlowTTSService(**kwargs)
    if aggregator is not None:
        stand_in._text_aggregator = aggregator  # type: ignore[assignment]
    return stand_in, _TTSTracing(stand_in, tracer, transforms)


async def _run(
    pipeline: Pipeline,
    frames: list[Frame],
    observers: list[BaseObserver],
    *,
    start_timeout_s: float,
    timeout_s: float,
):
    """Run the pipeline over the frames, as a test does, and wait for it to end."""
    worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, observers=observers)
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _on_pipeline_started(worker, frame):
        started.set()

    async def push_frames():
        await started.wait()
        for frame in frames:
            if isinstance(frame, SleepFrame):
                await asyncio.sleep(frame.sleep)
            else:
                await worker.queue_frame(frame)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    running = asyncio.ensure_future(runner.run())
    try:
        await asyncio.wait_for(started.wait(), start_timeout_s)
        await asyncio.wait_for(asyncio.gather(asyncio.shield(running), push_frames()), timeout_s)
    except TimeoutError as e:
        await runner.cancel(reason="textflow timeout")
        with contextlib.suppress(Exception):
            await running
        what = "start" if not started.is_set() else "finish"
        limit = start_timeout_s if not started.is_set() else timeout_s
        raise TextFlowRunError(f"the pipeline did not {what} within {limit:.0f}s") from e


async def run_textflow(setup: TextFlowSetup, text: str | None = None) -> TextFlowTrace:
    """Run ``text`` through the processors ``setup`` describes.

    Args:
        setup: The text path to run.
        text: The LLM's response. The setup's sample text when None.

    Returns:
        What happened to every segment of the response.
    """
    text = setup.sample_text if text is None else text
    if setup.bot is None:
        if setup.tts.live:
            raise TextFlowRunError("tts.live runs the bot's TTS service, so the setup needs a bot")
        return await _run_textflow(setup, text, None)
    # The bot is imported afresh each run, so edits to it show up.
    async with await capture_bot(setup.bot) as captured:
        setup = mirror_setup(captured, setup)
        if setup.tts.live is None:
            setup.tts.live = captured.tts is not None
        if setup.tts.live and captured.tts is None:
            raise TextFlowRunError(f"{setup.bot} has no TTSService to run live")
        trace = await _run_textflow(setup, text, captured)
        if (problem := await captured.close()) is not None:
            trace.errors.append(problem)
        return trace


async def _run_textflow(setup: TextFlowSetup, text: str, captured: CapturedBot | None):
    tracer = _Tracer()
    processors: list[FrameProcessor] = []
    names: dict[int, str] = {}
    processor = None
    if captured is not None:
        if captured.llm_text_processor is not None:
            bot_ltp = captured.llm_text_processor
            processor = _TracingLLMTextProcessor(
                tracer, _bot_transforms(list(bot_ltp._text_transforms)), bot_ltp._text_aggregator
            )
    elif setup.llm_text_processor is not None:
        spec = setup.llm_text_processor
        processor = _TracingLLMTextProcessor(
            tracer, _spec_transforms(spec.transforms), build_aggregator(spec.aggregator)
        )
    if processor is not None:
        names[id(processor)] = "llm_text_processor"
        processors.append(processor)

    tts, tts_tracing = _build_tts(setup, tracer, captured.tts if captured is not None else None)
    names[id(tts)] = "tts"
    processors.append(tts)

    transport = _SilentOutputTransport()
    names[id(transport)] = "transport"
    processors.append(transport)

    context = LLMContext()
    assistant = LLMAssistantAggregator(context)
    names[id(assistant)] = "context"
    processors.append(assistant)

    observers: list[BaseObserver] = [_Recorder(tracer, names, assistant)]
    if captured is not None and captured.rtvi_params is not None and setup.rtvi.enabled:
        rtvi_params = dataclasses.replace(captured.rtvi_params)
    elif captured is None and setup.rtvi.enabled:
        rtvi_params = RTVIObserverParams(
            bot_llm_enabled=setup.rtvi.bot_llm,
            bot_tts_enabled=setup.rtvi.bot_tts,
            skip_text_types=list(setup.rtvi.skip_text_types) or None,
            bot_output_transforms=_rtvi_transforms(setup.rtvi.transforms),
        )
    else:
        rtvi_params = None
    if rtvi_params is not None:
        # The processor only passes frames through here; it is not named so
        # its hops are left out and the frames it forwards come "from the LLM".
        rtvi = _ClientRTVIProcessor()
        processors.insert(0, rtvi)
        observers.append(_RecordingRTVIObserver(tracer, rtvi=rtvi, params=rtvi_params))

    chunks, frames = llm_response_frames(text, setup.chunking)
    pipeline = Pipeline(processors)
    # A live service connects at start and takes the time its provider takes.
    live = setup.tts.live
    await _run(
        pipeline,
        frames,
        observers,
        start_timeout_s=START_TIMEOUT_S if live else 5.0,
        timeout_s=LIVE_RUN_TIMEOUT_S if live else RUN_TIMEOUT_S,
    )

    return TextFlowTrace(
        setup=setup,
        bot=captured.summary if captured is not None else None,
        text=text,
        chunks=chunks,
        segments=_segments(tracer),
        context_messages=json.loads(json.dumps(context.messages, default=str)),
        rtvi_messages=tracer.rtvi_messages,
        provider_requests=[
            {"context_id": r.context_id, "text": r.text, "tokens": r.tokens}
            for r in tts_tracing.requests
        ],
        timeline=tracer.timeline,
        errors=tracer.errors,
    )


def _segments(tracer: _Tracer) -> list[TextFlowSegment]:
    segments: list[tuple[int, TextFlowSegment]] = list(tracer.dropped)
    for frame_id, frame in tracer.frames.items():
        events = tracer.events.get(frame_id, [])
        request = tracer.requests.get(frame_id)
        words = tracer.words.get(frame_id, [])

        if frame.skip_tts and not any(e.kind == "skip" for e in events):
            outcome = "unspoken"
        elif any(e.kind == "skip" for e in events):
            outcome = "skipped"
        elif any(e.kind in ("drop", "trim", "error") for e in events):
            outcome = "dropped"
        else:
            outcome = "spoken"

        word_events: list[WordEvent] = []
        progress_list = tracer.progress.get(frame_id, [])
        for i, word in enumerate(words):
            p = progress_list[i] if i < len(progress_list) else None
            word_events.append(
                WordEvent(
                    text=word.text,
                    llm_text=word.raw_text,
                    accumulated=p.accumulated_text if p else None,
                    remaining=p.remaining_text if p else None,
                )
            )
        parts = tracer.context_parts.get(frame_id)
        rtvi = [
            m for m in tracer.rtvi_messages if (m.get("data") or {}).get("segment_id") == frame_id
        ]
        segment = TextFlowSegment(
            segment_id=frame_id,
            produced_by=(
                "llm_text_processor"
                if tracer.produced_by.get(frame_id) == "llm_text_processor"
                else "tts"
            ),
            text_type=str(frame.text_type),
            llm_text=frame.raw_text if frame.raw_text is not None else frame.text,
            segment_text=frame.text,
            tts_text=request.text if request else None,
            outcome=outcome,
            events=events,
            words=word_events,
            context_text=concatenate_aggregated_text(parts) if parts else None,
            rtvi_messages=rtvi,
        )
        segments.append((tracer.order[frame_id], segment))
    return [segment for _, segment in sorted(segments, key=lambda s: s[0])]
