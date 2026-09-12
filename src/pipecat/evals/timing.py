#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""When the bot's reply to a scripted turn happened, measured on the harness's pipeline.

:class:`EvalTimingObserver` watches the frames pushed in the eval client's
pipeline and fills one :class:`~pipecat.evals.results.EvalTurnTiming` per
turn. The driver tells it when a turn's input went out; everything else it
reads off the pipeline:

===========================  =====================================================
measure                      frame, and where it comes from
===========================  =====================================================
anchor (spoken turn)         the output transport's ``BotStoppedSpeakingFrame``:
                             the last chunk of the user's utterance went out
``input_duration_ms``        the user side's ``TTSAudioRawFrame`` bytes for the
                             utterance
``llm_started_ms``           the input transport's ``LLMFullResponseStartFrame``
``first_token_ms``           its first ``LLMTextFrame`` after that
``llm_response_ms``          its ``LLMFullResponseEndFrame``
``function_call_ms``         its ``FunctionCallInProgressFrame``
``bot_started_speaking_ms``  its ``BotStartedSpeakingFrame`` (the bot's report)
``bot_stopped_speaking_ms``  its ``BotStoppedSpeakingFrame``, after a start
``bot_speech_onset_ms``      the bot-audio aggregator's ``UserStartedSpeakingFrame``:
                             the harness's own VAD heard the bot
``bot_metrics``              the input transport's ``MetricsFrame``
===========================  =====================================================

The harness's pipeline carries both sides' frames: the bot's come in through
the input transport, the user's are made by the user TTS and the output
transport. A frame is told apart by the processor that first pushed it.
"""

import time
from collections.abc import Callable

from pipecat.evals.results import EvalTurnTiming
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallInProgressFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport

# The measures a turn's timing takes, each the first of its kind after the anchor.
_MEASURES = (
    "llm_started",
    "first_token",
    "llm_response",
    "function_call",
    "bot_started_speaking",
    "bot_speech_onset",
    "bot_stopped_speaking",
)


def _ms(seconds: float) -> int:
    """Seconds to whole milliseconds."""
    return int(round(seconds * 1000))


def _bot_metrics(data: list[MetricsData]) -> list[dict]:
    """One ``MetricsFrame`` as the turn keeps it: one entry per processor.

    TTFB and processing time are joined on their processor's name; token
    usage is its own entry, since the bot reports it without one.
    """
    entries: dict[str, dict] = {}

    def entry(processor: str | None) -> dict:
        blank = {"processor": processor, "ttfb_ms": None, "processing_ms": None, "tokens": None}
        if not processor:
            return blank
        return entries.setdefault(processor, blank)

    usages = []
    for metrics in data:
        if isinstance(metrics, TTFBMetricsData):
            entry(metrics.processor)["ttfb_ms"] = _ms(metrics.value)
        elif isinstance(metrics, ProcessingMetricsData):
            entry(metrics.processor)["processing_ms"] = _ms(metrics.value)
        elif isinstance(metrics, LLMUsageMetricsData):
            usage = entry(None)
            usage["processor"] = metrics.processor or None
            usage["tokens"] = metrics.value.model_dump(exclude_none=True)
            usages.append(usage)
    return list(entries.values()) + usages


class _Turn:
    """One turn's timing, kept as raw stamps so its anchor can move.

    A spoken turn is anchored at the send until its utterance has gone out,
    then re-anchored at the end of it: the measures are recomputed and the
    stamps from before the new anchor dropped. Each measure takes its first
    stamp only.
    """

    def __init__(self, anchor: float):
        self.anchor = anchor
        self.timing = EvalTurnTiming()
        self._stamps: dict[str, float] = {}
        self._metrics: list[tuple[float, list[dict]]] = []

    def stamp(self, measure: str, at: float) -> None:
        """Record when ``measure`` happened, unless it already has, or it predates the anchor."""
        if measure in self._stamps or at < self.anchor:
            return
        self._stamps[measure] = at
        self._apply()

    def has(self, measure: str) -> bool:
        """Whether ``measure`` has been stamped."""
        return measure in self._stamps

    def add_metrics(self, at: float, metrics: list[dict]) -> None:
        """Keep a metrics report the bot sent at ``at``."""
        if at < self.anchor:
            return
        self._metrics.append((at, metrics))
        self._apply()

    def rebase(self, anchor: float, input_duration_ms: int) -> None:
        """Move the anchor to the end of the user's speech, forgetting what came before it."""
        self.anchor = anchor
        self._stamps = {k: v for k, v in self._stamps.items() if v >= anchor}
        self._metrics = [(at, m) for at, m in self._metrics if at >= anchor]
        self.timing.input_duration_ms = input_duration_ms
        self._apply()

    def _apply(self) -> None:
        for measure in _MEASURES:
            at = self._stamps.get(measure)
            setattr(self.timing, f"{measure}_ms", None if at is None else _ms(at - self.anchor))
        self.timing.bot_metrics = [m for _, metrics in self._metrics for m in metrics]


class EvalTimingObserver(BaseObserver):
    """Measures each scripted turn's latencies from the harness pipeline's frames.

    The driver calls :meth:`begin_turn` once a turn's input has been sent,
    which anchors the turn there; a spoken turn is re-anchored when the
    output transport reports the utterance has gone out. The
    :class:`~pipecat.evals.results.EvalTurnTiming` a turn gets keeps filling
    in until the next :meth:`begin_turn`, so a record that holds it sees the
    events that land after the turn's expectations resolved.

    The bot's LLM text and response end count only after its ``llm_started``
    in the same turn, and its stopped-speaking report only after a start: a
    response that began before the input is the previous turn's, however
    much of it lands after.
    """

    def __init__(self, *, time_source: Callable[[], float] = time.monotonic, **kwargs):
        """Initialize the observer.

        Args:
            time_source: Reads the current time in seconds. Supplying one lets
                a test place moments without waiting.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self._now = time_source
        self._turn = _Turn(self._now())
        # Frames already read: every processor that passes a frame along
        # pushes it again, and only its first push says where it came from.
        self._seen: set[int] = set()
        # Seconds of user audio on its way out, summed over the utterance's
        # frames until the output reports it sent.
        self._utterance_secs = 0.0

    @property
    def timing(self) -> EvalTurnTiming:
        """The current turn's timing, filled in as its frames arrive."""
        return self._turn.timing

    def begin_turn(self) -> EvalTurnTiming:
        """Start a turn: its input was just sent, and its reply is what follows.

        Returns:
            The turn's timing, updated in place from here to the next turn.
        """
        self._turn = _Turn(self._now())
        return self._turn.timing

    async def on_push_frame(self, data: FramePushed):
        """Read the frame the first time it is pushed, if it times something.

        Args:
            data: Frame push event containing the frame and its source.
        """
        frame = data.frame
        # A broadcast arrives as two frames with two IDs; read the downstream one.
        if frame.broadcast_sibling_id is not None and data.direction != FrameDirection.DOWNSTREAM:
            return
        if frame.id in self._seen:
            return
        if self._read(frame, data.source):
            self._seen.add(frame.id)

    def _read(self, frame: Frame, source) -> bool:
        """Time the frame; whether it was one to time."""
        at = self._now()
        if isinstance(source, BaseInputTransport):
            return self._read_bot(frame, at)
        if isinstance(source, BaseOutputTransport):
            return self._read_output(frame, at)
        if isinstance(frame, TTSAudioRawFrame):
            # The user's utterance on its way to the output, one frame at a time.
            self._utterance_secs += len(frame.audio) / (frame.sample_rate * 2 * frame.num_channels)
            return True
        if isinstance(frame, UserStartedSpeakingFrame):
            # The bot-audio aggregator's ruling: the harness heard the bot speak.
            self._turn.stamp("bot_speech_onset", at)
            return True
        return False

    def _read_bot(self, frame: Frame, at: float) -> bool:
        """Time a frame the bot sent, as the input transport produced it."""
        if isinstance(frame, LLMFullResponseStartFrame):
            self._turn.stamp("llm_started", at)
        elif isinstance(frame, LLMTextFrame):
            if self._turn.has("llm_started"):
                self._turn.stamp("first_token", at)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._turn.has("llm_started"):
                self._turn.stamp("llm_response", at)
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._turn.stamp("function_call", at)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._turn.stamp("bot_started_speaking", at)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._turn.has("bot_started_speaking"):
                self._turn.stamp("bot_stopped_speaking", at)
        elif isinstance(frame, MetricsFrame):
            self._turn.add_metrics(at, _bot_metrics(frame.data))
        else:
            return False
        return True

    def _read_output(self, frame: Frame, at: float) -> bool:
        """Time the output transport's report on the user's utterance going out."""
        if isinstance(frame, BotStoppedSpeakingFrame):
            # The utterance's last chunk went out: the user stopped speaking,
            # and the turn is measured from here.
            if self._utterance_secs:
                self._turn.rebase(at, _ms(self._utterance_secs))
                self._utterance_secs = 0.0
            return True
        return isinstance(frame, BotStartedSpeakingFrame)
