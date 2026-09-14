#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What an eval run produces.

Per-assertion failures, per-turn outcomes, the run's :class:`EvalScriptResult` (a
scenario) or :class:`EvalSimulationResult` (a simulation), the live progress
records, and the harness's own trace.
"""

import time
from dataclasses import dataclass, field

from pipecat.utils.deprecation import deprecated

# Categories for :attr:`EvalAssertionFailure.kind`, the stable key for grouping
# failures across runs. Each says how an assertion failed, so a repeated suite can
# report "10x timeout on turn 3" without parsing free-text reasons.
FAILURE_KINDS = (
    "timeout",  # no event of the expected type arrived within the budget
    "judge_no",  # the judge rejected the reply
    "judge_continue",  # the judge never accepted the reply before the budget ran out
    "no_judge",  # the scenario uses `eval:` but no judge could be built
    "no_content",  # the matched event carried no text to judge
    "text_mismatch",  # `text_contains` not present in the event's text
    "missing_function_call",  # an expected function call never arrived
    "function_args_mismatch",  # the call arrived with unexpected arguments
    "unexpected_event",  # an `absent:` expectation saw the event it forbade
    "send_after_timeout",  # a turn's `send_after` event never fired
    "connect_failed",  # never connected to the bot's eval transport
    "handshake_timeout",  # connected, but the bot never sent bot-ready
    "judge_no_verdict",  # the judge answered nothing usable about a simulation's goal
    "error",  # the harness itself raised (a sub-pipeline, the judge, ...), not the bot
)

# Statuses for :attr:`EvalScriptTurnResult.status`. ``not_run`` is distinct from a pass:
# a run that stops at the first failure leaves its later turns undriven, and
# counting those as passes would inflate any rate computed from the result.
TURN_STATUSES = ("passed", "failed", "not_run")


@dataclass
class EvalAssertionFailure:
    """A single failed assertion within an eval.

    Parameters:
        turn_index: Index of the turn that failed.
        expectation_index: Index of the expectation within the turn, or -1 for a
            turn-level failure (e.g. a ``send_after`` that never fired).
        event_name: The expectation's event name.
        reason: Human-readable explanation of the failure.
        kind: Machine-readable failure category, one of ``FAILURE_KINDS``. Says
            *how* the assertion failed (the judge rejected the reply, no event
            arrived, a function call was missing, ...), not what it means about
            the bot. ``reason`` is free text and differs on every run — often
            judge prose — so grouping failures across many runs keys on this.
    """

    turn_index: int
    expectation_index: int
    event_name: str
    reason: str
    kind: str

    def __str__(self) -> str:
        return (
            f"turn {self.turn_index} expectation {self.expectation_index} "
            f"({self.event_name}): {self.reason}"
        )


@dataclass
class EvalTurnTiming:
    """When the bot's reply to one turn happened, measured by the harness.

    Every measure is milliseconds from the turn's *input anchor*: for a text
    turn the moment the ``send-text`` message was sent; for a spoken turn
    (synthesized or an ``audio:`` recording) the moment the last chunk of the
    utterance went out to the bot, i.e. when the user stopped speaking. The
    anchor is the end of speech when ``input_duration_ms`` is above zero and
    the send otherwise. A turn that sends nothing is anchored where the
    harness began observing it. A measure is ``None`` when its event did not
    occur after the anchor.

    This is not the clock an expectation's ``within_ms`` runs on: that budget
    is anchored at the send for every turn, spoken or not (see
    :mod:`pipecat.evals.script`).

    Parameters:
        input_duration_ms: Length of the user audio played for the turn; 0
            for a text or DTMF turn.
        llm_started_ms: The bot's LLM began a completion (``llm_started``).
        first_token_ms: The first chunk of LLM text arrived.
        llm_response_ms: The LLM response ended (``llm_response``).
        function_call_ms: The turn's first function call (``function_call``).
        bot_started_speaking_ms: The bot reported it started sending speech
            (``bot_started_speaking``).
        bot_speech_onset_ms: The harness's own VAD heard speech in the bot's
            audio. Audio mode only; includes the VAD's start window.
        bot_stopped_speaking_ms: The bot reported it stopped speaking
            (``bot_stopped_speaking``); only counted after it started.
        bot_metrics: The bot's own ``metrics`` reports that arrived during the
            turn, in order, each a dict of ``processor``, ``ttfb_ms``,
            ``processing_ms`` and ``tokens``, ``None`` for the parts a report
            did not carry. The bot's token usage is reported without a
            processor name.
    """

    input_duration_ms: int = 0
    llm_started_ms: int | None = None
    first_token_ms: int | None = None
    llm_response_ms: int | None = None
    function_call_ms: int | None = None
    bot_started_speaking_ms: int | None = None
    bot_speech_onset_ms: int | None = None
    bot_stopped_speaking_ms: int | None = None
    bot_metrics: list[dict] = field(default_factory=list)

    @property
    def voice_to_voice_ms(self) -> int | None:
        """From the end of the user's speech to the bot's audible speech.

        ``bot_speech_onset_ms`` when the turn was spoken (the anchor is then
        the end of the utterance); ``None`` for a text turn, whose anchor is
        the send, or when the bot's speech was not heard.
        """
        if not self.input_duration_ms:
            return None
        return self.bot_speech_onset_ms

    @property
    def speech_padding_ms(self) -> int | None:
        """The silence the bot sent before audible speech.

        The gap between the bot's own ``bot_started_speaking`` and the
        harness hearing its speech; ``None`` unless both were seen.
        """
        if self.bot_speech_onset_ms is None or self.bot_started_speaking_ms is None:
            return None
        return self.bot_speech_onset_ms - self.bot_started_speaking_ms


@dataclass
class EvalScriptTurnResult:
    """Outcome of one turn within a scenario run.

    A turn's expectations share one deadline and stop at the first to time
    out, so the turn is the unit a run is scored by.

    Parameters:
        turn_index: Index of the turn in the scenario.
        status: One of ``TURN_STATUSES``. ``not_run`` means the run ended before
            reaching this turn — see
            :attr:`~pipecat.evals.script.EvalScriptScenario.stop_on_failure`.
        failures: The turn's failed assertions, in order; empty unless ``status``
            is ``failed``.
        duration_ms: Wall-clock time the turn took, in milliseconds, judge
            latency included; 0 when the turn was not run.
        timing: When the bot's reply happened, relative to the turn's input
            (see :class:`EvalTurnTiming`); ``None`` when the turn was not
            driven to a send.
    """

    turn_index: int
    status: str = "not_run"
    failures: list[EvalAssertionFailure] = field(default_factory=list)
    duration_ms: int = 0
    timing: EvalTurnTiming | None = None


@deprecated(
    "`EvalTurnResult` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `EvalScriptTurnResult` instead."
)
@dataclass
class EvalTurnResult(EvalScriptTurnResult):
    """Deprecated alias for :class:`EvalScriptTurnResult`.

    .. deprecated:: 1.9.0
        Use :class:`EvalScriptTurnResult` instead. Will be removed in 2.0.0.
    """


@dataclass
class EvalScriptResult:
    """Outcome of running a scenario in an :class:`~pipecat.evals.script_session.EvalScriptSession`.

    Parameters:
        scenario_name: Name of the scenario that was run.
        passed: Whether every assertion passed.
        failures: The assertions that failed, in order.
        turns: One :class:`EvalScriptTurnResult` per scenario turn, in order — what a
            per-turn pass rate is computed from, without needing the scenario
            file for a denominator. ``failures`` is these turns' failures
            flattened, plus any that belong to no turn (a failed connect).
        duration_ms: Wall-clock time the run took, in milliseconds.
        events_seen: Every friendly event observed, for diagnostics.
        debug_log: Timestamped trace of the harness's own decisions (events
            received, audio transcribed, matcher progress), for diagnosing flaky
            runs. Saved per-scenario by the orchestrator alongside the bot log.
        skipped: When set, the scenario was not run (e.g. a ``tts_response``
            assertion without audio mode); the string is the reason. Such a result
            is neither passed nor failed.
    """

    scenario_name: str
    passed: bool
    failures: list[EvalAssertionFailure] = field(default_factory=list)
    turns: list[EvalScriptTurnResult] = field(default_factory=list)
    duration_ms: int = 0
    events_seen: list[dict] = field(default_factory=list)
    debug_log: list[str] = field(default_factory=list)
    skipped: str | None = None


@deprecated(
    "`EvalResult` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `EvalScriptResult` instead."
)
@dataclass
class EvalResult(EvalScriptResult):
    """Deprecated alias for :class:`EvalScriptResult`.

    .. deprecated:: 1.9.0
        Use :class:`EvalScriptResult` instead. Will be removed in 2.0.0.
    """


# How a simulation run came to an end, for :attr:`EvalSimulationResult.ended_by`.
SIMULATION_ENDINGS = (
    "end_call",  # the persona called its end_call tool
    "bot",  # the bot ended the call (it closed the connection)
    "max_turns",  # the persona's turn cap was reached
    "max_duration",  # the run's wall-clock cap was reached
    "silence",  # neither side did anything for ``max_silence_s``
    "error",  # the run did not complete (see ``error``)
)


@dataclass
class EvalSimulationTurnVerdict:
    """The judge's verdict on one bot turn, for a per-turn metric.

    Parameters:
        turn: The bot turn, 1-based, counting the turns in which the bot said
            something; it indexes the ``assistant`` messages of the run's
            conversation.
        passed: Whether the turn satisfied the criterion.
        reason: The judge's one-sentence justification.
        verdict: The judge's answer: ``yes``, ``no``, or ``none`` when it gave
            no verdict on the turn, which counts as a no.
    """

    turn: int
    passed: bool
    reason: str
    verdict: str = "no"


@dataclass
class EvalSimulationMetricScore:
    """One quality metric's outcome for a simulation run.

    Parameters:
        name: The metric's name, from the simulation file.
        score: The share of the bot's turns the judge answered yes for, in
            0..1, each turn a yes or a no; ``None`` when there was no turn to
            judge.
        passed: Whether the metric let the run pass: its score reached its
            ``min_score``, or it has none.
        reason: What the score rests on: the turns that fell short and why,
            or that every turn passed.
        min_score: The score the metric needed, or ``None`` when it only
            reports.
        verdicts: The judge's verdict on each bot turn, in order; empty for a
            measured metric.
        value: What a measured metric measured, in its unit; ``None`` for a
            judged one, or when there was nothing to measure.
        failure_kind: How the metric failed, for grouping across runs: ``judge_no``
            when the judge rejected a turn, ``judge_no_verdict`` when it only
            left turns unanswered, ``out_of_range`` for a measure outside its
            bounds, ``function_calls`` for a call list that did not match;
            ``None`` when it passed.
    """

    name: str
    score: float | None
    passed: bool = True
    reason: str = ""
    min_score: float | None = None
    verdicts: list[EvalSimulationTurnVerdict] = field(default_factory=list)
    value: float | None = None
    failure_kind: str | None = None


@dataclass
class EvalSimulationResult:
    """Outcome of one run of a simulation.

    Parameters:
        simulation_name: Name of the simulation that was run.
        succeeded: Whether the judge decided the goal was achieved.
        reason: The judge's justification for ``succeeded``, or the error.
        error: When set, the run did not complete (a failed connect, a harness
            error); ``succeeded`` is then False and the run is neither a goal
            success nor a goal failure.
        metrics: The quality metrics' outcomes.
        messages: The conversation, with the persona's turns as ``user`` messages
            and the bot's as ``assistant`` (the convention scenarios' judges use);
            the ``assistant`` messages are the bot turns the metrics scored.
        turns: How many turns the persona took.
        ended_by: How the run ended, one of ``SIMULATION_ENDINGS``.
        end_call: The persona's own ``end_call`` claim (``success``, ``reason``)
            when it made one; advisory, the judge decides ``succeeded``.
        duration_ms: Wall-clock time the run took, in milliseconds.
        events_seen: Every friendly event observed, for diagnostics.
        debug_log: Timestamped trace of the harness's own decisions.
    """

    simulation_name: str
    succeeded: bool
    reason: str = ""
    error: str | None = None
    metrics: list[EvalSimulationMetricScore] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    turns: int = 0
    ended_by: str = "error"
    end_call: dict | None = None
    duration_ms: int = 0
    events_seen: list[dict] = field(default_factory=list)
    debug_log: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the run completed, achieved its goal, and no metric fell short."""
        return self.error is None and self.succeeded and all(m.passed for m in self.metrics)

    @property
    def failure(self) -> str | None:
        """Why the run did not pass, or ``None``: the error, the goal, or the first failed metric."""
        if self.error is not None:
            return self.error
        if not self.succeeded:
            return f"goal not met: {self.reason}"
        for metric in self.metrics:
            if metric.passed:
                continue
            if metric.min_score is None:
                return f"{metric.name}: {metric.reason}"
            score = "unscored" if metric.score is None else f"{metric.score:.2f}"
            return f"{metric.name} {score} below {metric.min_score:.2f}: {metric.reason}"
        return None


@dataclass
class EvalScriptTurnProgress:
    """A real-time progress record emitted while a turn runs (for verbose output).

    Parameters:
        turn_index: The turn being run.
        expectation_index: Index of the expectation, or -1 for turn-level records
            (the turn header, or a ``send_after`` that never fired).
        event_name: The expectation's event (or the user text for a turn header).
        status: ``turn`` (header), ``matched``, ``failed``, ``timeout``, or
            ``timing`` (the turn's latency summary, once its expectations
            resolved; ``expectation_index`` is -1 and ``event_name`` empty).
        detail: Optional extra text (failure reason, user utterance, the
            timing summary, ...).
    """

    turn_index: int
    expectation_index: int
    event_name: str
    status: str
    detail: str = ""


@dataclass
class EvalSimulationProgress:
    """A real-time progress record emitted while a simulation runs (for verbose output).

    Parameters:
        status: ``bot`` for a response the bot finished, ``user`` for a turn the
            persona spoke, or ``ended`` once the conversation is over.
        text: What was said; for ``ended``, how the conversation ended
            (:data:`SIMULATION_ENDINGS`).
        turn: The persona's turn count so far.
    """

    status: str
    text: str
    turn: int


# What a session's ``on_progress`` handlers receive: a scripted scenario's
# per-turn records, or a simulation's conversation as it happens.
EvalProgress = EvalScriptTurnProgress | EvalSimulationProgress


@deprecated(
    "`EvalTurnProgress` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `EvalScriptTurnProgress` instead."
)
@dataclass
class EvalTurnProgress(EvalScriptTurnProgress):
    """Deprecated alias for :class:`EvalScriptTurnProgress`.

    .. deprecated:: 1.9.0
        Use :class:`EvalScriptTurnProgress` instead. Will be removed in 2.0.0.
    """


class EvalTrace:
    """Timestamped, turn-tagged log of the harness's own decisions, kept as the result's ``debug_log``.

    The tag is the turn the harness is processing (``[--]`` before the first).
    An event that lands while a turn waits on ``send_after`` is tagged with
    that turn even though it is the previous turn's output; the ``send:``
    lines mark the boundary.
    """

    def __init__(self):
        """Initialize an empty trace; :meth:`start` anchors its clock."""
        self.lines: list[str] = []
        # Index of the turn being processed; -1 outside any turn.
        self.turn: int = -1
        self._t0: float = 0.0

    def start(self) -> None:
        """Anchor the trace's timestamps at now (the start of the run)."""
        self._t0 = time.monotonic()

    def log(self, msg: str) -> None:
        """Append one line, stamped with the seconds since :meth:`start` and the turn."""
        t = time.monotonic() - self._t0 if self._t0 else 0.0
        tag = f"t{self.turn}" if self.turn >= 0 else "--"
        self.lines.append(f"{t:8.3f}  [{tag:>3}]  {msg}")
