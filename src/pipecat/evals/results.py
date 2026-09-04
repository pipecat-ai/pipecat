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
    "harness_error",  # the harness itself raised (sub-pipeline, judge, ...)
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
class EvalScriptTurnResult:
    """Outcome of one turn within a scenario run.

    The turn is the unit a run is scored by: a turn's expectations share a single
    deadline anchored at the send and stop at the first one to time out, so they
    are not scored independently of each other.

    Parameters:
        turn_index: Index of the turn in the scenario.
        status: One of ``TURN_STATUSES``. ``not_run`` means the run ended before
            reaching this turn — see
            :attr:`~pipecat.evals.script.EvalScriptScenario.stop_on_failure`.
        failures: The turn's failed assertions, in order; empty unless ``status``
            is ``failed``.
        duration_ms: Wall-clock time the turn took, in milliseconds; 0 when the
            turn was not run.
    """

    turn_index: int
    status: str = "not_run"
    failures: list[EvalAssertionFailure] = field(default_factory=list)
    duration_ms: int = 0


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
    "error",  # the run did not complete (see ``error``)
)


@dataclass
class EvalSimulationMetricScore:
    """One judged quality criterion's outcome for a simulation run.

    Parameters:
        name: The metric's name, from the simulation file.
        score: 1.0 if the judge said the criterion held, else 0.0.
        reason: The judge's justification.
        weight: The metric's weight in :attr:`EvalSimulationResult.quality`.
    """

    name: str
    score: float
    reason: str = ""
    weight: float = 1.0


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
        quality: Weighted mean of the metrics' scores, or None without metrics.
        metrics: The judged quality criteria's outcomes.
        messages: The conversation, with the persona's turns as ``user`` messages
            and the bot's as ``assistant`` (the convention scenarios' judges use).
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
    quality: float | None = None
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
        """Whether the run completed and achieved its goal."""
        return self.error is None and self.succeeded


@dataclass
class EvalScriptTurnProgress:
    """A real-time progress record emitted while a turn runs (for verbose output).

    Parameters:
        turn_index: The turn being run.
        expectation_index: Index of the expectation, or -1 for turn-level records
            (the turn header, or a ``send_after`` that never fired).
        event_name: The expectation's event (or the user text for a turn header).
        status: ``turn`` (header), ``matched``, ``failed``, or ``timeout``.
        detail: Optional extra text (failure reason, user utterance, ...).
    """

    turn_index: int
    expectation_index: int
    event_name: str
    status: str
    detail: str = ""


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
    """Timestamped, turn-tagged trace of the harness's own decisions.

    Every part of the harness logs here (events received, sends, matcher
    progress, errors), and the lines become :attr:`EvalScriptResult.debug_log`. The
    tag is the turn the harness is currently *processing* (``[--]`` before the
    first turn). Because events are logged the moment they arrive, an event that
    lands while a turn is still waiting on ``send_after`` is tagged with that
    waiting turn even though it's the previous turn's output — the
    ``send_after: waiting`` / ``send:`` lines make that boundary visible.
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
