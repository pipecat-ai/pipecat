#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session: drives a bot over RTVI and asserts on the events it emits.

An :class:`EvalSession` connects to a running bot's eval transport (a
``SingleClientWebsocketServerTransport`` speaking RTVI via
:class:`~pipecat.evals.serializer.RTVIEvalSerializer`), walks through a parsed
:class:`~pipecat.evals.scenario.EvalScenario`, and verifies that the expected
semantic events arrive in order, with the right payloads, within their latency
budgets. It returns an :class:`EvalResult`.

The session is a thin RTVI client. It builds outgoing messages with the RTVI
models (:mod:`pipecat.processors.frameworks.rtvi.models`) and translates the
RTVI server messages it receives back into a small set of friendly event names
the scenario files assert on:

==========================      ==============================================
scenario ``event:``             RTVI server message(s)
==========================      ==============================================
``user_started_speaking``       ``user-started-speaking``
``user_stopped_speaking``       ``user-stopped-speaking``
``vad_user_started_speaking``   ``vad-user-started-speaking`` (raw VAD, ungated by turn detection)
``vad_user_stopped_speaking``   ``vad-user-stopped-speaking`` (raw VAD, ungated by turn detection)
``user_transcription``          ``user-transcription`` (final only)
``llm_started``                 ``bot-llm-started``
``llm_response``                the LLM text: ``bot-llm-text`` joined at ``bot-llm-stopped``
``tts_response``                the TTS's spoken text: one segment per ``bot-tts-text``
                                (audio modality only)
``response``                    local-STT transcription of the bot's actual audio
                                (audio modality only); ``llm_response`` in text modality
``function_call``               ``llm-function-call-in-progress``
``function_call_stopped``       ``llm-function-call-stopped``; its ``args`` carry
                                ``tool_call_id`` and ``cancelled``, so a scenario
                                can tell work that was stopped from work that
                                finished on its own
==========================      ==============================================

Matching semantics: expected events must appear in the specified order, but
unmatched events may appear between them (so a scenario doesn't have to
enumerate every event the bot emits). The ``within_ms`` budget for each
expectation is measured from the most recent ``send-text`` / ``raw-audio`` / ``dtmf`` send
(default 60s when omitted).

A turn with a failed assertion ends the scenario, since the conversation is in
an unknown state from there on. A scenario that scores each turn independently
sets ``stop_on_failure: false`` to have every turn driven and reported.

An ``llm_response`` with a content check (``text_contains`` / ``eval:``)
aggregates: the harness accumulates the text of successive response segments
within the turn and re-checks on each one, so an interim filler ("Let me check
on that.") or the on-connect greeting is rolled past rather than mistaken for
the turn's answer. Responses that began before the turn's input are skipped, so
an interrupted prior turn doesn't bleed in. The judge returns yes / no /
continue; ``text_contains`` treats a missing substring as continue. The
``within_ms`` budget bounds the wait. A ``user_transcription`` with
``text_contains`` aggregates the same way: an STT may finalize one utterance in
several pieces, and the check runs on the pieces accumulated so far. Substring
checks ignore differences in whitespace, so pieces that carry their own spacing
still match a phrase.

Example::

    scenario = EvalScenario.load("scenarios/greeting.yaml")
    result = await EvalSession.from_scenario(scenario, "ws://localhost:7860").run()
    if result.passed:
        print("PASS")
    else:
        for f in result.failures:
            print(f"  {f}")

    # Per-turn outcomes, for a scenario scored a turn at a time.
    scored = [t for t in result.turns if t.status != "not_run"]
    print(f"{sum(1 for t in scored if t.status == 'passed')}/{len(scored)} turns")
"""

import asyncio
import base64
import mimetypes
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from loguru import logger
from websockets.asyncio.client import ClientConnection

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.audio import load_user_audio
from pipecat.evals.judge import EvalJudge
from pipecat.evals.scenario import (
    FUNCTION_CALL_EVENTS,
    EvalExpectation,
    EvalScenario,
    EvalSendAfter,
    EvalTurn,
    describe_config,
)
from pipecat.evals.serializer import (
    EVAL_CANCEL_MESSAGE_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
)
from pipecat.evals.speech import EvalSpeech
from pipecat.evals.transcribe import EvalTranscriber
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.rtvi_client import RTVIClientSerializer
from pipecat.transports.websocket.client import WebsocketClientParams
from pipecat.transports.websocket.rtvi_client import RTVIClientTransport
from pipecat.utils.base_object import BaseObject
from pipecat.workers.runner import WorkerRunner

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000
SEND_AFTER_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01
BOT_READY_TIMEOUT_S = 10.0

# Audio injection: each synthesized utterance is sent as a few large ``raw-audio``
# messages (sliced to stay well under the websocket message-size limit). The eval
# transport's virtual mic (``pipecat.evals.transport.EvalMicrophone``) plays them
# into the bot's pipeline at real-time cadence with silence in between, so the
# harness doesn't pace frames — and no continuous frame stream crosses the wire.
SEND_CHUNK_MS = 1000

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

# Statuses for :attr:`EvalTurnResult.status`. ``not_run`` is distinct from a pass:
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
class EvalTurnResult:
    """Outcome of one turn within a scenario run.

    The turn is the unit a run is scored by: a turn's expectations share a single
    deadline anchored at the send and stop at the first one to time out, so they
    are not scored independently of each other.

    Parameters:
        turn_index: Index of the turn in the scenario.
        status: One of ``TURN_STATUSES``. ``not_run`` means the run ended before
            reaching this turn — see
            :attr:`~pipecat.evals.scenario.EvalScenario.stop_on_failure`.
        failures: The turn's failed assertions, in order; empty unless ``status``
            is ``failed``.
        duration_ms: Wall-clock time the turn took, in milliseconds; 0 when the
            turn was not run.
    """

    turn_index: int
    status: str = "not_run"
    failures: list[EvalAssertionFailure] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class EvalResult:
    """Outcome of running a scenario in an :class:`EvalSession`.

    Parameters:
        scenario_name: Name of the scenario that was run.
        passed: Whether every assertion passed.
        failures: The assertions that failed, in order.
        turns: One :class:`EvalTurnResult` per scenario turn, in order — what a
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
    turns: list[EvalTurnResult] = field(default_factory=list)
    duration_ms: int = 0
    events_seen: list[dict] = field(default_factory=list)
    debug_log: list[str] = field(default_factory=list)
    skipped: str | None = None


@dataclass
class EvalTurnProgress:
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


class _BotFrameSink(FrameProcessor):
    """Pipeline tap that turns the bot's incoming frames into matcher events.

    Sits at the end of the eval pipeline; for every frame the
    :class:`RTVIClientTransport` produces it calls back into the session's
    :meth:`EvalSession._frames_to_events` and enqueues the results for the
    matcher, then passes the frame on. Outgoing frames (the RTVI client messages
    the session injects) flow through untouched — they don't map to events.
    """

    def __init__(self, session: "EvalSession"):
        super().__init__()
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        for event in self._session._frames_to_events(frame):
            await self._session._enqueue(event)
        await self.push_frame(frame, direction)


class EvalSession(BaseObject):
    """Runs one :class:`EvalScenario` against a bot over a single WebSocket session.

    Connects as an RTVI client, drives each turn (sending ``send-text``,
    ``raw-audio``, or ``dtmf``), collects the RTVI events the bot emits, and asserts on them.
    Build one with :meth:`from_scenario` (which constructs the judge, speech, and
    transcriber the scenario needs), then await :meth:`run`.

    Event handlers available:

    - on_progress: Called with an :class:`EvalTurnProgress` as each turn and each
      expectation resolves. Records are emitted in order, and :meth:`run` waits for
      every handler before it returns.

    Example::

        @session.event_handler("on_progress")
        async def on_progress(session, progress):
            print(progress.event_name, progress.status)
    """

    def __init__(
        self,
        scenario: EvalScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
        on_progress: Callable[[EvalTurnProgress], None] | None = None,
        record_path: str | None = None,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        judge: EvalJudge | None = None,
        speech: EvalSpeech | None = None,
        transcriber: EvalTranscriber | None = None,
    ):
        """Initialize the eval session.

        The ``judge``, ``speech``, and ``transcriber`` are injected pre-built:
        :meth:`from_scenario` constructs the defaults from the scenario's config
        (via the respective ``from_config``) and passes them in. Construct and
        pass your own to override them (e.g. a custom judge LLM or TTS service).

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            default_timeout_ms: Per-expectation latency budget for expectations
                without their own ``within_ms`` (the turn's expectations share one
                deadline anchored at the send). Defaults to 60s.
            on_progress: Optional callback invoked with a :class:`EvalTurnProgress`
                as each turn and expectation resolves (used for verbose output).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            record_path: When set (and the scenario is audio mode), asks the eval
                transport to record the conversation audio to this path (bot-side).
            stop_bot: When True, ask the bot to cancel its pipeline (and exit) on
                teardown via ``eval-cancel``. The suite enables it to clean up
                each spawned bot.
            trigger_disconnect: When True (or when the scenario sets
                ``trigger_disconnect``), ask the eval transport to fire the bot's
                ``on_client_disconnected`` handler when this connection ends.
                Bots often cancel their pipeline there, so it is off by default
                to avoid that between scenarios.
            judge: The :class:`~pipecat.evals.judge.EvalJudge` for ``eval:``
                assertions, or ``None`` if the scenario has none.
            speech: The :class:`~pipecat.evals.speech.EvalSpeech` for synthesizing
                user audio, or ``None`` for text-mode scenarios. Started and
                stopped by the session.
            transcriber: The :class:`~pipecat.evals.transcribe.EvalTranscriber`
                for the ``response`` event, or ``None`` when unused. Started and
                stopped by the session.
        """
        super().__init__()

        self._scenario = scenario
        self._bot_url = bot_url
        self._connect_timeout_s = connect_timeout_s
        self._default_timeout_ms = default_timeout_ms
        self._record_path = record_path
        self._stop_bot = stop_bot
        # Either the run-wide CLI flag or the scenario's own field opts in.
        self._trigger_disconnect = trigger_disconnect or scenario.trigger_disconnect

        self._ws: ClientConnection | None = None
        # The eval pipeline's worker that talks to the bot (built in run()).
        self._worker: PipelineWorker | None = None
        # Set by the transport's on_bot_ready handler once the bot completes the
        # RTVI handshake; _handshake() waits on it.
        self._bot_ready_event = asyncio.Event()
        self._queue: asyncio.Queue = asyncio.Queue()
        # function_call events popped while matching another expectation, held so
        # the turn's calls can be matched by name in any order (reset per turn).
        self._pending_function_calls: list[dict] = []
        self._latest_event_times: dict[str, float] = {}
        self._events_seen: list[dict] = []
        # Timestamped trace of the harness's own decisions, for diagnosing flakes.
        self._debug_log: list[str] = []
        self._debug_t0: float = 0.0
        self._current_turn: int = -1
        self._next_id = 0
        self._judge: EvalJudge | None = judge

        # One persistent TTS pipeline reused across the scenario's audio turns,
        # started in run(); None for text-mode scenarios.
        self._speech: EvalSpeech | None = speech

        # Accumulates the bot's output text for the current response, to
        # synthesize llm_response. Source depends on the mode: bot-llm-text in
        # text mode (skip-TTS), bot-tts-text in audio mode (what was spoken).
        self._text_buffer: list[str] = []

        # Set on an interruption, cleared at the next bot-llm-started. While set,
        # llm_response segments are dropped: the interrupted response can still
        # flush a trailing token *after* the interruption event (it was generated
        # before the interrupt propagated), and that straggler must not be
        # attributed to the new turn. The genuinely new response begins at the
        # next bot-llm-started.
        self._awaiting_llm_restart: bool = False

        # Text content of the most recently matched event (the bot's response, or
        # a user transcript), surfaced to verbose progress. Empty for events with
        # no text (llm_started, function_call, speaking events).
        self._last_match_text: str = ""

        # response (audio modality): the harness captures the bot's actual audio
        # and transcribes it locally for the judge. Lazy — only set up when a
        # scenario asserts `response`.
        self._wants_response: bool = any(
            exp.event == "response" for turn in scenario.turns for exp in turn.expect
        )
        self._transcriber: EvalTranscriber | None = transcriber
        self._tts_audio: bytearray = bytearray()  # current spoken segment's audio
        self._tts_sample_rate: int = 0

        self._register_event_handler("on_progress")
        if on_progress is not None:
            self._add_legacy_progress_callback(on_progress)

    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
        on_progress: Callable[[EvalTurnProgress], None] | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        judge: EvalJudge | None = None,
        speech: EvalSpeech | None = None,
        transcriber: EvalTranscriber | None = None,
    ) -> "EvalSession":
        """Build a ready-to-run session from a scenario, constructing what it needs.

        Builds the judge, speech, and transcriber the scenario calls for — each via
        its ``from_config`` — and injects them into a new session. Pass ``judge`` /
        ``speech`` / ``transcriber`` to override any of them with your own pre-built
        instance. Then await :meth:`run`::

            session = EvalSession.from_scenario(scenario, "ws://localhost:7860")
            result = await session.run()

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            default_timeout_ms: Per-expectation latency budget for expectations
                without their own ``within_ms``. Defaults to 60s.
            on_progress: Optional per-turn/expectation progress callback (verbose).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            record_path: Optional path to record the conversation audio (audio mode).
            cache_dir: Optional directory for cached synthesized user audio
                (default ``<user-cache-dir>/pipecat/tts``).
            use_cache: When False, ignore cached user audio and force fresh synthesis
                (no cache reads or writes). Defaults to True.
            stop_bot: When True, ask the bot to cancel its pipeline (and exit) on
                teardown. Leave False to keep it running for more scenarios.
            trigger_disconnect: When True, fire the bot's ``on_client_disconnected``
                handler when the connection ends (the scenario's own
                ``trigger_disconnect`` field also opts in). Off by default.
            judge: Override the judge (default: built from ``scenario.judge`` when the
                scenario has ``eval:`` assertions).
            speech: Override the user-audio generator (default: built from
                ``scenario.user_speech`` in audio mode).
            transcriber: Override the bot-audio transcriber (default: built from
                ``scenario.transcriber`` when the scenario asserts ``response``).

        Returns:
            A configured session, ready for :meth:`run`.
        """
        turns = scenario.turns
        if judge is None and any(exp.eval is not None for turn in turns for exp in turn.expect):
            with logger.contextualize(eval_pipeline="judge"):
                judge = EvalJudge.from_config(scenario.judge)

        if speech is None and scenario.user_speech is not None:
            with logger.contextualize(eval_pipeline="speech"):
                speech = EvalSpeech.from_config(
                    scenario.user_speech, cache_dir=cache_dir, use_cache=use_cache
                )

        wants_response = any(exp.event == "response" for turn in turns for exp in turn.expect)
        if transcriber is None and wants_response and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                transcriber = EvalTranscriber.from_config(scenario.transcriber)

        session = cls(
            scenario,
            bot_url,
            connect_timeout_s=connect_timeout_s,
            default_timeout_ms=default_timeout_ms,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            judge=judge,
            speech=speech,
            transcriber=transcriber,
        )
        if on_progress is not None:
            session._add_legacy_progress_callback(on_progress)
        return session

    async def run(self) -> EvalResult:
        """Connect, drive the scenario, and return the result."""
        started = time.monotonic()
        self._debug_t0 = started
        self._debug(f"run: scenario {self._scenario.name!r} -> {self._bot_url}")
        # Record which speech / transcription / judge services and models were used,
        # so a saved eval.log is self-describing (no need to cross-reference config).
        for line in describe_config(self._scenario).splitlines():
            self._debug(line)

        # One record per scenario turn, filled in as the turns are driven. They
        # start as not_run and stay that way on every path that ends the run
        # early, so the result always says which turns were actually scored.
        turns = [EvalTurnResult(turn_index=i) for i in range(len(self._scenario.turns))]

        # The `response` transcription needs the bot's actual audio; without audio
        # mode there's nothing to transcribe, so skip rather than fail. (Normally
        # unreachable: EvalScenario.load resolves `response` to llm_response in text
        # modality; this guards Scenarios built directly.)
        if self._wants_response and not self._scenario.bot_audio:
            reason = "asserts 'response' transcription but judge modality is text (no audio)"
            logger.warning(f"Eval '{self._scenario.name}': {reason}; skipping")
            return EvalResult(
                scenario_name=self._scenario.name,
                passed=False,
                turns=turns,
                skipped=reason,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        # Readiness probe: retry-connect until the bot accepts (so callers can launch
        # the bot and connect immediately), then close it — the transport owns the
        # real session. A bot that never accepts is a clean <connect> failure. The
        # eval transport only fires on_client_disconnected when trigger_disconnect is
        # set, so this transient probe doesn't perturb the bot.
        deadline = time.monotonic() + self._connect_timeout_s
        connect_error: Exception | None = None
        while self._ws is None and time.monotonic() < deadline:
            try:
                self._ws = await websockets.connect(self._connect_url())
            except OSError as e:  # not accepting connections yet
                connect_error = e
                await asyncio.sleep(0.25)
        if self._ws is None:
            e = connect_error or TimeoutError("timed out")
            return EvalResult(
                scenario_name=self._scenario.name,
                passed=False,
                failures=[
                    EvalAssertionFailure(
                        turn_index=-1,
                        expectation_index=-1,
                        event_name="<connect>",
                        reason=f"failed to connect to {self._bot_url}: {e.__class__.__name__}",
                        kind="connect_failed",
                    )
                ],
                turns=turns,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        await self._ws.close()
        self._ws = None

        # Build the eval pipeline: the RTVI client transport talks to the bot, and a
        # sink turns the bot's frames into matcher events.
        params = WebsocketClientParams(
            audio_in_enabled=self._scenario.bot_audio,
            audio_out_enabled=self._scenario.bot_audio,
            serializer=RTVIClientSerializer(),
        )
        transport = RTVIClientTransport(self._connect_url(), params)

        @transport.event_handler("on_bot_ready")
        async def _on_bot_ready(_transport):
            self._bot_ready_event.set()

        pipeline = Pipeline([transport.input(), _BotFrameSink(self), transport.output()])
        worker = PipelineWorker(pipeline, enable_rtvi=False, cancel_on_idle_timeout=False)
        self._worker = worker
        runner = WorkerRunner()
        await runner.add_workers(worker)
        run_task = asyncio.create_task(runner.run())

        failures: list[EvalAssertionFailure] = []
        try:
            # Start the injected sub-pipelines (built by from_scenario from the
            # scenario config). Each tags its logs with an ``eval_pipeline`` label
            # via logger.contextualize: the tasks created here inherit it
            # (contextvars copy into asyncio tasks), so the underlying service's
            # logs carry the label too, regardless of which TTS/STT/LLM service is
            # used. The CLI routes each label to its own log file (see
            # _LOG_CATEGORIES). These run under the same `try` as the turns so a
            # sub-pipeline that fails to start (e.g. a local model under load) is
            # surfaced as a failure rather than propagating out raw (see below).
            if self._speech is not None:
                with logger.contextualize(eval_pipeline="speech"):
                    await self._speech.start()

            if self._transcriber is not None:
                with logger.contextualize(eval_pipeline="transcription"):
                    self._transcriber.debug = self._debug
                    await self._transcriber.start()

            self._debug("connected")
            try:
                await self._handshake()
                self._debug("handshake: ok (bot-ready)")
            except TimeoutError:
                self._debug("handshake: failed (bot-ready not received)")
                failures.append(
                    EvalAssertionFailure(
                        turn_index=-1,
                        expectation_index=-1,
                        event_name="<bot-ready>",
                        reason=f"bot-ready not received within {int(BOT_READY_TIMEOUT_S * 1000)}ms",
                        kind="handshake_timeout",
                    )
                )
            else:
                for turn_idx, turn in enumerate(self._scenario.turns):
                    self._current_turn = turn_idx
                    self._debug(f"--- turn {turn_idx}: {turn.user!r}")
                    turn_started = time.monotonic()
                    turn_failures = await self._run_turn(turn, turn_idx)
                    record = turns[turn_idx]
                    record.status = "failed" if turn_failures else "passed"
                    record.failures = turn_failures
                    record.duration_ms = int((time.monotonic() - turn_started) * 1000)
                    failures.extend(turn_failures)
                    if turn_failures:
                        # By default a failed turn ends the scenario: it leaves the
                        # conversation in an unknown state, so running the rest just
                        # burns another timeout per turn (e.g. a broken greeting turn
                        # shouldn't cost the full budget here and again on the
                        # question). A scenario whose turns are scored independently
                        # sets stop_on_failure: false and drives all of them.
                        if self._scenario.stop_on_failure:
                            self._debug(
                                f"turn {turn_idx} failed; stopping scenario (stop_on_failure)"
                            )
                            break
                        self._debug(f"turn {turn_idx} failed; continuing (stop_on_failure: false)")
        except Exception as e:
            # An unexpected harness-side error (a sub-pipeline failing to start
            # under load, a judge/transcriber raising mid-turn, ...) would
            # otherwise propagate up to the suite and be swallowed as a bare
            # "error: <str>" with no eval.log. Capture it as a failure so the
            # reason and full traceback land in the result's debug trace (saved
            # to <bot>.eval.log) and the run still reports a structured outcome.
            self._debug(f"error: {type(e).__name__}: {e}")
            for line in traceback.format_exc().rstrip().splitlines():
                self._debug(line)
            failure = EvalAssertionFailure(
                turn_index=self._current_turn,
                expectation_index=-1,
                event_name="<error>",
                reason=f"{type(e).__name__}: {e}",
                kind="harness_error",
            )
            failures.append(failure)
            # The raise happened either inside a turn — which is that turn's
            # failure — or before any of them started (a sub-pipeline that never
            # came up), where _current_turn is still -1 and every turn is not_run.
            if 0 <= self._current_turn < len(turns):
                record = turns[self._current_turn]
                record.status = "failed"
                record.failures.append(failure)
        finally:
            # Tear each sub-pipeline down under the same eval_pipeline label as its
            # setup, so its shutdown logs are attributed to it.
            if self._speech is not None:
                with logger.contextualize(eval_pipeline="speech"):
                    await self._speech.aclose()
            if self._transcriber is not None:
                with logger.contextualize(eval_pipeline="transcription"):
                    await self._transcriber.aclose()
            # Optionally ask the bot to tear its pipeline down gracefully so it exits
            # on its own (best-effort; skipped by default so it stays up for more
            # scenarios).
            if self._stop_bot:
                await self._send_cancel()
            # Stop the eval pipeline: end the worker (which disconnects the
            # transport), falling back to cancel if it doesn't wind down cleanly.
            try:
                await self._worker.queue_frame(EndFrame())
                await asyncio.wait_for(run_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Progress handlers run as tasks; wait them out so every record is
            # delivered before the caller has the result in hand.
            await self.cleanup()

        self._debug(f"done: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures,
            failures=failures,
            turns=turns,
            duration_ms=int((time.monotonic() - started) * 1000),
            events_seen=self._events_seen,
            debug_log=self._debug_log,
        )

    def _connect_url(self) -> str:
        """Bot URL with the per-connection eval query flags.

        ``skip_tts`` (text mode) silences the bot before any LLM runs; the eval
        transport must read it at connect time because frames are ordered and a
        later message can't precede an on-connect greeting (see
        :mod:`pipecat.evals.transport`). ``user_audio`` turns on the transport's
        virtual mic whenever the harness sends audio, whether synthesized or
        played from a turn's ``audio:`` file; without it the transport plays no
        mic at all, so a text-mode scenario never feeds silence into the bot's
        STT. ``capture_bot_audio`` makes the bot forward its synthesized audio for
        ``tts_response`` transcription. ``record`` asks the eval transport to
        record the conversation audio (audio mode only). ``trigger_disconnect``
        asks the transport to fire the bot's ``on_client_disconnected`` handler
        when the connection ends (off by default, since bots often cancel there).
        """
        from urllib.parse import quote

        flags = []
        if not self._scenario.bot_audio:
            flags.append("skip_tts=true")
        if self._speech is not None or any(t.audio for t in self._scenario.turns):
            flags.append("user_audio=true")
        if self._wants_response:
            flags.append("capture_bot_audio=true")
        if self._record_path and self._scenario.bot_audio:
            flags.append(f"record={quote(self._record_path, safe='')}")
        if self._trigger_disconnect:
            flags.append("trigger_disconnect=true")
        if not flags:
            return self._bot_url
        sep = "&" if "?" in self._bot_url else "?"
        return f"{self._bot_url}{sep}{'&'.join(flags)}"

    def _message_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def _required_report_level(self) -> str | None:
        """Minimal function-call report level the scenario's assertions need.

        Returns ``"full"`` if any ``function_call`` expectation checks ``args``,
        ``"name"`` if one checks ``name`` only, else ``None`` (no elevation —
        the bot's default applies and a function_call event still arrives).
        """
        needs_name = False
        for turn in self._scenario.turns:
            for exp in turn.expect:
                if exp.event not in FUNCTION_CALL_EVENTS:
                    continue
                # name/args live in exp.calls (the parser normalizes the single
                # name:/args: shorthand into it too).
                for call in exp.calls or []:
                    if call.args is not None:
                        return "full"
                    if call.name is not None:
                        needs_name = True
        return "name" if needs_name else None

    def _needs_vad_events(self) -> bool:
        """Whether the scenario references raw VAD speaking events.

        These (``vad_user_started_speaking`` / ``vad_user_stopped_speaking``) are
        off by default; the harness asks the bot's RTVIObserver to emit them only
        when a scenario asserts on or schedules from them.
        """
        vad_events = {"vad_user_started_speaking", "vad_user_stopped_speaking"}
        for turn in self._scenario.turns:
            if turn.send_after is not None and turn.send_after.event in vad_events:
                return True
            if any(exp.event in vad_events for exp in turn.expect):
                return True
        return False

    async def _handshake(self) -> None:
        """Send client-ready, wait for bot-ready, then optionally seed context.

        ``bot-ready`` is a hard gate: the eval framework requires an RTVI bot, so a
        bot that never announces readiness either isn't a valid eval target or
        hasn't finished starting (services still connecting). Rather than fire
        turns at a half-started bot — which produces flaky, hard-to-read failures —
        we raise :class:`TimeoutError` so the caller reports a clean connect-level
        failure.
        """
        # The transport sends client-ready on connect and fires on_bot_ready when
        # the bot answers; our handler sets _bot_ready_event. Hard gate — raises
        # TimeoutError if the bot never announces readiness.
        await asyncio.wait_for(self._bot_ready_event.wait(), timeout=BOT_READY_TIMEOUT_S)

        # Ask the bot's RTVIObserver to expose what this scenario needs, for the
        # duration of this eval only (bots keep their defaults; only the eval
        # transport understands this): raise the function-call report level if it
        # asserts on call name/args, and enable raw VAD speaking events if it uses
        # them.
        level = self._required_report_level()
        vad = self._needs_vad_events()
        if level is not None or vad:
            config: dict = {}
            if level is not None:
                config["function_call_report_level"] = {"*": level}
            if vad:
                config["vad_user_speaking"] = True
            configure = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CONFIGURE_MESSAGE_TYPE, "d": config},
            )
            await self._send(configure)

        # Only send the eval-context when the scenario provides starting context.
        # An implicit empty one would race with bot startup flows (e.g. a greeting
        # added in on_client_connected), wiping the bot's context right after it
        # set it up.
        if self._scenario.context:
            context_message = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CONTEXT_MESSAGE_TYPE, "d": {"messages": self._scenario.context}},
            )
            await self._send(context_message)

    async def _send(self, message: RTVI.Message) -> None:
        """Send an RTVI client message to the bot through the transport pipeline."""
        assert self._worker is not None  # pipeline built before any send
        await self._worker.queue_frame(
            OutputTransportMessageUrgentFrame(message=message.model_dump())
        )

    async def _send_cancel(self) -> None:
        """Ask the bot to cancel its pipeline so it shuts down gracefully.

        Best-effort: the connection may already be gone, in which case the
        orchestrator's kill fallback handles teardown.
        """
        try:
            message = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CANCEL_MESSAGE_TYPE, "d": {}},
            )
            await self._send(message)
        except Exception:
            pass

    def _debug(self, msg: str) -> None:
        """Append a timestamped, turn-tagged line to the per-scenario debug trace.

        The tag is the turn the harness is currently *processing* (``[--]`` before
        the first turn). Because events are logged the moment they arrive, an event
        that lands while a turn is still waiting on ``send_after`` is tagged with
        that waiting turn even though it's the previous turn's output — the
        ``send_after: waiting`` / ``send:`` lines make that boundary visible.
        """
        t = time.monotonic() - self._debug_t0 if self._debug_t0 else 0.0
        tag = f"t{self._current_turn}" if self._current_turn >= 0 else "--"
        self._debug_log.append(f"{t:8.3f}  [{tag:>3}]  {msg}")

    async def _enqueue(self, event: dict) -> None:
        """Record and queue a friendly event for the matcher."""
        self._events_seen.append(event)
        self._latest_event_times[event["type"]] = time.monotonic()
        preview = event.get("text") or event.get("transcript") or event.get("name") or ""
        self._debug(f"event: {event['type']}" + (f"  {str(preview)!r}" if preview else ""))
        await self._queue.put(event)

    def _drop_pending_bot_output(self, why: str) -> None:
        """Drop the bot's un-matched output, so a later turn can't match it.

        Clears the response buffers and drains the bot's pending output from the
        event queue, so a greeting (or any prior bot output) can't be matched
        against this turn. ``user_transcription`` is preserved: a DTMF keypress
        emits its transcription immediately before the turn-start interruption,
        and that transcription is the turn's *input*, not the stale bot output
        this discard is meant to clear — dropping it would race the matcher.
        Diagnostics (``events_seen``, ``latest_event_times``) are left intact for
        send_after lookups.

        Args:
            why: What prompted the drop, for the debug trace.
        """
        self._text_buffer = []
        self._tts_audio = bytearray()
        preserved: list[dict] = []
        dropped = 0
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if event.get("type") == "user_transcription":
                preserved.append(event)
            else:
                dropped += 1
        for event in preserved:
            self._queue.put_nowait(event)
        if dropped:
            self._debug(f"discard: dropped {dropped} queued event(s) {why}")

    def _translate(self, message: dict) -> list[dict]:
        """Translate one RTVI server message into zero or more friendly events.

        .. note::
            Superseded by :meth:`_frames_to_events` now that the harness consumes
            the transport's frames; kept (with ``TestTranslate``) as the reference
            for the message-name -> event mapping.
        """
        msg_type = message.get("type")
        data = message.get("data") or {}

        match msg_type:
            case "bot-ready":
                return [{"type": "bot_ready"}]
            case "user-started-speaking":
                # A new user turn in audio mode. Drop any leftover bot output from
                # a prior turn so it isn't aggregated into this one.
                self._drop_pending_bot_output("on interruption")
                self._awaiting_llm_restart = True
                return [{"type": "user_started_speaking"}]
            case "bot-interrupted":
                # The bot's in-flight output was cut off — a VAD barge-in or a
                # run_immediately text interrupt. Drop it so only what the bot says
                # *after* the interruption is matched. Service-independent, the same
                # path for both modalities, and no timestamps.
                self._drop_pending_bot_output("on interruption")
                self._awaiting_llm_restart = True
                return [{"type": "bot_interrupted"}]
            case "user-stopped-speaking":
                return [{"type": "user_stopped_speaking"}]
            case "vad-user-started-speaking":
                return [{"type": "vad_user_started_speaking"}]
            case "vad-user-stopped-speaking":
                return [{"type": "vad_user_stopped_speaking"}]
            case "user-transcription":
                if data.get("final"):
                    return [{"type": "user_transcription", "transcript": data.get("text", "")}]
                return []
            case "bot-llm-started":
                # The genuinely new response begins here, so stragglers from an
                # interrupted prior response are now behind us.
                self._awaiting_llm_restart = False
                self._text_buffer = []
                return [{"type": "llm_started"}]
            case "bot-llm-text":
                # The LLM's text output -> llm_response (both modalities). Buffer
                # it and emit one segment at bot-llm-stopped (a clean boundary —
                # bot-llm-text reliably precedes bot-llm-stopped).
                if self._awaiting_llm_restart:
                    return []
                self._text_buffer.append(data.get("text", ""))
                return []
            case "bot-llm-stopped":
                # A stopped that arrives before the post-interruption restart is the
                # tail of the interrupted response; drop it instead of emitting it
                # as this turn's llm_response.
                if self._awaiting_llm_restart:
                    self._text_buffer = []
                    return []
                return [self._segment_event("llm_response", "".join(self._text_buffer))]
            case "bot-tts-text":
                # Audio mode: the text the TTS reports speaking -> tts_response,
                # one segment per spoken sentence, emitted as it arrives. We can't
                # bound on bot-tts-stopped because some TTS services emit the text
                # *after* the audio finishes (e.g. OpenAI). The matcher aggregates
                # the segments of the turn.
                if self._scenario.bot_audio:
                    return [self._segment_event("tts_response", data.get("text", ""))]
                return []
            case "bot-tts-stopped":
                return []
            case "llm-function-call-in-progress":
                return [
                    {
                        "type": "function_call",
                        "name": data.get("function_name"),
                        "args": dict(data.get("arguments") or {}),
                    }
                ]
            case "llm-function-call-stopped":
                # How the call ended is the assertable part, so `cancelled` sits
                # in `args` alongside the id: a scenario matches both through the
                # same `calls:`/`args:` check a function_call uses.
                return [
                    {
                        "type": "function_call_stopped",
                        "name": data.get("function_name"),
                        "args": {
                            "tool_call_id": data.get("tool_call_id"),
                            "cancelled": data.get("cancelled"),
                        },
                    }
                ]
            case _:
                return []

    def _frames_to_events(self, frame: Frame) -> list[dict]:
        """Translate one incoming pipeline frame into zero or more friendly events.

        The :class:`~pipecat.transports.websocket.rtvi_client.RTVIClientTransport`
        deserializes the bot's RTVI server messages into frames; this maps those
        frames to the events the matcher consumes, applying the modality/aggregation
        rules (buffer the LLM text, suppress an interrupted response's straggler,
        emit ``tts_response`` only in audio mode, etc.).
        """
        if isinstance(frame, UserStartedSpeakingFrame):
            self._discard_interrupted_output()
            self._awaiting_llm_restart = True
            return [{"type": "user_started_speaking"}]
        if isinstance(frame, InterruptionFrame):
            self._discard_interrupted_output()
            self._awaiting_llm_restart = True
            return [{"type": "bot_interrupted"}]
        if isinstance(frame, UserStoppedSpeakingFrame):
            return [{"type": "user_stopped_speaking"}]
        if isinstance(frame, VADUserStartedSpeakingFrame):
            return [{"type": "vad_user_started_speaking"}]
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            return [{"type": "vad_user_stopped_speaking"}]
        if isinstance(frame, TranscriptionFrame):
            return [{"type": "user_transcription", "transcript": frame.text}]
        if isinstance(frame, LLMFullResponseStartFrame):
            self._awaiting_llm_restart = False
            self._text_buffer = []
            return [{"type": "llm_started"}]
        if isinstance(frame, LLMTextFrame):
            if self._awaiting_llm_restart:
                return []
            self._text_buffer.append(frame.text)
            return []
        if isinstance(frame, LLMFullResponseEndFrame):
            if self._awaiting_llm_restart:
                self._text_buffer = []
                return []
            return [self._segment_event("llm_response", "".join(self._text_buffer))]
        if isinstance(frame, TTSTextFrame):
            if self._scenario.bot_audio:
                return [self._segment_event("tts_response", frame.text)]
            return []
        if isinstance(frame, FunctionCallInProgressFrame):
            return [
                {
                    "type": "function_call",
                    "name": frame.function_name or None,
                    "args": dict(frame.arguments or {}),
                }
            ]
        return []

    def _segment_event(self, event_type: str, text: str) -> dict:
        """Build one response segment of ``event_type``.

        Used for ``llm_response`` (the LLM text) and ``tts_response`` (the TTS's
        spoken text). The text may be empty (e.g. an interrupted response); the
        matcher aggregates successive segments until the content check passes.
        """
        return {"type": event_type, "text": text}

    @staticmethod
    def _match_summary(event: dict) -> str:
        """A short human label for a matched event, for verbose progress.

        For ``function_call`` it's the call signature (``name(arg=value, ...)``);
        for everything else it's the event's text content (or empty).
        """
        if event.get("type") == "function_call":
            args = event.get("args") or {}
            sig = ", ".join(f"{k}={v}" for k, v in args.items())
            return f"{event.get('name') or '?'}({sig})"
        return _event_text(event)

    def _add_legacy_progress_callback(
        self, on_progress: Callable[[EvalTurnProgress], None]
    ) -> None:
        """Register a bare ``on_progress`` callback as an ``on_progress`` handler.

        The callback takes only the record, so it is wrapped to drop the session
        that event handlers receive as their first argument.
        """
        warnings.warn(
            "`on_progress` is deprecated since 1.9.0 and will be removed in 2.0.0. "
            "Use the `on_progress` event handler instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        self.add_event_handler("on_progress", lambda _session, record: on_progress(record))

    async def _progress(self, record: EvalTurnProgress) -> None:
        """Emit a progress record to the ``on_progress`` handlers."""
        await self._call_event_handler("on_progress", record)

    async def _run_turn(self, turn: EvalTurn, turn_idx: int) -> list[EvalAssertionFailure]:
        """Drive one turn: optionally honor send_after, send user input, match expectations.

        The user turn is sent as ``send-text`` (text mode) or, in audio mode, as
        chunked ``raw-audio`` messages that the bot's STT transcribes for real --
        the turn's ``audio:`` recording when it names one, otherwise its text
        synthesized by the user TTS.
        """
        failures: list[EvalAssertionFailure] = []
        # The turn's function calls match by name in any order; start each turn
        # with an empty buffer so a prior turn's calls can't carry over.
        self._pending_function_calls = []

        if turn.send_after is not None:
            try:
                await self._wait_send_after(turn.send_after)
            except TimeoutError as e:
                # Only the event-anchored wait can time out; the pure-delay form
                # just sleeps. So event is never None here, but fall back for typing.
                event_name = turn.send_after.event or "send_after"
                failures.append(
                    EvalAssertionFailure(
                        turn_index=turn_idx,
                        expectation_index=-1,
                        event_name=event_name,
                        reason=f"send_after never fired: {e}",
                        kind="send_after_timeout",
                    )
                )
                self._debug(f"FAIL: {event_name}: {failures[-1].reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, -1, event_name, "timeout", failures[-1].reason)
                )
                return failures

        # Register the turn's image (if any) before the user input, so the bot can
        # serve it when it requests a user image during the turn.
        if turn.image is not None:
            await self._send_image(turn.image)

        # Anything still queued belongs to an earlier turn: this turn's input hasn't
        # been sent, so the bot cannot have responded to it yet. Drop it, or an
        # expectation here can match — and a judge can rule on — output the bot
        # produced for a previous turn. The bot's own interruption events close this
        # window too, but only once the input reaches it, which is far too late when
        # `send_after` holds the send back for seconds.
        #
        # Before the send, not after: by the time the input has streamed, the bot has
        # begun reacting to it, and this turn's own `user_started_speaking` /
        # `bot_interrupted` would be dropped along with the stale output. Turns that
        # send nothing are observation-only and exist to match exactly this pending
        # output (a bot-first greeting), so they keep it.
        if turn.user is not None or turn.dtmf is not None:
            self._drop_pending_bot_output("before send")

        if turn.user is not None:
            how = turn.audio or ("audio" if self._speech is not None else "text")
            self._debug(f"send: {turn.user!r} ({how})")
            if turn.audio is not None:
                await self._send_audio_file(turn.audio)
            elif self._speech is not None:
                await self._send_user_audio(turn.user)
            else:
                await self._send_user_text(turn.user, self._scenario.bot_audio)
            # Record the user turn in the judge's conversation, so a later reply is
            # judged in context (e.g. a terse "That's four" answering this question).
            if self._judge is not None:
                self._judge.add_user_message(turn.user)
        elif turn.dtmf is not None:
            self._debug(f"send: dtmf {turn.dtmf!r}")
            await self._send_user_dtmf(turn.dtmf)
            # Record the keypresses for judge context, so the bot's reply is judged
            # knowing what was pressed.
            if self._judge is not None:
                self._judge.add_user_message(f"(DTMF keypad input: {turn.dtmf})")

        await self._progress(EvalTurnProgress(turn_idx, -1, turn.user or turn.dtmf or "", "turn"))

        # All of a turn's expectations share one deadline anchored at the send, so a
        # stalled turn fails within a single ``within_ms`` budget instead of spending
        # a fresh budget per expectation — e.g. a missing function call followed by a
        # missing response fails in 60s total, not 120s.
        anchor = time.monotonic()
        for exp_idx, expectation in enumerate(turn.expect):
            budget_ms = expectation.within_ms or self._default_timeout_ms

            try:
                failure = await self._match_and_verify(
                    expectation, anchor, budget_ms, turn_idx, exp_idx
                )
            except TimeoutError:
                reason = f"no matching {expectation.event!r} event arrived within {budget_ms}ms"
                failures.append(
                    EvalAssertionFailure(
                        turn_index=turn_idx,
                        expectation_index=exp_idx,
                        event_name=expectation.event,
                        reason=reason,
                        kind="timeout",
                    )
                )
                self._debug(f"FAIL: {expectation.event}: {reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "timeout", reason)
                )
                break

            if failure:
                failures.append(failure)
                self._debug(f"FAIL: {expectation.event}: {failure.reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "failed", failure.reason)
                )
            else:
                await self._progress(
                    EvalTurnProgress(
                        turn_idx, exp_idx, expectation.event, "matched", self._last_match_text
                    )
                )

        return failures

    async def _send_user_text(self, text: str, bot_audio: bool) -> None:
        """Send a text user turn via the RTVI ``send-text`` message.

        ``audio_response`` mirrors the scenario's ``bot_audio``: when False the
        LLM bypasses TTS for this turn (content-only evals).
        """
        message = RTVI.Message(
            type="send-text",
            id=self._message_id(),
            data=RTVI.SendTextData(
                content=text,
                options=RTVI.SendTextOptions(run_immediately=True, audio_response=bot_audio),
            ).model_dump(),
        )
        await self._send(message)

    async def _send_user_dtmf(self, keys: str) -> None:
        """Send a DTMF keypress turn as one RTVI ``dtmf`` message.

        The bot's ``RTVIProcessor`` turns each key into an ``InputDTMFFrame``
        pushed downstream, the same path a telephony transport's keypress takes.
        The bot's ``DTMFAggregator`` (if any) accumulates them and flushes — on
        the ``#`` terminator or its idle timeout — into a transcription the bot
        reacts to. Use ``send_after`` across turns to pace key sequences.
        """
        message = RTVI.Message(
            type="dtmf",
            id=self._message_id(),
            data={"buttons": list(keys)},
        )
        await self._send(message)

    async def _send_image(self, image_path: str) -> None:
        """Register an image (base64, with its MIME type) for the current turn.

        The eval transport serves it back as a ``UserImageRawFrame`` when the bot
        requests a user image. The file is sent as-is (already PNG/JPEG/...), so
        nothing is decoded or re-encoded.
        """
        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        self._debug(f"send: image {path.name} ({mime})")
        message = RTVI.Message(
            type="client-message",
            id=self._message_id(),
            data={"t": EVAL_IMAGE_MESSAGE_TYPE, "d": {"image": encoded, "format": mime}},
        )
        await self._send(message)

    async def _send_user_audio(self, text: str) -> None:
        """Render ``text`` to audio (cached) and send it to the bot.

        The whole utterance goes out as a few large ``raw-audio`` messages; the
        eval transport's virtual mic plays it into the bot's pipeline at
        real-time cadence (see ``pipecat.evals.transport.EvalMicrophone``).
        """
        assert self._speech is not None  # only called for audio-mode turns
        pcm, sample_rate = await self._speech.generate(text)
        for chunk in _audio_chunks(pcm, sample_rate):
            await self._send_raw_audio(chunk, sample_rate)

    async def _send_audio_file(self, path: str) -> None:
        """Play a turn's ``audio:`` recording to the bot in place of synthesizing it.

        The file's own sample rate travels with the audio, so a recording does
        not have to match the bot's input rate.
        """
        pcm, sample_rate = await load_user_audio(path)
        for chunk in _audio_chunks(pcm, sample_rate):
            await self._send_raw_audio(chunk, sample_rate)

    async def _send_raw_audio(self, chunk: bytes, sample_rate: int) -> None:
        """Send one PCM chunk to the bot as an RTVI ``raw-audio`` message."""
        message = RTVI.Message(
            type="raw-audio",
            id=self._message_id(),
            data={
                "base64Audio": base64.b64encode(chunk).decode("ascii"),
                "sampleRate": sample_rate,
                "numChannels": 1,
            },
        )
        await self._send(message)

    async def _wait_send_after(self, send_after: EvalSendAfter) -> None:
        """Block until ``send_after.event`` has been seen + ``delay_ms`` has elapsed.

        If the event was seen earlier in the run, anchor on that time (potentially
        fire immediately). Otherwise, poll the latest_event_times map until the
        event arrives, then anchor on that.

        With no event (``send_after.event is None``), it's a pure time delay:
        sleep ``delay_ms`` from now (i.e. from the previous turn's send).
        """
        target_delay_s = send_after.delay_ms / 1000.0

        if send_after.event is None:
            self._debug(f"send_after: waiting {send_after.delay_ms}ms")
            await asyncio.sleep(target_delay_s)
            return

        deadline = time.monotonic() + SEND_AFTER_MAX_WAIT_S
        self._debug(f"send_after: waiting for {send_after.event!r} + {send_after.delay_ms}ms")

        while True:
            seen_at = self._latest_event_times.get(send_after.event)
            if seen_at is not None:
                wait_s = max(0.0, (seen_at + target_delay_s) - time.monotonic())
                await asyncio.sleep(wait_s)
                return

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"event {send_after.event!r} not seen within "
                    f"{int(SEND_AFTER_MAX_WAIT_S * 1000)}ms"
                )

            await asyncio.sleep(SEND_AFTER_POLL_S)

    async def _match_and_verify(
        self,
        expectation: EvalExpectation,
        anchor: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Wait for the expected event and verify it. Returns a failure or None.

        Most events match a single event and are checked once. An ``llm_response``
        carrying a content check (``text_contains`` / ``eval:``) instead
        *aggregates*: it accumulates the text of successive response segments and
        re-checks on each new segment until the check passes, the judge
        affirmatively rejects, or the ``within_ms`` budget expires. A
        ``user_transcription`` with ``text_contains`` aggregates the same way,
        since an STT may finalize one utterance in several pieces.

        Output that predates this turn (the greeting, or a turn that was
        interrupted) isn't specially filtered: in audio mode the reader already
        drops it from the queue when ``user-started-speaking`` fires, and anything
        that slips through (e.g. a text-mode greeting) is harmless — the judge
        returns "continue" until the turn's real answer is aggregated in.

        Raises:
            TimeoutError: when no matching event arrives at all (so the caller can
                report "no matching event arrived"). A response that arrives but
                never satisfies the content check returns a failure instead.
        """
        deadline = anchor + (budget_ms / 1000.0)
        self._last_match_text = ""

        if expectation.absent:
            return await self._match_absent(expectation, deadline, budget_ms, turn_idx, exp_idx)

        aggregates = (
            expectation.event in ("response", "llm_response", "tts_response")
            and (expectation.text_contains is not None or expectation.eval is not None)
        ) or (
            expectation.event == "user_transcription"
            and expectation.text_contains is not None
            and expectation.eval is None
        )
        if not aggregates:
            if expectation.event in FUNCTION_CALL_EVENTS:
                # A call expectation holds the set of calls the turn should make;
                # it completes only when all are found, in any order (a response
                # arriving doesn't short-circuit it).
                return await self._match_function_calls(expectation, deadline, turn_idx, exp_idx)
            self._debug(f"match: waiting for {expectation.event!r}")
            event = await self._next_matching_event(expectation.event, deadline)
            payload_failure = self._check_payload(event, expectation, turn_idx, exp_idx)
            if payload_failure:
                return payload_failure
            judge_failure = await self._check_judge(event, expectation, turn_idx, exp_idx)
            if judge_failure is None:
                self._last_match_text = self._match_summary(event)
            return judge_failure

        def fail(reason: str, kind: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(turn_idx, exp_idx, expectation.event, reason, kind)

        if expectation.eval is not None and self._judge is None:
            return fail("scenario uses 'eval:' but no judge could be built", "no_judge")

        check = "+".join(
            name
            for name, val in (
                ("text_contains", expectation.text_contains),
                ("eval", expectation.eval),
            )
            if val is not None
        )
        self._debug(f"match: waiting for {expectation.event!r} ({check})")
        aggregate = ""
        last_reason = ""
        seen_any = False
        while True:
            try:
                event = await self._next_matching_event(expectation.event, deadline)
            except TimeoutError:
                if not seen_any:
                    raise  # no response at all → caller logs "no matching event arrived"
                self._debug(f"eval: timeout, not satisfied: {last_reason}")
                # Without `eval:` the only way to be unsatisfied is a missing
                # substring: `text_contains` is monotonic, so it holds out for more
                # text rather than failing outright.
                return fail(
                    f"not satisfied within {budget_ms}ms: {last_reason}",
                    "judge_continue" if expectation.eval is not None else "text_mismatch",
                )

            seen_any = True
            delta = _event_text(event)
            aggregate += delta
            # Feed each segment to the judge as its own assistant message, so it
            # judges the bot's reply in the conversation's context (the cumulative
            # `aggregate` is kept only for text_contains and the match summary).
            if expectation.eval is not None and self._judge is not None:
                self._judge.add_assistant_message(delta)
            status, reason = await self._evaluate_aggregate(aggregate, expectation)
            self._debug(f"eval: {status} (aggregate={aggregate.strip()!r}) {reason}")
            if status == "pass":
                self._last_match_text = aggregate
                return None
            if status == "fail":
                # Only the judge can affirmatively fail an aggregate.
                return fail(reason, "judge_no")
            # "continue": wait for the next segment, separated by a space so
            # sentences don't run together (e.g. "...that. The weather...").
            aggregate += " "
            last_reason = reason

    async def _match_absent(
        self,
        expectation: EvalExpectation,
        deadline: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Inverted match: pass when NO event of this type arrives before the deadline.

        The budget is the whole point here — the expectation holds the turn open
        for ``within_ms`` and succeeds only if the event type stays absent for
        that entire window. An arriving event fails immediately with its content
        in the reason, so a duplicate-output regression shows what the bot said.
        """
        self._debug(f"match: expecting NO {expectation.event!r} for {budget_ms}ms")
        try:
            event = await self._next_matching_event(expectation.event, deadline)
        except TimeoutError:
            # The quiet window held: absence confirmed.
            self._last_match_text = f"no {expectation.event!r} for {budget_ms}ms"
            return None
        return EvalAssertionFailure(
            turn_index=turn_idx,
            expectation_index=exp_idx,
            event_name=expectation.event,
            reason=(
                f"expected no {expectation.event!r} within {budget_ms}ms, "
                f"but one arrived: {self._match_summary(event)}"
            ),
            kind="unexpected_event",
        )

    async def _next_matching_event(self, event_type: str, deadline: float) -> dict:
        """Pop events from the queue until one of ``event_type`` arrives.

        Events that don't match are dropped (so a scenario doesn't have to
        enumerate every event the bot emits). They remain in ``events_seen`` and
        ``latest_event_times`` for diagnostics and send_after lookups. Raises
        TimeoutError once ``deadline`` passes.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()

            async with asyncio.timeout(remaining):
                event = await self._queue.get()

            if event.get("type") == event_type:
                return event

    async def _match_function_calls(
        self,
        expectation: EvalExpectation,
        deadline: float,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Match every call in a ``function_call`` expectation, in any order.

        Iterates the expectation's ``calls`` (each a name + optional args), claiming
        a matching call for each from the turn's calls (buffered + still arriving).
        Passes only when all are claimed within the budget; otherwise returns a
        failure naming the call that was missing or whose args didn't match.
        """

        def fail(reason: str, kind: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(turn_idx, exp_idx, expectation.event, reason, kind)

        def spec_sig(spec) -> str:
            name = spec.name or "any function"
            if not spec.args:
                return name
            args = ", ".join(f"{k}={v!r}" for k, v in spec.args.items())
            return f"{name}({args})"

        matched: list[str] = []
        for spec in expectation.calls or []:
            want = spec.args or None
            self._debug(f"match: waiting for {expectation.event!r} ({spec_sig(spec)})")
            try:
                event = await self._next_function_call(spec.name, deadline, want, expectation.event)
            except TimeoutError:
                # A call of the right name with the wrong arguments is a different
                # failure from the call never being made, and the bot's arguments
                # are what the reader needs to see.
                near = [
                    ev.get("args")
                    for ev in self._pending_function_calls
                    if ev.get("type") == expectation.event
                    and (spec.name is None or ev.get("name") == spec.name)
                ]
                if want is not None and near:
                    return fail(
                        f"no {spec.name!r} call had args {want!r} (saw {near!r})",
                        "function_args_mismatch",
                    )
                missing = spec.name or "any function"
                seen = ", ".join(matched) if matched else "none"
                return fail(
                    f"function call {missing!r} not seen (matched: {seen})",
                    "missing_function_call",
                )
            matched.append(str(event.get("name")))

        self._last_match_text = ", ".join(matched) or "function call"
        return None

    async def _next_function_call(
        self,
        name: str | None,
        deadline: float,
        args: dict | None = None,
        event_type: str = "function_call",
    ) -> dict:
        """Return a call event of ``event_type`` matching ``name`` (``None`` = any) and ``args``.

        A turn's function calls can arrive in any order, so match against the
        per-turn buffer of calls seen but not yet claimed, plus newly arriving
        ones; a call that doesn't match is buffered so another expected call can
        claim it. ``args`` is a subset check and participates in matching, so the
        turn is satisfied by any call matching both name and arguments rather
        than by the first to share the name — which is what lets a call the LLM
        corrects and repeats satisfy it. Other event types are dropped, as in
        :meth:`_next_matching_event`. Raises TimeoutError once ``deadline`` passes.
        """

        def matches(ev: dict) -> bool:
            if ev.get("type") != event_type:
                return False
            if name is not None and ev.get("name") != name:
                return False
            if args is None:
                return True
            actual = ev.get("args") or {}
            return all(actual.get(k) == v for k, v in args.items())

        for i, ev in enumerate(self._pending_function_calls):
            if matches(ev):
                return self._pending_function_calls.pop(i)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()

            async with asyncio.timeout(remaining):
                event = await self._queue.get()

            if event.get("type") not in FUNCTION_CALL_EVENTS:
                continue
            if matches(event):
                return event
            self._pending_function_calls.append(event)

    async def _evaluate_aggregate(
        self, aggregate: str, expectation: EvalExpectation
    ) -> tuple[str, str]:
        """Evaluate the accumulated response text. Returns ``(status, reason)``.

        ``status`` is ``"pass"``, ``"fail"``, or ``"continue"``. ``text_contains``
        is monotonic, so a missing substring is ``"continue"`` (more text may
        arrive); only the judge can affirmatively ``"fail"``.
        """
        if expectation.text_contains is not None and not _text_contains(
            aggregate, expectation.text_contains
        ):
            return ("continue", f"does not contain {expectation.text_contains!r}")

        if expectation.eval is not None:
            if not aggregate.strip():
                return ("continue", "no response text yet")
            # _match_and_verify guarantees a judge exists before aggregating eval:.
            assert self._judge is not None
            with logger.contextualize(eval_pipeline="judge"):
                # The reply segments were added to the judge's conversation in the
                # aggregation loop; the judge evaluates that context, not `aggregate`.
                verdict = await self._judge.evaluate(expectation.eval)
            if verdict.verdict == "no":
                return ("fail", f"judge said no: {verdict.reason}")
            if verdict.verdict == "continue":
                return ("continue", f"judge said continue: {verdict.reason}")
            return ("pass", f"judge said yes: {verdict.reason}")

        return ("pass", "")

    @staticmethod
    def _check_payload(
        event: dict,
        expectation: EvalExpectation,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Apply payload-level checks to a matched event. Returns the first failure or None."""

        def fail(reason: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=reason,
                kind="text_mismatch",
            )

        if expectation.text_contains is not None:
            content = _event_text(event)
            if not _text_contains(content, expectation.text_contains):
                return fail(f"text {content!r} does not contain {expectation.text_contains!r}")

        return None

    async def _check_judge(
        self,
        event: dict,
        expectation: EvalExpectation,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Run the judge assertion if ``eval:`` was set on this expectation."""
        if expectation.eval is None:
            return None

        if self._judge is None:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason="scenario uses 'eval:' but no judge could be built",
                kind="no_judge",
            )

        content = event.get("text") or event.get("transcript")
        if not content:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"event has no text/transcript to judge: {event!r}",
                kind="no_content",
            )

        self._judge.add_assistant_message(content)
        verdict = await self._judge.evaluate(expectation.eval)
        if not verdict.passed:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"eval {expectation.eval!r}: judge said no — {verdict.reason}",
                kind="judge_no",
            )

        return None


def _event_text(event: dict) -> str:
    """The text an event carries: reply events use ``text``, ``user_transcription`` ``transcript``."""
    return event.get("text") or event.get("transcript") or ""


def _text_contains(content: str, needle: str) -> bool:
    """Whether ``needle`` occurs in ``content``, ignoring how either is spaced.

    Aggregated text joins segments with spaces and an STT's pieces may carry
    their own, so a phrase is matched on collapsed whitespace.
    """
    return " ".join(needle.split()) in " ".join(content.split())


def _audio_chunks(pcm: bytes, sample_rate: int):
    """Yield ``pcm`` as ~1s slices (16-bit mono), staying well under websocket limits.

    A websocket server's default max message size is 1MiB; one second of 16kHz
    mono is ~43KB base64-encoded, so even long utterances ship in a handful of
    messages.
    """
    bytes_per_chunk = (sample_rate * SEND_CHUNK_MS // 1000) * 2
    for offset in range(0, len(pcm), bytes_per_chunk):
        yield pcm[offset : offset + bytes_per_chunk]
