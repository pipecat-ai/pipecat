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
==========================      ==============================================

Matching semantics: expected events must appear in the specified order, but
unmatched events may appear between them (so a scenario doesn't have to
enumerate every event the bot emits). The ``within_ms`` budget for each
expectation is measured from the most recent ``send-text`` / ``raw-audio`` / ``dtmf`` send
(default 60s when omitted).

An ``llm_response`` with a content check (``text_contains`` / ``eval:``)
aggregates: the harness accumulates the text of successive response segments
within the turn and re-checks on each one, so an interim filler ("Let me check
on that.") or the on-connect greeting is rolled past rather than mistaken for
the turn's answer. Responses that began before the turn's input are skipped, so
an interrupted prior turn doesn't bleed in. The judge returns yes / no /
continue; ``text_contains`` treats a missing substring as continue. The
``within_ms`` budget bounds the wait.

Example::

    scenario = EvalScenario.load("scenarios/greeting.yaml")
    result = await EvalSession.from_scenario(scenario, "ws://localhost:7860").run()
    if result.passed:
        print("PASS")
    else:
        for f in result.failures:
            print(f"  {f}")
"""

import asyncio
import base64
import json
import mimetypes
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from loguru import logger
from websockets.asyncio.client import ClientConnection

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.judge import EvalJudge
from pipecat.evals.scenario import (
    EvalExpectation,
    EvalScenario,
    EvalSendAfter,
    EvalTurn,
    describe_config,
)
from pipecat.evals.serializer import (
    EVAL_BOT_AUDIO_TYPE,
    EVAL_CANCEL_MESSAGE_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
)
from pipecat.evals.speech import EvalSpeech
from pipecat.evals.transcribe import EvalTranscriber

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


@dataclass
class EvalAssertionFailure:
    """A single failed assertion within an eval.

    Parameters:
        turn_index: Index of the turn that failed.
        expectation_index: Index of the expectation within the turn, or -1 for a
            turn-level failure (e.g. a ``send_after`` that never fired).
        event_name: The expectation's event name.
        reason: Human-readable explanation of the failure.
    """

    turn_index: int
    expectation_index: int
    event_name: str
    reason: str

    def __str__(self) -> str:
        return (
            f"turn {self.turn_index} expectation {self.expectation_index} "
            f"({self.event_name}): {self.reason}"
        )


@dataclass
class EvalResult:
    """Outcome of running a scenario in an :class:`EvalSession`.

    Parameters:
        scenario_name: Name of the scenario that was run.
        passed: Whether every assertion passed.
        failures: The assertions that failed, in order.
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


class EvalSession:
    """Runs one :class:`EvalScenario` against a bot over a single WebSocket session.

    Connects as an RTVI client, drives each turn (sending ``send-text``,
    ``raw-audio``, or ``dtmf``), collects the RTVI events the bot emits, and asserts on them.
    Build one with :meth:`from_scenario` (which constructs the judge, speech, and
    transcriber the scenario needs), then await :meth:`run`.
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
        self._scenario = scenario
        self._bot_url = bot_url
        self._connect_timeout_s = connect_timeout_s
        self._default_timeout_ms = default_timeout_ms
        self._on_progress = on_progress
        self._record_path = record_path
        self._stop_bot = stop_bot
        # Either the run-wide CLI flag or the scenario's own field opts in.
        self._trigger_disconnect = trigger_disconnect or scenario.trigger_disconnect

        self._ws: ClientConnection | None = None
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
                ``scenario.user_audio`` in audio mode).
            transcriber: Override the bot-audio transcriber (default: built from
                ``scenario.transcriber`` when the scenario asserts ``response``).

        Returns:
            A configured session, ready for :meth:`run`.
        """
        turns = scenario.turns
        if judge is None and any(exp.eval is not None for turn in turns for exp in turn.expect):
            with logger.contextualize(eval_pipeline="judge"):
                judge = EvalJudge.from_config(scenario.judge)

        if speech is None and scenario.user_audio is not None:
            with logger.contextualize(eval_pipeline="speech"):
                speech = EvalSpeech.from_config(
                    scenario.user_audio, cache_dir=cache_dir, use_cache=use_cache
                )

        wants_response = any(exp.event == "response" for turn in turns for exp in turn.expect)
        if transcriber is None and wants_response and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                transcriber = EvalTranscriber.from_config(scenario.transcriber)

        return cls(
            scenario,
            bot_url,
            connect_timeout_s=connect_timeout_s,
            default_timeout_ms=default_timeout_ms,
            on_progress=on_progress,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            judge=judge,
            speech=speech,
            transcriber=transcriber,
        )

    async def run(self) -> EvalResult:
        """Connect, drive the scenario, and return the result."""
        started = time.monotonic()
        self._debug_t0 = started
        self._debug(f"run: scenario {self._scenario.name!r} -> {self._bot_url}")
        # Record which speech / transcription / judge services and models were used,
        # so a saved eval.log is self-describing (no need to cross-reference config).
        for line in describe_config(self._scenario).splitlines():
            self._debug(line)

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
                skipped=reason,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        # Retry the connect until the bot accepts it or we time out. Attempts
        # before the bot is listening fail with ConnectionRefused (no WebSocket
        # handshake, so no server-side error) — this doubles as readiness waiting,
        # which is why callers (e.g. the release orchestrator) can launch the bot
        # and connect straight away without a separate, handshake-noisy port probe.
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
                    )
                ],
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        failures: list[EvalAssertionFailure] = []
        reader_task: asyncio.Task | None = None
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

            reader_task = asyncio.create_task(self._reader_loop())

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
                    )
                )
            else:
                for turn_idx, turn in enumerate(self._scenario.turns):
                    self._current_turn = turn_idx
                    self._debug(f"--- turn {turn_idx}: {turn.user!r}")
                    turn_failures = await self._run_turn(turn, turn_idx)
                    failures.extend(turn_failures)
                    if turn_failures:
                        # Fail fast: a failed turn leaves the conversation in an
                        # unknown state, so running the rest just burns another
                        # timeout per turn (e.g. a broken greeting turn shouldn't
                        # cost the full budget here and again on the question).
                        self._debug(f"turn {turn_idx} failed; stopping scenario (fail-fast)")
                        break
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
            failures.append(
                EvalAssertionFailure(
                    turn_index=self._current_turn,
                    expectation_index=-1,
                    event_name="<error>",
                    reason=f"{type(e).__name__}: {e}",
                )
            )
        finally:
            if reader_task is not None:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Tear each sub-pipeline down under the same eval_pipeline label as its
            # setup, so its shutdown logs (e.g. "Cancelling pipeline worker") are
            # attributed to it rather than leaking into the harness catch-all.
            if self._speech is not None:
                with logger.contextualize(eval_pipeline="speech"):
                    await self._speech.aclose()
            if self._transcriber is not None:
                with logger.contextualize(eval_pipeline="transcription"):
                    await self._transcriber.aclose()
            # Optionally ask the bot to tear its pipeline down gracefully (closing
            # its STT/TTS/LLM connections) so the process exits on its own. Skipped
            # by default so the bot stays up for more scenarios; the eval transport
            # survives the disconnect either way (best-effort; the suite still has a
            # kill fallback).
            if self._stop_bot:
                await self._send_cancel()
            await self._ws.close()

        self._debug(f"done: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures,
            failures=failures,
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
        virtual mic for audio-mode scenarios; without it the transport plays no
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
        if self._speech is not None:
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
                if exp.event != "function_call":
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
        ready = RTVI.Message(
            type="client-ready",
            id=self._message_id(),
            data=RTVI.ClientReadyData(
                version=RTVI.PROTOCOL_VERSION,
                about=RTVI.AboutClientData(library="pipecat-evals"),
            ).model_dump(),
        )
        await self._send(ready)

        # Hard gate — raises TimeoutError if the bot never announces readiness.
        await self._wait_for_event("bot_ready", BOT_READY_TIMEOUT_S)

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
        """Serialize and send an RTVI message over the WebSocket."""
        assert self._ws is not None  # connected before any send
        await self._ws.send(message.model_dump_json())

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

    async def _reader_loop(self) -> None:
        """Drain the WS, translate RTVI messages to friendly events, enqueue them."""
        assert self._ws is not None  # started only after a successful connect
        try:
            async for raw in self._ws:
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning(f"Eval session: dropping non-JSON message: {raw!r}")
                    continue
                if message.get("label") != RTVI.MESSAGE_LABEL:
                    continue
                if self._wants_response:
                    await self._handle_tts_audio(message)
                for event in self._translate(message):
                    await self._enqueue(event)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
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

    def _discard_interrupted_output(self) -> None:
        """Drop the bot's interrupted, un-matched output (on user interruption).

        Clears the response buffers and drains the bot's pending output from the
        event queue, so a greeting (or any prior bot output) the user just
        interrupted can't be matched against this turn. ``user_transcription`` is
        preserved: a DTMF keypress emits its transcription immediately before the
        turn-start interruption, and that transcription is the turn's *input*, not
        the stale bot output this discard is meant to clear — dropping it would
        race the matcher. Diagnostics (``events_seen``, ``latest_event_times``)
        are left intact for send_after lookups.
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
            self._debug(f"discard: dropped {dropped} queued event(s) on interruption")

    async def _handle_tts_audio(self, message: dict) -> None:
        """Accumulate the bot's audio and emit a ``response`` per spoken turn.

        We bound on the speaking boundaries (``bot-started-speaking`` /
        ``bot-stopped-speaking``), not the TTS ones: the output transport delays
        audio by PTS to play it out, so ``bot-tts-stopped`` fires while the tail
        is still streaming. ``bot-stopped-speaking`` fires once the audio has
        actually finished — only then is the buffer complete. The transcription is
        enqueued as a ``response`` event, aggregated like ``llm_response``.
        """
        msg_type = message.get("type")
        if msg_type == EVAL_BOT_AUDIO_TYPE:
            data = message.get("data") or {}
            self._tts_audio.extend(base64.b64decode(data.get("audio", "")))
            self._tts_sample_rate = int(data.get("sampleRate", 0)) or self._tts_sample_rate
        elif msg_type == "bot-started-speaking":
            self._debug(f"bot-started-speaking: discarding {len(self._tts_audio)}B buffered")
            self._tts_audio = bytearray()
        elif msg_type == "bot-stopped-speaking":
            self._debug(
                f"bot-stopped-speaking: {len(self._tts_audio)}B @ {self._tts_sample_rate}Hz, "
                f"transcriber={self._transcriber is not None}"
            )
            if not self._tts_audio or self._transcriber is None:
                return
            pcm, sample_rate = bytes(self._tts_audio), self._tts_sample_rate
            self._tts_audio = bytearray()
            text = await self._transcriber.transcribe(pcm, sample_rate)
            self._debug(f"  transcribed: {len(pcm)}B -> {text!r}")
            await self._enqueue({"type": "response", "text": text})

    def _translate(self, message: dict) -> list[dict]:
        """Translate one RTVI server message into zero or more friendly events."""
        msg_type = message.get("type")
        data = message.get("data") or {}

        match msg_type:
            case "bot-ready":
                return [{"type": "bot_ready"}]
            case "user-started-speaking":
                # A new user turn in audio mode. Drop any leftover bot output from
                # a prior turn so it isn't aggregated into this one.
                self._discard_interrupted_output()
                self._awaiting_llm_restart = True
                return [{"type": "user_started_speaking"}]
            case "bot-interrupted":
                # The bot's in-flight output was cut off — a VAD barge-in or a
                # run_immediately text interrupt. Drop it so only what the bot says
                # *after* the interruption is matched. Service-independent, the same
                # path for both modalities, and no timestamps.
                self._discard_interrupted_output()
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
            case _:
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
        return event.get("text") or event.get("transcript") or ""

    def _progress(self, record: EvalTurnProgress) -> None:
        """Emit a progress record to the on_progress callback, if one was given."""
        if self._on_progress is not None:
            self._on_progress(record)

    async def _run_turn(self, turn: EvalTurn, turn_idx: int) -> list[EvalAssertionFailure]:
        """Drive one turn: optionally honor send_after, send user input, match expectations.

        The user turn is sent as ``send-text`` (text mode) or, when the scenario
        provides a ``user_audio`` block, as chunked ``raw-audio`` messages that
        the bot's STT transcribes for real.
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
                    )
                )
                self._debug(f"FAIL: {event_name}: {failures[-1].reason}")
                self._progress(
                    EvalTurnProgress(turn_idx, -1, event_name, "timeout", failures[-1].reason)
                )
                return failures

        # Register the turn's image (if any) before the user input, so the bot can
        # serve it when it requests a user image during the turn.
        if turn.image is not None:
            await self._send_image(turn.image)

        if turn.user is not None:
            self._debug(f"send: {turn.user!r} ({'audio' if self._speech is not None else 'text'})")
            if self._speech is not None:
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

        self._progress(EvalTurnProgress(turn_idx, -1, turn.user or turn.dtmf or "", "turn"))

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
                    )
                )
                self._debug(f"FAIL: {expectation.event}: {reason}")
                self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "timeout", reason)
                )
                break

            if failure:
                failures.append(failure)
                self._debug(f"FAIL: {expectation.event}: {failure.reason}")
                self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "failed", failure.reason)
                )
            else:
                self._progress(
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
        """Send a DTMF keypress turn: one RTVI ``dtmf`` message per key.

        The bot's ``RTVIProcessor`` turns each into an ``InputDTMFFrame`` pushed
        downstream, the same path a telephony transport's keypress takes. The
        bot's ``DTMFAggregator`` (if any) accumulates them and flushes — on the
        ``#`` terminator or its idle timeout — into a transcription the bot reacts
        to. Keys go out back-to-back; use ``send_after`` across turns to pace them.
        """
        for key in keys:
            message = RTVI.Message(
                type="dtmf",
                id=self._message_id(),
                data={"button": key},
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

    async def _wait_for_event(self, event_name: str, timeout_s: float) -> None:
        """Block until ``event_name`` has been seen, or raise TimeoutError."""
        deadline = time.monotonic() + timeout_s
        while event_name not in self._latest_event_times:
            if time.monotonic() >= deadline:
                raise TimeoutError(event_name)
            await asyncio.sleep(SEND_AFTER_POLL_S)

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
        affirmatively rejects, or the ``within_ms`` budget expires.

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

        aggregates = expectation.event in ("response", "llm_response", "tts_response") and (
            expectation.text_contains is not None or expectation.eval is not None
        )
        if not aggregates:
            if expectation.event == "function_call":
                # A function_call expectation holds the set of calls the turn should
                # make; it completes only when all are found, in any order (a
                # response arriving doesn't short-circuit it).
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

        def fail(reason: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(turn_idx, exp_idx, expectation.event, reason)

        if expectation.eval is not None and self._judge is None:
            return fail("scenario uses 'eval:' but no judge could be built")

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
                return fail(f"not satisfied within {budget_ms}ms: {last_reason}")

            seen_any = True
            delta = event.get("text", "")
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
                return fail(reason)
            # "continue": wait for the next segment, separated by a space so
            # sentences don't run together (e.g. "...that. The weather...").
            aggregate += " "
            last_reason = reason

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

        def fail(reason: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(turn_idx, exp_idx, expectation.event, reason)

        def spec_sig(spec) -> str:
            name = spec.name or "any function"
            if not spec.args:
                return name
            args = ", ".join(f"{k}={v!r}" for k, v in spec.args.items())
            return f"{name}({args})"

        matched: list[str] = []
        for spec in expectation.calls or []:
            self._debug(f"match: waiting for {expectation.event!r} ({spec_sig(spec)})")
            try:
                event = await self._next_function_call(spec.name, deadline)
            except TimeoutError:
                want = spec.name or "any function"
                seen = ", ".join(matched) if matched else "none"
                return fail(f"function call {want!r} not seen (matched: {seen})")
            if spec.args:
                actual = event.get("args") or {}
                missing = {k: v for k, v in spec.args.items() if actual.get(k) != v}
                if missing:
                    return fail(
                        f"call {event.get('name')!r} args {actual!r} missing expected {missing!r}"
                    )
            matched.append(str(event.get("name")))

        self._last_match_text = ", ".join(matched) or "function call"
        return None

    async def _next_function_call(self, name: str | None, deadline: float) -> dict:
        """Return a ``function_call`` event with the given ``name`` (``None`` = any).

        A turn's function calls can arrive in any order, so match against the
        per-turn buffer of calls seen but not yet claimed, plus newly arriving
        ones; a call with a different name is buffered so another expected call can
        claim it. Other event types are dropped, as in :meth:`_next_matching_event`.
        Raises TimeoutError once ``deadline`` passes.
        """

        def matches(ev: dict) -> bool:
            return name is None or ev.get("name") == name

        for i, ev in enumerate(self._pending_function_calls):
            if matches(ev):
                return self._pending_function_calls.pop(i)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()

            async with asyncio.timeout(remaining):
                event = await self._queue.get()

            if event.get("type") != "function_call":
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
        if expectation.text_contains is not None and expectation.text_contains not in aggregate:
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
            )

        if expectation.text_contains is not None:
            # Resolve the event's text content: llm_response carries "text",
            # user_transcription carries "transcript".
            content = event.get("text") or event.get("transcript") or ""
            if expectation.text_contains not in content:
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
            )

        content = event.get("text") or event.get("transcript")
        if not content:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"event has no text/transcript to judge: {event!r}",
            )

        self._judge.add_assistant_message(content)
        verdict = await self._judge.evaluate(expectation.eval)
        if not verdict.passed:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"eval {expectation.eval!r}: judge said no — {verdict.reason}",
            )

        return None


def _audio_chunks(pcm: bytes, sample_rate: int):
    """Yield ``pcm`` as ~1s slices (16-bit mono), staying well under websocket limits.

    A websocket server's default max message size is 1MiB; one second of 16kHz
    mono is ~43KB base64-encoded, so even long utterances ship in a handful of
    messages.
    """
    bytes_per_chunk = (sample_rate * SEND_CHUNK_MS // 1000) * 2
    for offset in range(0, len(pcm), bytes_per_chunk):
        yield pcm[offset : offset + bytes_per_chunk]
