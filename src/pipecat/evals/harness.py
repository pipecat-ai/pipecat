#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session: drives a bot over RTVI and asserts on the events it emits.

An :class:`EvalSession` connects to a running bot's eval transport (a
``WebsocketServerTransport`` speaking RTVI via
:class:`~pipecat.evals.serializer.RTVIEvalSerializer`), walks through a parsed
:class:`~pipecat.evals.scenario.Scenario`, and verifies that the expected
semantic events arrive in order, with the right payloads, within their latency
budgets. It returns an :class:`EvalResult`.

The session is a thin RTVI client. It builds outgoing messages with the RTVI
models (:mod:`pipecat.processors.frameworks.rtvi.models`) and translates the
RTVI server messages it receives back into a small set of friendly event names
the scenario files assert on:

==========================  ==============================================
scenario ``event:``         RTVI server message(s)
==========================  ==============================================
``user_started_speaking``   ``user-started-speaking``
``user_stopped_speaking``   ``user-stopped-speaking``
``user_transcription``      ``user-transcription`` (final only)
``llm_started``             ``bot-llm-started``
``llm_response``            the LLM text: ``bot-llm-text`` joined at ``bot-llm-stopped``
``tts_response``            the TTS's spoken text: one segment per ``bot-tts-text``
                            (audio modality only)
``response``                local-Whisper transcription of the bot's actual audio
                            (audio modality only); ``llm_response`` in text modality
``function_call``           ``llm-function-call-in-progress``
==========================  ==============================================

Matching semantics: expected events must appear in the specified order, but
unmatched events may appear between them (so a scenario doesn't have to
enumerate every event the bot emits). The ``within_ms`` budget for each
expectation is measured from the most recent ``send-text`` / ``raw-audio`` send
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

    scenario = load_scenario("scenarios/greeting.yaml")
    result = await run_scenario(scenario, "ws://localhost:7860")
    if result.passed:
        print("PASS")
    else:
        for f in result.failures:
            print(f"  {f}")
"""

import asyncio
import base64
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.judge import Judge, build_default_judge
from pipecat.evals.scenario import Expectation, Scenario, SendAfter, Turn
from pipecat.evals.serializer import (
    EVAL_BOT_AUDIO_TYPE,
    EVAL_CANCEL_MESSAGE_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_RESET_MESSAGE_TYPE,
)

# ``websockets`` is imported lazily inside the methods that use it. That keeps
# this module importable (and the CLI plugin loadable via pipecat-cli) even
# when the optional ``websockets-base`` extra isn't installed — users who
# actually run an eval get a clear ImportError at that point.

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000
SEND_AFTER_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01
BOT_READY_TIMEOUT_S = 10.0

# Audio injection: stream the user's audio as ~20ms PCM slices at real-time mic
# cadence (one frame every ~20ms), padded with ~500ms of leading/trailing silence.
# The bot's VAD/STT endpoint on wall-clock timing, so bursting the whole utterance
# compresses the speech/silence boundary and mis-segments it — _send_user_audio
# paces the sends, which is what makes a short trailing pad enough for the VAD to
# detect end-of-turn. Padding is added at send time (not baked into the cached
# audio), so it can change without invalidating the cache.
AUDIO_CHUNK_MS = 20
AUDIO_LEADING_SILENCE_MS = 500
AUDIO_TRAILING_SILENCE_MS = 500


@dataclass
class AssertionFailure:
    """A single failed assertion within an eval."""

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
    """Outcome of running a scenario in an :class:`EvalSession`."""

    scenario_name: str
    passed: bool
    failures: list[AssertionFailure] = field(default_factory=list)
    duration_ms: int = 0
    events_seen: list[dict] = field(default_factory=list)
    # Timestamped trace of the harness's own decisions (events received, audio
    # transcribed, matcher progress), for diagnosing flaky runs. Saved per-scenario
    # by the orchestrator alongside the bot log.
    debug_log: list[str] = field(default_factory=list)
    # When set, the scenario was not run (e.g. tts_response without audio mode);
    # the string is the reason. Neither passed nor failed.
    skipped: str | None = None


@dataclass
class TurnProgress:
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
    """Runs one :class:`Scenario` against a bot over a single WebSocket session.

    Connects as an RTVI client, drives each turn (sending ``send-text`` or
    ``raw-audio``), collects the RTVI events the bot emits, and asserts on them.
    Use :meth:`run`, or the :func:`run_scenario` convenience wrapper.
    """

    def __init__(
        self,
        scenario: Scenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        on_progress: Callable[[TurnProgress], None] | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
    ):
        """Initialize the eval session.

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            on_progress: Optional callback invoked with a :class:`TurnProgress`
                as each turn and expectation resolves (used for verbose output).
            record_path: When set (and the scenario is audio mode), asks the eval
                transport to record the conversation audio to this path (bot-side).
            cache_dir: Where to cache synthesized user audio. Defaults to
                ``<user-cache-dir>/pipecat/tts``.
        """
        self._scenario = scenario
        self._bot_url = bot_url
        self._connect_timeout_s = connect_timeout_s
        self._on_progress = on_progress
        self._record_path = record_path
        self._cache_dir = cache_dir

        self._ws: Any = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._latest_event_times: dict[str, float] = {}
        self._events_seen: list[dict] = []
        # Timestamped trace of the harness's own decisions, for diagnosing flakes.
        self._debug_log: list[str] = []
        self._debug_t0: float = 0.0
        self._current_turn: int = -1
        self._next_id = 0
        self._judge: Judge | None = None

        # One persistent TTS pipeline reused across the scenario's audio turns
        # (created in run() only when the scenario uses user_audio).
        self._voice: Any = None

        # Accumulates the bot's output text for the current response, to
        # synthesize llm_response. Source depends on the mode: bot-llm-text in
        # text mode (skip-TTS), bot-tts-text in audio mode (what was spoken).
        self._text_buffer: list[str] = []

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
        self._transcriber: Any = None
        self._tts_audio: bytearray = bytearray()  # current spoken segment's audio
        self._tts_sample_rate: int = 0

    async def run(self) -> EvalResult:
        """Connect, drive the scenario, and return the result."""
        import websockets  # lazy: see note at the top of the module

        started = time.monotonic()
        self._debug_t0 = started
        self._debug(f"run: scenario {self._scenario.name!r} -> {self._bot_url}")

        # The `response` transcription needs the bot's actual audio; without audio
        # mode there's nothing to transcribe, so skip rather than fail. (Normally
        # unreachable: load_scenario resolves `response` to llm_response in text
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
                    AssertionFailure(
                        turn_index=-1,
                        expectation_index=-1,
                        event_name="<connect>",
                        reason=f"failed to connect to {self._bot_url}: {e.__class__.__name__}",
                    )
                ],
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        # Lazily construct the judge — only if the scenario uses eval: assertions.
        if any(exp.eval is not None for turn in self._scenario.turns for exp in turn.expect):
            self._judge = build_default_judge(self._scenario.judge)

        # One TTS pipeline for the whole scenario's audio turns.
        if self._scenario.user_audio is not None:
            from pipecat.evals.voice import (
                EvalVoice,
                build_tts_service,
                tts_cache_key,
                tts_sample_rate,
            )

            cfg = self._scenario.user_audio
            sample_rate = tts_sample_rate(cfg)
            self._voice = EvalVoice(
                build_tts_service(cfg, sample_rate=sample_rate),
                sample_rate=sample_rate,
                cache_key=tts_cache_key(cfg),
                cache_dir=self._cache_dir,
            )
            await self._voice.start()

        # One STT pipeline to transcribe the bot's audio for `response`.
        if self._wants_response:
            from pipecat.evals.transcribe import EvalTranscriber, build_stt_service

            self._transcriber = EvalTranscriber(build_stt_service(self._scenario.transcriber))
            self._transcriber.debug = self._debug
            await self._transcriber.start()

        failures: list[AssertionFailure] = []
        reader_task = asyncio.create_task(self._reader_loop())

        self._debug("connected")
        try:
            try:
                await self._handshake()
                self._debug("handshake: ok (bot-ready)")
            except TimeoutError:
                self._debug("handshake: failed (bot-ready not received)")
                failures.append(
                    AssertionFailure(
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
                    failures.extend(await self._run_turn(turn, turn_idx))
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
            if self._voice is not None:
                await self._voice.aclose()
            if self._transcriber is not None:
                await self._transcriber.aclose()
            # Ask the bot to tear its pipeline down gracefully (closing its STT/TTS/
            # LLM connections) so the process exits on its own — best-effort; the
            # orchestrator still has a kill fallback.
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
        :mod:`pipecat.evals.transport`). ``capture_audio`` makes the bot forward
        its synthesized audio for ``tts_response`` transcription. ``record`` asks
        the eval transport to record the conversation audio (audio mode only).
        """
        from urllib.parse import quote

        flags = []
        if not self._scenario.bot_audio:
            flags.append("skip_tts=true")
        if self._wants_response:
            flags.append("capture_audio=true")
        if self._record_path and self._scenario.bot_audio:
            flags.append(f"record={quote(self._record_path, safe='')}")
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
                if exp.args is not None:
                    return "full"
                if exp.name is not None:
                    needs_name = True
        return "name" if needs_name else None

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
        await self._ws.send(ready.model_dump_json())

        # Hard gate — raises TimeoutError if the bot never announces readiness.
        await self._wait_for_event("bot_ready", BOT_READY_TIMEOUT_S)

        # If the scenario asserts on function-call name/args, ask the bot's
        # RTVIObserver to report them for the duration of this eval. Agents keep
        # the secure NONE default; only the eval transport understands this.
        level = self._required_report_level()
        if level is not None:
            configure = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={
                    "t": EVAL_CONFIGURE_MESSAGE_TYPE,
                    "d": {"function_call_report_level": {"*": level}},
                },
            )
            await self._ws.send(configure.model_dump_json())

        # Only send a reset when the scenario actually asked for one. An implicit
        # empty reset would race with bot startup flows (e.g. a greeting added in
        # on_client_connected), wiping the bot's context right after it set it up.
        if self._scenario.reset:
            reset = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_RESET_MESSAGE_TYPE, "d": {"messages": self._scenario.reset}},
            )
            await self._ws.send(reset.model_dump_json())

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
            await self._ws.send(message.model_dump_json())
        except Exception:
            pass

    async def _reader_loop(self) -> None:
        """Drain the WS, translate RTVI messages to friendly events, enqueue them."""
        import websockets  # lazy: see note at the top of the module

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

        Clears the response buffers and drains the unmatched event queue, so a
        greeting (or any prior bot output) the user just interrupted can't be
        matched against this turn. Diagnostics (``events_seen``,
        ``latest_event_times``) are left intact for send_after lookups.
        """
        self._text_buffer = []
        self._tts_audio = bytearray()
        dropped = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
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
                return [{"type": "user_started_speaking"}]
            case "bot-interrupted":
                # The bot's in-flight output was cut off — a VAD barge-in or a
                # run_immediately text interrupt. Drop it so only what the bot says
                # *after* the interruption is matched. Service-independent, the same
                # path for both modalities, and no timestamps.
                self._discard_interrupted_output()
                return [{"type": "bot_interrupted"}]
            case "user-stopped-speaking":
                return [{"type": "user_stopped_speaking"}]
            case "user-transcription":
                if data.get("final"):
                    return [{"type": "user_transcription", "transcript": data.get("text", "")}]
                return []
            case "bot-llm-started":
                self._text_buffer = []
                return [{"type": "llm_started"}]
            case "bot-llm-text":
                # The LLM's text output -> llm_response (both modalities). Buffer
                # it and emit one segment at bot-llm-stopped (a clean boundary —
                # bot-llm-text reliably precedes bot-llm-stopped).
                self._text_buffer.append(data.get("text", ""))
                return []
            case "bot-llm-stopped":
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

    def _progress(self, record: TurnProgress) -> None:
        """Emit a progress record to the on_progress callback, if one was given."""
        if self._on_progress is not None:
            self._on_progress(record)

    async def _run_turn(self, turn: Turn, turn_idx: int) -> list[AssertionFailure]:
        """Drive one turn: optionally honor send_after, send user input, match expectations.

        The user turn is sent as ``send-text`` (text mode) or, when the scenario
        provides a ``user_audio`` block, as chunked ``raw-audio`` messages that
        the bot's STT transcribes for real.
        """
        failures: list[AssertionFailure] = []

        if turn.send_after is not None:
            try:
                await self._wait_send_after(turn.send_after)
            except TimeoutError as e:
                failures.append(
                    AssertionFailure(
                        turn_index=turn_idx,
                        expectation_index=-1,
                        event_name=turn.send_after.event,
                        reason=f"send_after never fired: {e}",
                    )
                )
                self._progress(
                    TurnProgress(
                        turn_idx, -1, turn.send_after.event, "timeout", failures[-1].reason
                    )
                )
                return failures

        if turn.user is not None:
            self._debug(f"send: {turn.user!r} ({'audio' if self._voice is not None else 'text'})")
            if self._voice is not None:
                await self._send_user_audio(turn.user)
            else:
                await self._send_user_text(turn.user, self._scenario.bot_audio)

        self._progress(TurnProgress(turn_idx, -1, turn.user or "", "turn"))

        for exp_idx, expectation in enumerate(turn.expect):
            budget_ms = expectation.within_ms or DEFAULT_EVENT_TIMEOUT_MS

            try:
                failure = await self._match_and_verify(expectation, budget_ms, turn_idx, exp_idx)
            except TimeoutError:
                reason = f"no matching {expectation.event!r} event arrived within {budget_ms}ms"
                failures.append(
                    AssertionFailure(
                        turn_index=turn_idx,
                        expectation_index=exp_idx,
                        event_name=expectation.event,
                        reason=reason,
                    )
                )
                self._progress(
                    TurnProgress(turn_idx, exp_idx, expectation.event, "timeout", reason)
                )
                break

            if failure:
                failures.append(failure)
                self._progress(
                    TurnProgress(turn_idx, exp_idx, expectation.event, "failed", failure.reason)
                )
            else:
                self._progress(
                    TurnProgress(
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
        await self._ws.send(message.model_dump_json())

    async def _send_user_audio(self, text: str) -> None:
        """Render ``text`` to audio (cached) and stream it as ``raw-audio`` chunks.

        The chunks are sent at real-time mic cadence (one ~20ms frame every ~20ms)
        rather than in a burst: the bot's VAD/STT endpoint on wall-clock timing, so
        a bursted utterance compresses the speech/silence boundary and gets
        mis-segmented. Pacing is monotonic-clock based so it doesn't drift.
        """
        pcm, sample_rate = await self._voice.generate(text)
        loop = asyncio.get_running_loop()
        next_send = loop.time()
        for chunk in _audio_chunks(pcm, sample_rate):
            message = RTVI.Message(
                type="raw-audio",
                id=self._message_id(),
                data={
                    "base64Audio": base64.b64encode(chunk).decode("ascii"),
                    "sampleRate": sample_rate,
                    "numChannels": 1,
                },
            )
            await self._ws.send(message.model_dump_json())
            next_send += AUDIO_CHUNK_MS / 1000
            await asyncio.sleep(max(0, next_send - loop.time()))

    async def _wait_for_event(self, event_name: str, timeout_s: float) -> None:
        """Block until ``event_name`` has been seen, or raise TimeoutError."""
        deadline = time.monotonic() + timeout_s
        while event_name not in self._latest_event_times:
            if time.monotonic() >= deadline:
                raise TimeoutError(event_name)
            await asyncio.sleep(SEND_AFTER_POLL_S)

    async def _wait_send_after(self, send_after: SendAfter) -> None:
        """Block until ``send_after.event`` has been seen + ``delay_ms`` has elapsed.

        If the event was seen earlier in the run, anchor on that time (potentially
        fire immediately). Otherwise, poll the latest_event_times map until the
        event arrives, then anchor on that.
        """
        target_delay_s = send_after.delay_ms / 1000.0
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
        expectation: Expectation,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> AssertionFailure | None:
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
        deadline = time.monotonic() + (budget_ms / 1000.0)
        self._last_match_text = ""

        aggregates = expectation.event in ("response", "llm_response", "tts_response") and (
            expectation.text_contains is not None or expectation.eval is not None
        )
        if not aggregates:
            event = await self._next_matching_event(expectation.event, deadline)
            payload_failure = self._check_payload(event, expectation, turn_idx, exp_idx)
            if payload_failure:
                return payload_failure
            judge_failure = await self._check_judge(event, expectation, turn_idx, exp_idx)
            if judge_failure is None:
                self._last_match_text = self._match_summary(event)
            return judge_failure

        def fail(reason: str) -> AssertionFailure:
            return AssertionFailure(turn_idx, exp_idx, expectation.event, reason)

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
        self._debug(f"match: aggregating {expectation.event!r} ({check})")
        aggregate = ""
        last_reason = ""
        seen_any = False
        while True:
            try:
                event = await self._next_matching_event(expectation.event, deadline)
            except TimeoutError:
                if not seen_any:
                    self._debug(f"match: timeout, no {expectation.event!r} event arrived")
                    raise  # no response at all → "no matching event arrived"
                self._debug(f"eval: timeout, not satisfied: {last_reason}")
                return fail(f"not satisfied within {budget_ms}ms: {last_reason}")

            seen_any = True
            aggregate += event.get("text", "")
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

    async def _evaluate_aggregate(
        self, aggregate: str, expectation: Expectation
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
            verdict = await self._judge.evaluate(expectation.eval, aggregate)
            if verdict.verdict == "no":
                return ("fail", f"judge said no: {verdict.reason}")
            if verdict.verdict == "continue":
                return ("continue", f"judge said continue: {verdict.reason}")
            return ("pass", f"judge said yes: {verdict.reason}")

        return ("pass", "")

    @staticmethod
    def _check_payload(
        event: dict,
        expectation: Expectation,
        turn_idx: int,
        exp_idx: int,
    ) -> AssertionFailure | None:
        """Apply payload-level checks to a matched event. Returns the first failure or None."""

        def fail(reason: str) -> AssertionFailure:
            return AssertionFailure(
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

        if expectation.name is not None:
            actual_name = event.get("name")
            if actual_name != expectation.name:
                return fail(f"name {actual_name!r} != expected {expectation.name!r}")

        if expectation.args is not None:
            # Subset match: every expected key/value must be present in the call,
            # so extra arguments the model includes (e.g. a `format`/unit field)
            # don't fail the assertion.
            actual_args = event.get("args") or {}
            missing = {k: v for k, v in expectation.args.items() if actual_args.get(k) != v}
            if missing:
                return fail(f"args {actual_args!r} missing expected {missing!r}")

        return None

    async def _check_judge(
        self,
        event: dict,
        expectation: Expectation,
        turn_idx: int,
        exp_idx: int,
    ) -> AssertionFailure | None:
        """Run the judge assertion if ``eval:`` was set on this expectation."""
        if expectation.eval is None:
            return None

        if self._judge is None:
            return AssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason="scenario uses 'eval:' but no judge could be built",
            )

        content = event.get("text") or event.get("transcript")
        if not content:
            return AssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"event has no text/transcript to judge: {event!r}",
            )

        verdict = await self._judge.evaluate(expectation.eval, content)
        if not verdict.passed:
            return AssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"eval {expectation.eval!r}: judge said no — {verdict.reason}",
            )

        return None


def _audio_chunks(pcm: bytes, sample_rate: int):
    """Yield ~20ms PCM slices, padded with leading and trailing silence.

    The padding is generated here, at send time, rather than stored in the cached
    audio, so it can change without invalidating the TTS cache.
    """
    samples_per_chunk = sample_rate * AUDIO_CHUNK_MS // 1000
    silence = b"\x00\x00" * samples_per_chunk
    bytes_per_chunk = samples_per_chunk * 2  # 16-bit mono

    for _ in range(AUDIO_LEADING_SILENCE_MS // AUDIO_CHUNK_MS):
        yield silence
    for offset in range(0, len(pcm), bytes_per_chunk):
        yield pcm[offset : offset + bytes_per_chunk]
    for _ in range(AUDIO_TRAILING_SILENCE_MS // AUDIO_CHUNK_MS):
        yield silence


async def run_scenario(
    scenario: Scenario,
    bot_url: str,
    *,
    connect_timeout_s: float = 5.0,
    on_progress: Callable[[TurnProgress], None] | None = None,
    record_path: str | None = None,
    cache_dir: str | None = None,
) -> EvalResult:
    """Run a scenario against a bot at the given WebSocket URL.

    Convenience wrapper around :class:`EvalSession`.

    Args:
        scenario: The parsed scenario to run.
        bot_url: WebSocket URL of the bot's eval transport.
        connect_timeout_s: How long to wait for the bot to accept the WS
            connection before giving up.
        on_progress: Optional per-turn/expectation progress callback (verbose).
        record_path: Optional path to record the conversation audio (audio mode).
        cache_dir: Optional directory for cached synthesized user audio
            (default ``<user-cache-dir>/pipecat/tts``).

    Returns:
        The structured outcome.
    """
    return await EvalSession(
        scenario,
        bot_url,
        connect_timeout_s=connect_timeout_s,
        on_progress=on_progress,
        record_path=record_path,
        cache_dir=cache_dir,
    ).run()
