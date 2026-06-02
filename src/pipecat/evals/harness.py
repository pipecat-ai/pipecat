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
``llm_response``            text mode: ``bot-llm-text`` joined at ``bot-llm-stopped``;
                            audio mode: one segment per ``bot-tts-text`` (spoken sentence)
``tts_response``            local-Whisper transcription of the bot's audio, per
                            spoken segment (audio mode only)
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

# Audio injection: chunk PCM into ~20ms slices (typical mic cadence) and append
# trailing silence so the bot's VAD detects end-of-speech.
AUDIO_CHUNK_MS = 20
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
        connect_timeout_s: float = 5.0,
        on_progress: Callable[[TurnProgress], None] | None = None,
    ):
        """Initialize the eval session.

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            on_progress: Optional callback invoked with a :class:`TurnProgress`
                as each turn and expectation resolves (used for verbose output).
        """
        self._scenario = scenario
        self._bot_url = bot_url
        self._connect_timeout_s = connect_timeout_s
        self._on_progress = on_progress

        self._ws: Any = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._latest_event_times: dict[str, float] = {}
        self._events_seen: list[dict] = []
        self._next_id = 0
        self._judge: Judge | None = None

        # One persistent TTS pipeline reused across the scenario's audio turns
        # (created in run() only when the scenario uses user_audio).
        self._voice: Any = None

        # Accumulates the bot's output text for the current response, to
        # synthesize llm_response. Source depends on the mode: bot-llm-text in
        # text mode (skip-TTS), bot-tts-text in audio mode (what was spoken).
        self._text_buffer: list[str] = []

        # When the current response began (bot-llm-started). Stamped onto each
        # llm_response so the matcher can ignore responses that started before a
        # turn's input (the on-connect greeting, or a turn the current one
        # interrupted). See _match_and_verify.
        self._response_started_at: float = 0.0

        # Text content of the most recently matched event (the bot's response, or
        # a user transcript), surfaced to verbose progress. Empty for events with
        # no text (llm_started, function_call, speaking events).
        self._last_match_text: str = ""

        # tts_response: when a scenario asserts on the bot's actual spoken audio,
        # the harness captures that audio and transcribes it locally. Lazy — only
        # set up when needed.
        self._wants_tts_response: bool = any(
            exp.event == "tts_response" for turn in scenario.turns for exp in turn.expect
        )
        self._transcriber: Any = None
        self._tts_audio: bytearray = bytearray()  # current spoken segment's audio
        self._tts_sample_rate: int = 0

    async def run(self) -> EvalResult:
        """Connect, drive the scenario, and return the result."""
        import websockets  # lazy: see note at the top of the module

        started = time.monotonic()

        # tts_response needs the bot's actual audio; without audio mode there's
        # nothing to transcribe, so skip rather than run a guaranteed failure.
        if self._wants_tts_response and not self._scenario.bot_audio:
            reason = "asserts tts_response but bot_audio is off (no audio to transcribe)"
            logger.warning(f"Eval '{self._scenario.name}': {reason}; skipping")
            return EvalResult(
                scenario_name=self._scenario.name,
                passed=False,
                skipped=reason,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        try:
            async with asyncio.timeout(self._connect_timeout_s):
                self._ws = await websockets.connect(self._connect_url())
        except (TimeoutError, OSError) as e:
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
                build_tts_service(cfg, sample_rate),
                sample_rate=sample_rate,
                cache_key=tts_cache_key(cfg),
            )
            await self._voice.start()

        # One STT pipeline to transcribe the bot's audio for tts_response.
        if self._wants_tts_response:
            from pipecat.evals.transcribe import EvalTranscriber, build_stt_service

            self._transcriber = EvalTranscriber(build_stt_service(self._scenario.transcriber))
            await self._transcriber.start()

        failures: list[AssertionFailure] = []
        reader_task = asyncio.create_task(self._reader_loop())

        try:
            await self._handshake()
            for turn_idx, turn in enumerate(self._scenario.turns):
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
            await self._ws.close()

        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures,
            failures=failures,
            duration_ms=int((time.monotonic() - started) * 1000),
            events_seen=self._events_seen,
        )

    def _connect_url(self) -> str:
        """Bot URL with the per-connection eval query flags.

        ``skip_tts`` (text mode) silences the bot before any LLM runs; the eval
        transport must read it at connect time because frames are ordered and a
        later message can't precede an on-connect greeting (see
        :mod:`pipecat.evals.transport`). ``capture_audio`` makes the bot forward
        its synthesized audio for ``tts_response`` transcription.
        """
        flags = []
        if not self._scenario.bot_audio:
            flags.append("skip_tts=true")
        if self._wants_tts_response:
            flags.append("capture_audio=true")
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
        """Send client-ready, wait for bot-ready, then optionally seed context."""
        ready = RTVI.Message(
            type="client-ready",
            id=self._message_id(),
            data=RTVI.ClientReadyData(
                version=RTVI.PROTOCOL_VERSION,
                about=RTVI.AboutClientData(library="pipecat-evals"),
            ).model_dump(),
        )
        await self._ws.send(ready.model_dump_json())

        try:
            await self._wait_for_event("bot_ready", BOT_READY_TIMEOUT_S)
        except TimeoutError:
            logger.warning("Eval session: bot-ready not received; proceeding anyway")

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
                if self._wants_tts_response:
                    await self._handle_tts_audio(message)
                for event in self._translate(message):
                    await self._enqueue(event)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass

    async def _enqueue(self, event: dict) -> None:
        """Record and queue a friendly event for the matcher."""
        self._events_seen.append(event)
        self._latest_event_times[event["type"]] = time.monotonic()
        await self._queue.put(event)

    async def _handle_tts_audio(self, message: dict) -> None:
        """Accumulate the bot's audio and emit a ``tts_response`` per spoken turn.

        We bound on the speaking boundaries (``bot-started-speaking`` /
        ``bot-stopped-speaking``), not the TTS ones: the output transport delays
        audio by PTS to play it out, so ``bot-tts-stopped`` fires while the tail
        is still streaming. ``bot-stopped-speaking`` fires once the audio has
        actually finished — only then is the buffer complete. The transcription
        is stamped with the response's start so the matcher anchors/aggregates it
        like ``llm_response``.
        """
        msg_type = message.get("type")
        if msg_type == EVAL_BOT_AUDIO_TYPE:
            data = message.get("data") or {}
            self._tts_audio.extend(base64.b64decode(data.get("audio", "")))
            self._tts_sample_rate = int(data.get("sampleRate", 0)) or self._tts_sample_rate
        elif msg_type == "bot-started-speaking":
            self._tts_audio = bytearray()
        elif msg_type == "bot-stopped-speaking":
            if not self._tts_audio or self._transcriber is None:
                return
            pcm, sample_rate = bytes(self._tts_audio), self._tts_sample_rate
            started_at = self._response_started_at
            self._tts_audio = bytearray()
            text = await self._transcriber.transcribe(pcm, sample_rate)
            await self._enqueue({"type": "tts_response", "text": text, "started_at": started_at})

    def _translate(self, message: dict) -> list[dict]:
        """Translate one RTVI server message into zero or more friendly events."""
        msg_type = message.get("type")
        data = message.get("data") or {}

        match msg_type:
            case "bot-ready":
                return [{"type": "bot_ready"}]
            case "user-started-speaking":
                return [{"type": "user_started_speaking"}]
            case "user-stopped-speaking":
                return [{"type": "user_stopped_speaking"}]
            case "user-transcription":
                if data.get("final"):
                    return [{"type": "user_transcription", "transcript": data.get("text", "")}]
                return []
            case "bot-llm-started":
                self._text_buffer = []
                self._response_started_at = time.monotonic()
                return [{"type": "llm_started"}]
            case "bot-llm-text":
                # Text mode (skip-TTS): the LLM text is the bot's output. Buffer
                # it and emit one segment at bot-llm-stopped (a clean boundary —
                # bot-llm-text reliably precedes bot-llm-stopped).
                if not self._scenario.bot_audio:
                    self._text_buffer.append(data.get("text", ""))
                return []
            case "bot-llm-stopped":
                if not self._scenario.bot_audio:
                    return [self._response_event("".join(self._text_buffer))]
                return []
            case "bot-tts-text":
                # Audio mode: each spoken sentence is a response segment, emitted
                # as it arrives. We can't bound on bot-tts-stopped because some
                # TTS services emit the text *after* the audio finishes (e.g.
                # OpenAI), which would yield empty responses. The matcher
                # aggregates the segments of the turn.
                if self._scenario.bot_audio:
                    return [self._response_event(data.get("text", ""))]
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

    def _response_event(self, text: str) -> dict:
        """Build one ``llm_response`` segment, stamped with the response's start.

        A segment is the full LLM text in text mode, or one spoken sentence in
        audio mode. The text may be empty (e.g. an interrupted response).
        ``started_at`` lets the matcher aggregate the segments of *this* turn and
        skip earlier ones (the greeting, or a prior turn the current one
        interrupted).
        """
        return {
            "type": "llm_response",
            "text": text,
            "started_at": self._response_started_at,
        }

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

        anchor = time.monotonic()
        if turn.user is not None:
            if self._voice is not None:
                await self._send_user_audio(turn.user)
            else:
                await self._send_user_text(turn.user, self._scenario.bot_audio)
            anchor = time.monotonic()

        # A turn that sends input only accepts a response that began after the
        # send (skipping the greeting / an interrupted prior turn). An
        # observe-only turn has no input to anchor on — it observes whatever the
        # bot produced autonomously (e.g. the on-connect greeting), so don't skip.
        match_floor = anchor if turn.user is not None else 0.0

        self._progress(TurnProgress(turn_idx, -1, turn.user or "", "turn"))

        for exp_idx, expectation in enumerate(turn.expect):
            budget_ms = expectation.within_ms or DEFAULT_EVENT_TIMEOUT_MS

            try:
                failure = await self._match_and_verify(
                    expectation, anchor, budget_ms, turn_idx, exp_idx, match_floor
                )
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
        """Render ``text`` to audio (cached) and stream it as ``raw-audio`` chunks."""
        pcm, sample_rate = await self._voice.generate(text)
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
        anchor: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
        match_floor: float,
    ) -> AssertionFailure | None:
        """Wait for the expected event and verify it. Returns a failure or None.

        Most events match a single event and are checked once. An ``llm_response``
        carrying a content check (``text_contains`` / ``eval:``) instead
        *aggregates*: it accumulates the text of successive response segments
        within the turn and re-checks on each new segment until the check passes,
        the judge affirmatively rejects, or the ``within_ms`` budget expires.
        Segments whose response began before ``match_floor`` (the greeting, or a
        turn the current one interrupted) are skipped; observe-only turns pass a
        floor of 0 so they can match the autonomous greeting.

        Raises:
            TimeoutError: when no matching event arrives at all (so the caller can
                report "no matching event arrived"). A response that arrives but
                never satisfies the content check returns a failure instead.
        """
        deadline = anchor + (budget_ms / 1000.0)
        self._last_match_text = ""

        aggregates = expectation.event in ("llm_response", "tts_response") and (
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

        aggregate = ""
        last_reason = ""
        seen_any = False
        while True:
            try:
                event = await self._next_matching_event(expectation.event, deadline)
            except TimeoutError:
                if not seen_any:
                    raise  # no response at all → "no matching event arrived"
                return fail(f"not satisfied within {budget_ms}ms: {last_reason}")

            # Ignore responses that began before this turn's input (greeting /
            # interrupted prior turn). Observe-only turns use a floor of 0.
            if event.get("started_at", 0.0) < match_floor:
                continue

            seen_any = True
            aggregate += event.get("text", "")
            status, reason = await self._evaluate_aggregate(aggregate, expectation)
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
            return (
                "continue",
                f"text {aggregate!r} does not contain {expectation.text_contains!r}",
            )

        if expectation.eval is not None:
            if not aggregate.strip():
                return ("continue", "no response text yet")
            # _match_and_verify guarantees a judge exists before aggregating eval:.
            assert self._judge is not None
            verdict = await self._judge.evaluate(expectation.eval, aggregate)
            if verdict.verdict == "no":
                return ("fail", f"eval {expectation.eval!r}: judge said no — {verdict.reason}")
            if verdict.verdict == "continue":
                return ("continue", f"eval {expectation.eval!r}: incomplete — {verdict.reason}")

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
    """Yield ~20ms PCM slices followed by trailing silence."""
    bytes_per_chunk = (sample_rate * AUDIO_CHUNK_MS // 1000) * 2  # 16-bit mono
    for offset in range(0, len(pcm), bytes_per_chunk):
        yield pcm[offset : offset + bytes_per_chunk]
    silence = b"\x00\x00" * (sample_rate * AUDIO_CHUNK_MS // 1000)
    for _ in range(AUDIO_TRAILING_SILENCE_MS // AUDIO_CHUNK_MS):
        yield silence


async def run_scenario(
    scenario: Scenario,
    bot_url: str,
    connect_timeout_s: float = 5.0,
    on_progress: Callable[[TurnProgress], None] | None = None,
) -> EvalResult:
    """Run a scenario against a bot at the given WebSocket URL.

    Convenience wrapper around :class:`EvalSession`.

    Args:
        scenario: The parsed scenario to run.
        bot_url: WebSocket URL of the bot's eval transport.
        connect_timeout_s: How long to wait for the bot to accept the WS
            connection before giving up.
        on_progress: Optional per-turn/expectation progress callback (verbose).

    Returns:
        The structured outcome.
    """
    return await EvalSession(scenario, bot_url, connect_timeout_s, on_progress).run()
