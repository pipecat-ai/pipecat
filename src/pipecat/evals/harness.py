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
``llm_response``            ``bot-llm-text`` accumulated until ``bot-llm-stopped``
``function_call``           ``llm-function-call-in-progress``
==========================  ==============================================

Matching semantics: expected events must appear in the specified order, but
unmatched events may appear between them (so a scenario doesn't have to
enumerate every event the bot emits). The ``within_ms`` budget for each
expectation is measured from the most recent ``send-text`` / ``raw-audio`` send.

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
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.judge import Judge, build_default_judge
from pipecat.evals.scenario import Expectation, Scenario, SendAfter, Turn
from pipecat.evals.serializer import EVAL_RESET_MESSAGE_TYPE

# ``websockets`` is imported lazily inside the methods that use it. That keeps
# this module importable (and the CLI plugin loadable via pipecat-cli) even
# when the optional ``websockets-base`` extra isn't installed — users who
# actually run an eval get a clear ImportError at that point.

DEFAULT_EVENT_TIMEOUT_MS = 5000
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


class EvalSession:
    """Runs one :class:`Scenario` against a bot over a single WebSocket session.

    Connects as an RTVI client, drives each turn (sending ``send-text`` or
    ``raw-audio``), collects the RTVI events the bot emits, and asserts on them.
    Use :meth:`run`, or the :func:`run_scenario` convenience wrapper.
    """

    def __init__(self, scenario: Scenario, bot_url: str, connect_timeout_s: float = 5.0):
        """Initialize the eval session.

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
        """
        self._scenario = scenario
        self._bot_url = bot_url
        self._connect_timeout_s = connect_timeout_s

        self._ws: Any = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._latest_event_times: dict[str, float] = {}
        self._events_seen: list[dict] = []
        self._next_id = 0
        self._judge: Judge | None = None
        # Accumulates bot-llm-text deltas between bot-llm-started and
        # bot-llm-stopped to synthesize the llm_response event's text.
        self._llm_buffer: list[str] = []

    async def run(self) -> EvalResult:
        """Connect, drive the scenario, and return the result."""
        import websockets  # lazy: see note at the top of the module

        started = time.monotonic()

        try:
            async with asyncio.timeout(self._connect_timeout_s):
                self._ws = await websockets.connect(self._bot_url)
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
            await self._ws.close()

        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures,
            failures=failures,
            duration_ms=int((time.monotonic() - started) * 1000),
            events_seen=self._events_seen,
        )

    def _message_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

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
                for event in self._translate(message):
                    self._events_seen.append(event)
                    self._latest_event_times[event["type"]] = time.monotonic()
                    await self._queue.put(event)
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass

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
                self._llm_buffer = []
                return [{"type": "llm_started"}]
            case "bot-llm-text":
                self._llm_buffer.append(data.get("text", ""))
                return []
            case "bot-llm-stopped":
                return [{"type": "llm_response", "text": "".join(self._llm_buffer)}]
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
                return failures

        anchor = time.monotonic()
        if turn.user is not None:
            if self._scenario.user_audio is not None:
                await self._send_user_audio(turn.user, self._scenario.user_audio)
            else:
                await self._send_user_text(turn.user, self._scenario.bot_audio)
            anchor = time.monotonic()

        for exp_idx, expectation in enumerate(turn.expect):
            budget_ms = expectation.within_ms or DEFAULT_EVENT_TIMEOUT_MS

            try:
                matched = await self._await_event(expectation, anchor, budget_ms)
            except TimeoutError:
                failures.append(
                    AssertionFailure(
                        turn_index=turn_idx,
                        expectation_index=exp_idx,
                        event_name=expectation.event,
                        reason=(
                            f"no matching {expectation.event!r} event arrived within {budget_ms}ms"
                        ),
                    )
                )
                break

            failure = self._check_payload(matched, expectation, turn_idx, exp_idx)
            if failure:
                failures.append(failure)

            judge_failure = await self._check_judge(matched, expectation, turn_idx, exp_idx)
            if judge_failure:
                failures.append(judge_failure)

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

    async def _send_user_audio(self, text: str, user_audio: dict) -> None:
        """Render ``text`` to audio (cached) and stream it as ``raw-audio`` chunks."""
        from pipecat.evals.voice import generate_or_load

        pcm, sample_rate = await generate_or_load(text, user_audio)
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

    async def _await_event(self, expectation: Expectation, anchor: float, budget_ms: int) -> dict:
        """Pop events from the queue until one matching ``expectation.event`` arrives.

        Events that don't match the expected name are dropped (so a scenario
        doesn't have to enumerate every event the bot emits). They remain in
        ``events_seen`` and ``latest_event_times`` for diagnostics and send_after
        lookups.
        """
        deadline = anchor + (budget_ms / 1000.0)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()

            async with asyncio.timeout(remaining):
                event = await self._queue.get()

            if event.get("type") == expectation.event:
                return event

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

        if expectation.transcript_contains is not None:
            transcript = event.get("transcript", "")
            if expectation.transcript_contains not in transcript:
                return fail(
                    f"transcript {transcript!r} does not contain "
                    f"{expectation.transcript_contains!r}"
                )

        if expectation.text_contains is not None:
            text = event.get("text", "")
            if expectation.text_contains not in text:
                return fail(f"text {text!r} does not contain {expectation.text_contains!r}")

        if expectation.name is not None:
            actual_name = event.get("name")
            if actual_name != expectation.name:
                return fail(f"name {actual_name!r} != expected {expectation.name!r}")

        if expectation.args is not None:
            actual_args = event.get("args")
            if actual_args != expectation.args:
                return fail(f"args {actual_args!r} != expected {expectation.args!r}")

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
) -> EvalResult:
    """Run a scenario against a bot at the given WebSocket URL.

    Convenience wrapper around :class:`EvalSession`.

    Args:
        scenario: The parsed scenario to run.
        bot_url: WebSocket URL of the bot's eval transport.
        connect_timeout_s: How long to wait for the bot to accept the WS
            connection before giving up.

    Returns:
        The structured outcome.
    """
    return await EvalSession(scenario, bot_url, connect_timeout_s).run()
