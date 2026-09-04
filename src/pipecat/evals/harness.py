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
import time
import traceback
import warnings
from collections.abc import Callable

from loguru import logger

from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.matcher import ExpectationMatcher
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalResult,
    EvalTrace,
    EvalTurnProgress,
    EvalTurnResult,
)
from pipecat.evals.scenario import EvalScenario, EvalSendAfter, EvalTurn, describe_config
from pipecat.evals.services import stt_service_from_config, tts_service_from_config
from pipecat.evals.tts import CachingTTSService
from pipecat.services.stt_service import STTService
from pipecat.utils.base_object import BaseObject

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000
SEND_AFTER_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01


class EvalSession(BaseObject):
    """Runs one :class:`EvalScenario` against a bot over a single WebSocket session.

    Connects as an RTVI client, drives each turn (sending ``send-text``,
    ``raw-audio``, or ``dtmf``), collects the RTVI events the bot emits, and
    asserts on them. Build one with :meth:`from_scenario` (which constructs the
    judge, user TTS, and STT the scenario needs), then await :meth:`run`.

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
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ):
        """Initialize the eval session.

        The ``judge``, ``user_tts``, and ``bot_stt`` are injected pre-built:
        :meth:`from_scenario` constructs the defaults from the scenario's config
        and passes them in. Construct and pass your own to override them (e.g. a
        custom judge LLM, TTS, or STT service).

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
            user_tts: The :class:`~pipecat.evals.tts.CachingTTSService` that
                synthesizes user audio (added to the eval pipeline in audio mode),
                or ``None`` for text-mode scenarios.
            bot_stt: The ``STTService`` that transcribes the bot's audio into the
                ``response`` event (added to the eval pipeline in audio mode), or
                ``None`` when unused.
        """
        super().__init__()

        self._scenario = scenario
        self._bot_url = bot_url
        self._default_timeout_ms = default_timeout_ms
        self._judge: EvalJudge | None = judge
        # response (audio modality): the bot's actual audio, transcribed by an STT
        # in the client's pipeline. Needs audio mode, so run() skips otherwise.
        self._wants_response: bool = any(
            exp.event == "response" for turn in scenario.turns for exp in turn.expect
        )

        # Timestamped trace of the harness's own decisions, for diagnosing flakes.
        self._trace = EvalTrace()
        # The bot's output as events: fed by the client's pipeline, read by the matcher.
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)
        self._matcher = ExpectationMatcher(stream=self._stream, judge=judge, trace=self._trace)
        # The connection to the bot: the eval pipeline and the user's sends.
        self._client = EvalClient(
            scenario=scenario,
            bot_url=bot_url,
            stream=self._stream,
            trace=self._trace,
            connect_timeout_s=connect_timeout_s,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            user_tts=user_tts,
            bot_stt=bot_stt,
        )

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
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ) -> "EvalSession":
        """Build a ready-to-run session from a scenario, constructing what it needs.

        Builds the judge, user TTS, and STT the scenario calls for and injects them
        into a new session. Pass ``judge`` /
        ``user_tts`` / ``bot_stt`` to override any of them with your own pre-built
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
            user_tts: Override the user-audio TTS (default: built from
                ``scenario.user_speech`` in audio mode).
            bot_stt: Override the bot-audio STT (default: built from
                ``scenario.transcriber`` when the scenario asserts ``response``).

        Returns:
            A configured session, ready for :meth:`run`.
        """
        turns = scenario.turns
        if judge is None and any(exp.eval is not None for turn in turns for exp in turn.expect):
            with logger.contextualize(eval_pipeline="judge"):
                judge = EvalJudge.from_config(scenario.judge)

        if user_tts is None and scenario.user_speech is not None:
            with logger.contextualize(eval_pipeline="speech"):
                user_tts = tts_service_from_config(
                    scenario.user_speech, cache_dir=cache_dir, use_cache=use_cache
                )

        wants_response = any(exp.event == "response" for turn in turns for exp in turn.expect)
        if bot_stt is None and wants_response and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                bot_stt = stt_service_from_config(scenario.transcriber)

        session = cls(
            scenario,
            bot_url,
            connect_timeout_s=connect_timeout_s,
            default_timeout_ms=default_timeout_ms,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            judge=judge,
            user_tts=user_tts,
            bot_stt=bot_stt,
        )
        if on_progress is not None:
            session._add_legacy_progress_callback(on_progress)
        return session

    async def run(self) -> EvalResult:
        """Connect, drive the scenario, and return the result."""
        started = time.monotonic()
        self._trace.start()
        self._trace.log(f"run: scenario {self._scenario.name!r} -> {self._bot_url}")
        # Record which speech / transcription / judge services and models were used,
        # so a saved eval.log is self-describing (no need to cross-reference config).
        for line in describe_config(self._scenario).splitlines():
            self._trace.log(line)

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
            return self._result(started, turns, [], skipped=reason)

        # A bot that never accepts is a clean <connect> failure.
        try:
            await self._client.wait_for_bot()
        except (OSError, TimeoutError) as e:
            failure = EvalAssertionFailure(
                turn_index=-1,
                expectation_index=-1,
                event_name="<connect>",
                reason=f"failed to connect to {self._bot_url}: {e.__class__.__name__}",
                kind="connect_failed",
            )
            return self._result(started, turns, [failure])

        failures = await self._drive(turns)
        self._trace.log(f"done: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
        return self._result(started, turns, failures)

    def _result(
        self,
        started: float,
        turns: list[EvalTurnResult],
        failures: list[EvalAssertionFailure],
        skipped: str | None = None,
    ) -> EvalResult:
        """Assemble the run's result."""
        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures and skipped is None,
            failures=failures,
            turns=turns,
            duration_ms=int((time.monotonic() - started) * 1000),
            events_seen=self._stream.events_seen,
            debug_log=self._trace.lines,
            skipped=skipped,
        )

    async def _drive(self, turns: list[EvalTurnResult]) -> list[EvalAssertionFailure]:
        """Start the client, run the handshake and the turns, and tear down."""
        await self._client.start()
        failures: list[EvalAssertionFailure] = []
        try:
            # The STT and user TTS run inside the client's pipeline; the judge runs
            # out-of-band during matching. Everything below is under this `try` so
            # a service that fails to start (e.g. a local model under load)
            # surfaces as a failure rather than propagating raw.
            self._trace.log("connected")
            try:
                await self._client.handshake()
                self._trace.log("handshake: ok (bot-ready)")
            except TimeoutError as e:
                self._trace.log("handshake: failed (bot-ready not received)")
                failures.append(
                    EvalAssertionFailure(
                        turn_index=-1,
                        expectation_index=-1,
                        event_name="<bot-ready>",
                        reason=str(e),
                        kind="handshake_timeout",
                    )
                )
            else:
                failures = await self._run_turns(turns)
        except Exception as e:
            # An unexpected harness-side error (a sub-pipeline failing to start
            # under load, a judge/transcriber raising mid-turn, ...) would
            # otherwise propagate up to the suite and be swallowed as a bare
            # "error: <str>" with no eval.log. Capture it as a failure so the
            # reason and full traceback land in the result's debug trace (saved
            # to <bot>.eval.log) and the run still reports a structured outcome.
            self._trace.log(f"error: {type(e).__name__}: {e}")
            for line in traceback.format_exc().rstrip().splitlines():
                self._trace.log(line)
            failure = EvalAssertionFailure(
                turn_index=self._trace.turn,
                expectation_index=-1,
                event_name="<error>",
                reason=f"{type(e).__name__}: {e}",
                kind="harness_error",
            )
            failures.append(failure)
            # The raise happened either inside a turn — which is that turn's
            # failure — or before any of them started (a sub-pipeline that never
            # came up), where the trace's turn is still -1 and every turn is not_run.
            if 0 <= self._trace.turn < len(turns):
                record = turns[self._trace.turn]
                record.status = "failed"
                record.failures.append(failure)
        finally:
            await self._client.stop()
            # Progress handlers run as tasks; wait them out so every record is
            # delivered before the caller has the result in hand.
            await self.cleanup()
        return failures

    async def _run_turns(self, turns: list[EvalTurnResult]) -> list[EvalAssertionFailure]:
        """Drive the scenario's turns in order, filling in their records."""
        failures: list[EvalAssertionFailure] = []
        for turn_idx, turn in enumerate(self._scenario.turns):
            self._trace.turn = turn_idx
            self._trace.log(f"--- turn {turn_idx}: {turn.user!r}")
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
                    self._trace.log(f"turn {turn_idx} failed; stopping scenario (stop_on_failure)")
                    break
                self._trace.log(f"turn {turn_idx} failed; continuing (stop_on_failure: false)")
        return failures

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
        self._matcher.reset_turn()

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
                self._trace.log(f"FAIL: {event_name}: {failures[-1].reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, -1, event_name, "timeout", failures[-1].reason)
                )
                return failures

        # Register the turn's image (if any) before the user input, so the bot can
        # serve it when it requests a user image during the turn.
        if turn.image is not None:
            await self._client.send_image(turn.image)

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
            self._stream.drop_pending_bot_output("before send")

        if turn.user is not None:
            how = turn.audio or ("audio" if self._client.has_user_tts else "text")
            self._trace.log(f"send: {turn.user!r} ({how})")
            if turn.audio is not None:
                await self._client.play(turn.audio)
            elif self._client.has_user_tts:
                await self._client.say(turn.user)
            else:
                await self._client.send_text(turn.user, audio_response=self._scenario.bot_audio)
            # Record the user turn in the judge's conversation, so a later reply is
            # judged in context (e.g. a terse "That's four" answering this question).
            if self._judge is not None:
                self._judge.add_user_message(turn.user)
        elif turn.dtmf is not None:
            self._trace.log(f"send: dtmf {turn.dtmf!r}")
            await self._client.send_dtmf(turn.dtmf)
            # Record the keypresses for judge context, so the bot's reply is judged
            # knowing what was pressed.
            if self._judge is not None:
                self._judge.add_user_message(f"(DTMF keypad input: {turn.dtmf})")

        if turn.user is not None or turn.dtmf is not None:
            # Suppress in-flight stragglers until the bot's fresh response begins
            # (bot-llm-started clears the flag), so this turn matches only what the
            # bot says in reply to this input.
            self._stream.awaiting_llm_restart = True

        await self._progress(EvalTurnProgress(turn_idx, -1, turn.user or turn.dtmf or "", "turn"))

        # All of a turn's expectations share one deadline anchored at the send, so a
        # stalled turn fails within a single ``within_ms`` budget instead of spending
        # a fresh budget per expectation — e.g. a missing function call followed by a
        # missing response fails in 60s total, not 120s.
        anchor = time.monotonic()
        for exp_idx, expectation in enumerate(turn.expect):
            budget_ms = expectation.within_ms or self._default_timeout_ms

            try:
                failure = await self._matcher.match(
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
                self._trace.log(f"FAIL: {expectation.event}: {reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "timeout", reason)
                )
                break

            if failure:
                failures.append(failure)
                self._trace.log(f"FAIL: {expectation.event}: {failure.reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "failed", failure.reason)
                )
            else:
                await self._progress(
                    EvalTurnProgress(
                        turn_idx,
                        exp_idx,
                        expectation.event,
                        "matched",
                        self._matcher.last_match_text,
                    )
                )

        return failures

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
            self._trace.log(f"send_after: waiting {send_after.delay_ms}ms")
            await asyncio.sleep(target_delay_s)
            return

        deadline = time.monotonic() + SEND_AFTER_MAX_WAIT_S
        self._trace.log(f"send_after: waiting for {send_after.event!r} + {send_after.delay_ms}ms")

        while True:
            seen_at = self._stream.latest_event_times.get(send_after.event)
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
