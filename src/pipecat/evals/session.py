#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session: drives a bot over RTVI and asserts on the events it emits.

An :class:`EvalSession` connects to a running bot's eval transport, walks
through a parsed :class:`~pipecat.evals.scenario.EvalScenario`, and verifies
that the expected events arrive in order, with the right payloads, within their
latency budgets. It returns an :class:`~pipecat.evals.results.EvalResult`.

The session composes a shared runtime with a driver. The runtime is the
:class:`~pipecat.evals.client.EvalClient`, a Pipecat pipeline acting as an RTVI
client that sends the user's turns, and the
:class:`~pipecat.evals.events.EvalEventStream`, the bot's output as the events
scenarios assert on. The driver decides what the user says next and how the
outcome is scored: the :class:`~pipecat.evals.driver.ScriptedDriver` plays the
scenario's turns and matches their expectations with the
:class:`~pipecat.evals.matcher.ExpectationMatcher`.

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

import time
import traceback
import warnings
from collections.abc import Callable

from loguru import logger

from pipecat.evals.client import EvalClient
from pipecat.evals.driver import EvalDriver, ScriptedDriver
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import EvalAssertionFailure, EvalResult, EvalTrace, EvalTurnProgress
from pipecat.evals.scenario import EvalScenario, describe_config
from pipecat.evals.services import stt_service_from_config, tts_service_from_config
from pipecat.evals.tts import CachingTTSService
from pipecat.services.stt_service import STTService
from pipecat.utils.base_object import BaseObject

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000


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

            record_path: When set (and the scenario is audio mode), the
                conversation audio (both sides) is recorded to this path.
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
        # response (audio modality): the bot's actual audio, transcribed by an STT
        # in the client's pipeline. Needs audio mode, so run() skips otherwise.
        self._wants_response: bool = any(
            exp.event == "response" for turn in scenario.turns for exp in turn.expect
        )

        # Timestamped trace of the harness's own decisions, for diagnosing flakes.
        self._trace = EvalTrace()
        # The bot's output as events: fed by the client's pipeline, read by the driver.
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)
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
        # What the user says next and how the outcome is scored: a scenario is
        # played by the scripted driver.
        self._driver: EvalDriver = ScriptedDriver(
            scenario=scenario,
            default_timeout_ms=default_timeout_ms,
            client=self._client,
            stream=self._stream,
            judge=judge,
            trace=self._trace,
            progress=self._progress,
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

        # The `response` transcription needs the bot's actual audio; without audio
        # mode there's nothing to transcribe, so skip rather than fail. (Normally
        # unreachable: EvalScenario.load resolves `response` to llm_response in text
        # modality; this guards Scenarios built directly.)
        if self._wants_response and not self._scenario.bot_audio:
            reason = "asserts 'response' transcription but judge modality is text (no audio)"
            logger.warning(f"Eval '{self._scenario.name}': {reason}; skipping")
            return self._result(started, [], skipped=reason)

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
            return self._result(started, [failure])

        failures = await self._drive()
        self._trace.log(f"done: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
        return self._result(started, failures)

    def _result(
        self, started: float, failures: list[EvalAssertionFailure], skipped: str | None = None
    ) -> EvalResult:
        """Assemble the run's result."""
        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures and skipped is None,
            failures=failures,
            turns=self._driver.turns,
            duration_ms=int((time.monotonic() - started) * 1000),
            events_seen=self._stream.events_seen,
            debug_log=self._trace.lines,
            skipped=skipped,
        )

    async def _drive(self) -> list[EvalAssertionFailure]:
        """Start the client, run the handshake and the driver, and tear down."""
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
                failures = await self._driver.run()
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
            turns = self._driver.turns
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
