#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The eval session: one conversation with a bot, driven to a result.

A session connects to a running bot's eval transport as an RTVI client,
runs the handshake, lets its driver converse, tears down, and returns the
driver's result, a failed connect or a harness error included. The two
session kinds build the client and the driver for their kind of scenario;
:meth:`EvalSession.from_scenario` builds whichever kind a scenario is.

Example::

    scenario = load_scenario_file("scenarios/greeting.yaml")
    result = await EvalSession.from_scenario(scenario, "ws://localhost:7860").run()
    print("PASS" if result.passed else "FAIL")
"""

import time
import traceback
from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from loguru import logger

from pipecat.evals.base_driver import BaseEvalDriver
from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import EvalAssertionFailure, EvalProgress, EvalTrace
from pipecat.evals.scenario import EvalKind, EvalScriptScenario, EvalSimulationScenario
from pipecat.evals.tts import CachingTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.evals.script_session import EvalScriptSession
    from pipecat.evals.simulation_session import EvalSimulationSession

R = TypeVar("R")


class EvalSession(BaseObject, Generic[R]):
    """One conversation with a bot, driven to a result.

    Connect, run the handshake, let the driver converse, tear down, and turn
    what happened, a failed connect or a harness error included, into the
    driver's result. The subclasses,
    :class:`~pipecat.evals.script_session.EvalScriptSession` and
    :class:`~pipecat.evals.simulation_session.EvalSimulationSession`, build
    the client and the driver for their kind of scenario; :meth:`from_scenario`
    picks the one a scenario needs.

    Event handlers available:

    - on_progress: Called with an :class:`~pipecat.evals.results.EvalProgress`
      record as the conversation advances: a scripted scenario's turns and
      expectations as they resolve, a simulation's lines as they are spoken.
      Records are emitted in order, and :meth:`run` waits for every handler
      before it returns.
    """

    def __init__(
        self,
        *,
        kind: EvalKind,
        name: str,
        bot_url: str,
    ):
        """Initialize the session's runtime.

        Args:
            kind: The scenario kind being run, for the trace.
            name: The scenario's or simulation's name.
            bot_url: WebSocket URL of the bot's eval transport.
        """
        super().__init__()
        self._kind = kind
        self._name = name
        self._bot_url = bot_url
        # Timestamped trace of the harness's own decisions, for diagnosing flakes.
        self._trace = EvalTrace()
        # Built by the subclass: the bot's output as events, the connection to
        # the bot, and what drives the conversation.
        self._stream: EvalEventStream
        self._client: EvalClient
        self._driver: BaseEvalDriver[R]
        self._register_event_handler("on_progress")

    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScriptScenario | EvalSimulationScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        default_timeout_ms: int | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        persona_llm: LLMService | None = None,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ) -> "EvalScriptSession | EvalSimulationSession":
        """Build a ready-to-run session for a scenario of either kind.

        A scripted scenario gets an
        :class:`~pipecat.evals.script_session.EvalScriptSession`, a
        simulation an
        :class:`~pipecat.evals.simulation_session.EvalSimulationSession`,
        each constructing the services it needs. Pass ``persona_llm``,
        ``judge``, ``user_tts``, or ``bot_stt`` to use your own. Then await
        :meth:`run`::

            session = EvalSession.from_scenario(scenario, "ws://localhost:7860")
            result = await session.run()

        Args:
            scenario: The parsed scenario to run, scripted or a simulation.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            default_timeout_ms: Scripted scenarios only: the latency budget for
                expectations without their own ``within_ms``. Defaults to 60s.
            record_path: Optional path to record the conversation audio (audio mode).
            cache_dir: Optional directory for cached synthesized user audio.
            use_cache: When False, ignore cached user audio and force fresh synthesis.
            stop_bot: When True, ask the bot to cancel its pipeline on teardown.
            trigger_disconnect: When True, fire the bot's ``on_client_disconnected``
                handler when the connection ends.
            persona_llm: Simulations only: override the persona LLM (default:
                built from the simulation's ``simulator``).
            judge: Override the judge (default: built from the scenario's ``judge``
                when the run needs one).
            user_tts: Override the user-audio TTS (default: built from the
                scenario's ``user_speech`` in audio mode).
            bot_stt: Override the bot-audio STT (default: built from the
                scenario's ``transcriber`` when the run transcribes the bot).

        Returns:
            A configured session of the scenario's kind, ready for :meth:`run`.

        Raises:
            ValueError: If ``persona_llm`` is given for a scripted scenario,
                which has no persona.
            TypeError: If ``scenario`` is neither kind.
        """
        # Imported here rather than at module level: both subclasses import this module.
        from pipecat.evals.script_session import DEFAULT_EVENT_TIMEOUT_MS, EvalScriptSession
        from pipecat.evals.simulation_session import EvalSimulationSession

        if isinstance(scenario, EvalSimulationScenario):
            return EvalSimulationSession.from_scenario(
                scenario,
                bot_url,
                connect_timeout_s=connect_timeout_s,
                record_path=record_path,
                cache_dir=cache_dir,
                use_cache=use_cache,
                stop_bot=stop_bot,
                trigger_disconnect=trigger_disconnect,
                persona_llm=persona_llm,
                judge=judge,
                user_tts=user_tts,
                bot_stt=bot_stt,
            )
        elif isinstance(scenario, EvalScriptScenario):
            if persona_llm is not None:
                raise ValueError(
                    f"persona_llm applies to simulations only; {scenario.name!r} is scripted"
                )
            return EvalScriptSession.from_scenario(
                scenario,
                bot_url,
                connect_timeout_s=connect_timeout_s,
                default_timeout_ms=(
                    DEFAULT_EVENT_TIMEOUT_MS if default_timeout_ms is None else default_timeout_ms
                ),
                record_path=record_path,
                cache_dir=cache_dir,
                use_cache=use_cache,
                stop_bot=stop_bot,
                trigger_disconnect=trigger_disconnect,
                judge=judge,
                user_tts=user_tts,
                bot_stt=bot_stt,
            )
        raise TypeError(f"expected a scripted scenario or a simulation, got {type(scenario)!r}")

    async def run(self) -> R:
        """Connect, drive the conversation, and return the result."""
        started = time.monotonic()
        self._trace.start()
        self._trace.log(f"run: {self._kind} {self._name!r} -> {self._bot_url}")
        # Record which speech / transcription / judge services and models were used,
        # so a saved eval.log is self-describing (no need to cross-reference config).
        for line in self._describe().splitlines():
            self._trace.log(line)

        skipped = self._skip_reason()
        if skipped is not None:
            logger.warning(f"Eval '{self._name}': {skipped}; skipping")
            return self._result(started, [], skipped=skipped)

        # A bot that never accepts is a clean <connect> failure.
        try:
            await self._client.wait_for_bot()
        except (OSError, TimeoutError) as e:
            reason = f"failed to connect to {self._bot_url}: {e.__class__.__name__}"
            return self._result(started, [self._failure("<connect>", reason, "connect_failed")])

        failures = await self._drive()
        self._trace.log(f"done: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
        return self._result(started, failures)

    async def _drive(self) -> list[EvalAssertionFailure]:
        """Start the client, converse, and tear down.

        An error in the harness itself is recorded as a failure with its
        traceback, so the run still yields a result.
        """
        await self._client.start()
        try:
            return await self._converse()
        except Exception as e:
            return [self._error_failure(e)]
        finally:
            await self._client.stop()
            # Progress handlers run as tasks; wait them out so every record is
            # delivered before the caller has the result in hand.
            await self.cleanup()

    async def _converse(self) -> list[EvalAssertionFailure]:
        """Run the handshake, then the driver; a bot that never says ready is a failure."""
        self._trace.log("connected")
        try:
            await self._client.handshake()
        except TimeoutError as e:
            self._trace.log("handshake: failed (bot-ready not received)")
            return [self._failure("<bot-ready>", str(e), "handshake_timeout")]
        self._trace.log("handshake: ok (bot-ready)")
        return await self._driver.run()

    def _result(
        self, started: float, failures: list[EvalAssertionFailure], skipped: str | None = None
    ) -> R:
        """Have the driver assemble the run's result."""
        return self._driver.result(
            failures=failures,
            duration_ms=int((time.monotonic() - started) * 1000),
            events_seen=self._stream.events_seen,
            debug_log=self._trace.lines,
            skipped=skipped,
        )

    @abstractmethod
    def _describe(self) -> str:
        """The run's config summary, one line per section, for the trace."""

    def _skip_reason(self) -> str | None:
        """Why the run can't be driven at all, or ``None`` to run it."""
        return None

    def _failure(self, event_name: str, reason: str, kind: str) -> EvalAssertionFailure:
        """A failure of the run itself rather than of an expectation, scored against the trace's current turn (-1 before any)."""
        return EvalAssertionFailure(
            turn_index=self._trace.turn,
            expectation_index=-1,
            event_name=event_name,
            reason=reason,
            kind=kind,
        )

    def _error_failure(self, e: Exception) -> EvalAssertionFailure:
        """Trace an unexpected error with its traceback and let the driver score it."""
        self._trace.log(f"error: {type(e).__name__}: {e}")
        for line in traceback.format_exc().rstrip().splitlines():
            self._trace.log(line)
        failure = self._failure("<error>", f"{type(e).__name__}: {e}", "error")
        self._driver.record_failure(failure)
        return failure

    async def _progress(self, record: EvalProgress) -> None:
        """Emit a progress record to the ``on_progress`` handlers."""
        await self._call_event_handler("on_progress", record)
