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
:meth:`EvalSession.from_scenario` builds whichever kind a scenario is, and
:class:`EvalSessionParams` is how the run behaves, whichever kind it is.

Example::

    scenario = load_scenario_file("scenarios/greeting.yaml")
    params = EvalSessionParams(stop_bot=True)
    result = await EvalSession.from_scenario(scenario, "ws://localhost:7860", params=params).run()
    print("PASS" if result.passed else "FAIL")
"""

import time
import traceback
import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, TypeVar, overload

from loguru import logger
from pydantic import BaseModel

from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalProgress,
    EvalScriptTurnProgress,
    EvalTrace,
)
from pipecat.evals.scenario import EvalKind, EvalScriptScenario, EvalSimulationScenario
from pipecat.utils.base_object import BaseObject

if TYPE_CHECKING:
    from pipecat.evals.base_driver import BaseEvalDriver
    from pipecat.evals.client import EvalClient
    from pipecat.evals.events import EvalEventStream
    from pipecat.evals.judge import EvalJudge
    from pipecat.evals.script_session import EvalScriptSession
    from pipecat.evals.simulation_session import EvalSimulationSession
    from pipecat.evals.tts import CachingTTSService
    from pipecat.services.llm_service import LLMService
    from pipecat.services.stt_service import STTService

R = TypeVar("R")

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000


class EvalSessionParams(BaseModel):
    """How a run behaves, whichever kind of scenario it is: timeouts, recording, caching, teardown.

    Plain configuration, so one instance serves many runs and crosses process
    boundaries; the services a run uses are passed to the session separately.

    Parameters:
        connect_timeout_s: How long to wait for the bot to accept the WS
            connection before giving up.
        default_timeout_ms: Scripted scenarios only: the latency budget for
            expectations without their own ``within_ms`` (the turn's expectations
            share one deadline anchored at the send). Defaults to 60s.
        record_path: Where to save the conversation audio, or ``None``. Only an
            audio-mode run records: a stereo WAV, the user on the left channel
            and the bot on the right.
        cache_dir: Directory for cached synthesized user audio, or ``None`` for
            the default (``<user-cache-dir>/pipecat/evals/tts``).
        use_cache: When False, ignore cached user audio and force fresh
            synthesis, with no cache reads or writes.
        stop_bot: When True, ask the bot to cancel its pipeline, and exit, on
            teardown. Leave False to keep it running for more scenarios.
        trigger_disconnect: When True, fire the bot's ``on_client_disconnected``
            handler when the connection ends. A scenario's own
            ``trigger_disconnect`` field also opts in. Bots often cancel their
            pipeline there, so it is off by default to avoid that between
            scenarios.
    """

    connect_timeout_s: float = 5.0
    default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS
    record_path: str | None = None
    cache_dir: str | None = None
    use_cache: bool = True
    stop_bot: bool = False
    trigger_disconnect: bool = False


def _params_with_deprecated_knobs(
    params: EvalSessionParams | None, caller: str, **knobs: object
) -> EvalSessionParams:
    """The run's params with the deprecated knob keyword arguments of ``caller`` folded in.

    A knob is passed by the old name, or ``None`` when the caller did not give
    it. A knob that was given overrides the ``params`` field of the same name,
    and warns that the keyword argument is deprecated.
    """
    given = {name: value for name, value in knobs.items() if value is not None}
    params = params or EvalSessionParams()
    if not given:
        return params
    names = ", ".join(f"`{name}`" for name in given)
    warnings.warn(
        f"{names} of `{caller}` {'is' if len(given) == 1 else 'are'} deprecated since 1.9.0 "
        "and will be removed in 2.0.0. Use `params=EvalSessionParams(...)` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    return params.model_copy(update=given)


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
        params: EvalSessionParams | None = None,
    ):
        """Initialize the session's runtime.

        Args:
            kind: The scenario kind being run, for the trace.
            name: The scenario's or simulation's name.
            bot_url: WebSocket URL of the bot's eval transport.
            params: How the run behaves; ``None`` for the defaults.
        """
        super().__init__()
        self._kind = kind
        self._name = name
        self._bot_url = bot_url
        self._params = params or EvalSessionParams()
        # Timestamped trace of the harness's own decisions, for diagnosing flakes.
        self._trace = EvalTrace()
        # Built by the subclass: the bot's output as events, the connection to
        # the bot, and what drives the conversation.
        self._stream: EvalEventStream
        self._client: EvalClient
        self._driver: BaseEvalDriver[R]
        self._register_event_handler("on_progress")

    @overload
    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScriptScenario,
        bot_url: str,
        *,
        params: EvalSessionParams | None = None,
        judge: "EvalJudge | None" = None,
        user_tts: "CachingTTSService | None" = None,
        bot_stt: "STTService | None" = None,
        on_progress: "Callable[[EvalScriptTurnProgress], None] | None" = None,
        connect_timeout_s: float | None = None,
        default_timeout_ms: int | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool | None = None,
        stop_bot: bool | None = None,
        trigger_disconnect: bool | None = None,
    ) -> "EvalScriptSession": ...

    @overload
    @classmethod
    def from_scenario(
        cls,
        scenario: EvalSimulationScenario,
        bot_url: str,
        *,
        params: EvalSessionParams | None = None,
        persona_llm: "LLMService | None" = None,
        judge: "EvalJudge | None" = None,
        user_tts: "CachingTTSService | None" = None,
        bot_stt: "STTService | None" = None,
        connect_timeout_s: float | None = None,
        default_timeout_ms: int | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool | None = None,
        stop_bot: bool | None = None,
        trigger_disconnect: bool | None = None,
    ) -> "EvalSimulationSession": ...

    @overload
    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScriptScenario | EvalSimulationScenario,
        bot_url: str,
        *,
        params: EvalSessionParams | None = None,
        persona_llm: "LLMService | None" = None,
        judge: "EvalJudge | None" = None,
        user_tts: "CachingTTSService | None" = None,
        bot_stt: "STTService | None" = None,
        on_progress: "Callable[[EvalScriptTurnProgress], None] | None" = None,
        connect_timeout_s: float | None = None,
        default_timeout_ms: int | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool | None = None,
        stop_bot: bool | None = None,
        trigger_disconnect: bool | None = None,
    ) -> "EvalScriptSession | EvalSimulationSession": ...

    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScriptScenario | EvalSimulationScenario,
        bot_url: str,
        *,
        params: EvalSessionParams | None = None,
        persona_llm: "LLMService | None" = None,
        judge: "EvalJudge | None" = None,
        user_tts: "CachingTTSService | None" = None,
        bot_stt: "STTService | None" = None,
        on_progress: "Callable[[EvalScriptTurnProgress], None] | None" = None,
        connect_timeout_s: float | None = None,
        default_timeout_ms: int | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool | None = None,
        stop_bot: bool | None = None,
        trigger_disconnect: bool | None = None,
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
            params: How the run behaves; ``None`` for the defaults.
            persona_llm: Simulations only: override the persona LLM (default:
                built from the simulation's ``simulator``).
            judge: Override the judge (default: built from the scenario's ``judge``
                when the run needs one).
            user_tts: Override the user-audio TTS (default: built from the
                scenario's ``user_speech`` in audio mode).
            bot_stt: Override the bot-audio STT (default: built from the
                scenario's ``transcriber`` when the run transcribes the bot).
            on_progress: Scripted scenarios only: a callback for each turn and
                expectation as it resolves.

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            connect_timeout_s: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            default_timeout_ms: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            record_path: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            cache_dir: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            use_cache: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            stop_bot: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            trigger_disconnect: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

        Returns:
            A configured session of the scenario's kind, ready for :meth:`run`.

        Raises:
            ValueError: If ``persona_llm`` is given for a scripted scenario,
                which has no persona, or ``on_progress`` for a simulation,
                which reports its progress through the event handler alone.
            TypeError: If ``scenario`` is neither kind.
        """
        # Imported here rather than at module level: both subclasses import this module.
        from pipecat.evals.script_session import EvalScriptSession
        from pipecat.evals.simulation_session import EvalSimulationSession

        params = _params_with_deprecated_knobs(
            params,
            "EvalSession.from_scenario",
            connect_timeout_s=connect_timeout_s,
            default_timeout_ms=default_timeout_ms,
            record_path=record_path,
            cache_dir=cache_dir,
            use_cache=use_cache,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
        )
        if isinstance(scenario, EvalSimulationScenario):
            if on_progress is not None:
                raise ValueError(
                    f"on_progress applies to scripted scenarios only; {scenario.name!r} is a "
                    "simulation"
                )
            return EvalSimulationSession.from_scenario(
                scenario,
                bot_url,
                params=params,
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
                params=params,
                on_progress=on_progress,
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
