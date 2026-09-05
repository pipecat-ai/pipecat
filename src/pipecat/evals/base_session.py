#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The runtime a session's driver runs on: one conversation with a bot.

A :class:`BaseEvalSession` connects to a running bot's eval transport as an RTVI
client and runs a driver over the shared runtime: the
:class:`~pipecat.evals.client.EvalClient`, a Pipecat pipeline acting as an RTVI
client that carries the user's side, and the
:class:`~pipecat.evals.events.EvalEventStream`, the bot's output as events. It
connects, runs the handshake, lets the driver converse, tears down, and has the
driver turn what happened (including a failed connect or a harness error) into
its result. :class:`~pipecat.evals.script_session.EvalScriptSession` and
:class:`~pipecat.evals.simulation_session.EvalSimulationSession` build the client and
the driver for their kind of eval.
"""

import time
import traceback
from abc import abstractmethod
from typing import Generic, TypeVar

from loguru import logger

from pipecat.evals.base_driver import BaseEvalDriver
from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalAssertionFailure, EvalProgress, EvalTrace
from pipecat.evals.scenario import EvalKind
from pipecat.utils.base_object import BaseObject

R = TypeVar("R")


class BaseEvalSession(BaseObject, Generic[R]):
    """One conversation with a bot over a single WebSocket session, driven to a result.

    The runtime the drivers share: connect, run the handshake, let the driver
    converse, tear down, and turn what happened (including a failed connect or
    a harness error) into the driver's result. Subclasses build the client and
    the driver for their kind of eval.

    Event handlers available:

    - on_progress: Called with an :class:`~pipecat.evals.results.EvalProgress`
      record as the conversation advances: a scripted scenario's turns and
      expectations as they resolve, a simulation's lines as they are spoken.
      Records are emitted in order, and :meth:`run` waits for every handler
      before it returns.
    """

    def __init__(self, *, kind: EvalKind, name: str, bot_url: str):
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
        self._register_event_handler("on_progress")
        # Built by the subclass: the bot's output as events, the connection to
        # the bot, and what drives the conversation.
        self._stream: EvalEventStream
        self._client: EvalClient
        self._driver: BaseEvalDriver[R]

    @abstractmethod
    def _describe(self) -> str:
        """The run's config summary, one line per section, for the trace."""

    def _skip_reason(self) -> str | None:
        """Why the run can't be driven at all, or ``None`` to run it."""
        return None

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

    async def _drive(self) -> list[EvalAssertionFailure]:
        """Start the client, converse, and tear down.

        An unexpected harness-side error (a sub-pipeline failing to start under
        load, a judge or transcriber raising mid-turn) is reported as a failure
        with its traceback in the trace, so the run still yields a structured
        result rather than a bare error at the suite.
        """
        await self._client.start()
        try:
            return await self._converse()
        except Exception as e:
            return [self._harness_error(e)]
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

    def _harness_error(self, e: Exception) -> EvalAssertionFailure:
        """Trace an unexpected error with its traceback and let the driver score it."""
        self._trace.log(f"error: {type(e).__name__}: {e}")
        for line in traceback.format_exc().rstrip().splitlines():
            self._trace.log(line)
        failure = self._failure("<error>", f"{type(e).__name__}: {e}", "harness_error")
        self._driver.record_failure(failure)
        return failure

    def _failure(self, event_name: str, reason: str, kind: str) -> EvalAssertionFailure:
        """A failure of the run itself rather than of an expectation.

        It is scored against the trace's current turn: -1 before the driver has
        started any (connecting, the handshake), else the turn under way.
        """
        return EvalAssertionFailure(
            turn_index=self._trace.turn,
            expectation_index=-1,
            event_name=event_name,
            reason=reason,
            kind=kind,
        )

    async def _progress(self, record: EvalProgress) -> None:
        """Emit a progress record to the ``on_progress`` handlers."""
        await self._call_event_handler("on_progress", record)
