#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Drivers: what the user says next, and how the outcome is judged.

An :class:`EvalDriver` runs the conversation with the bot over the session's
runtime (the client's pipeline, the event stream, the trace) and assembles the
run's result. :class:`ScriptedDriver` plays a scenario's ``turns:`` and matches
each turn's expectations; :class:`SimulationDriver` lets the persona LLM in the
pipeline hold the conversation and judges the whole of it.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.matcher import ExpectationMatcher
from pipecat.evals.persona import END_CALL_FUNCTION
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalResult,
    EvalTrace,
    EvalTurnProgress,
    EvalTurnResult,
    SimulationMetric,
    SimulationRunResult,
)
from pipecat.evals.scenario import EvalScenario, EvalSendAfter, EvalTurn
from pipecat.evals.simulation import EvalSimulation
from pipecat.frames.frames import FunctionCallResultProperties
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService

SEND_AFTER_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01
# The event the simulation driver appends when the persona calls end_call.
END_CALL_EVENT = "end_call"

R = TypeVar("R")


class EvalDriver(ABC, Generic[R]):
    """Base class for the drivers: drives the conversation and scores it.

    The runtime is shared, the client sends and the stream receives, and a
    driver decides what to send next and what counts as success. Subclasses
    implement :meth:`run` and :meth:`result`. The user-turn primitives here,
    :meth:`_say` and :meth:`_press`, keep the judge's transcript and the
    stream's turn bookkeeping consistent whichever driver sends.
    """

    def __init__(
        self,
        *,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalTurnProgress], Awaitable[None]],
    ):
        """Initialize the driver.

        Args:
            client: The connection to the bot, for the user's sends.
            stream: The bot's output as events.
            judge: The judge for ``eval:`` assertions, or ``None``; the user's
                turns are added to its conversation so replies are judged in
                context.
            trace: The run's trace.
            progress: Awaited with an :class:`EvalTurnProgress` as turns and
                expectations resolve.
        """
        self._client = client
        self._stream = stream
        self._judge = judge
        self._trace = trace
        self._progress = progress

    @abstractmethod
    async def run(self) -> list[EvalAssertionFailure]:
        """Drive the conversation to its end and return the failures."""

    @abstractmethod
    def result(
        self,
        *,
        failures: list[EvalAssertionFailure],
        duration_ms: int,
        events_seen: list[dict],
        debug_log: list[str],
        skipped: str | None = None,
    ) -> R:
        """Assemble the run's result from what the driver scored and the session saw.

        Args:
            failures: The run's failures: the driver's own, plus the session's
                (a failed connect or handshake, a harness error).
            duration_ms: Wall-clock time the run took.
            events_seen: Every event observed, for diagnostics.
            debug_log: The run's trace.
            skipped: Why the run was not driven at all, or ``None``.
        """

    def record_failure(self, failure: EvalAssertionFailure) -> None:
        """Note a run-level failure the session raised while the driver was running.

        Args:
            failure: The failure, scored against the trace's current turn.
        """

    async def _say(self, text: str, *, audio_file: str | None = None) -> None:
        """Send one user utterance to the bot.

        The utterance goes out as the recording in ``audio_file`` when given,
        spoken by the user TTS when the client has one, else as text. Bot output
        still queued from an earlier turn is dropped first, so nothing the bot
        said before this input can be matched as its reply. The drop has to
        precede the send: once the input reaches the bot, its reaction to this
        very input would be dropped along with the stale output.

        Args:
            text: What the user says; also recorded in the judge's conversation
                so a later reply is judged in context (e.g. a terse "That's
                four" answering this question).
            audio_file: Optional recording to play in place of synthesizing
                ``text``.
        """
        self._stream.drop_pending_bot_output("before send")
        how = audio_file or ("audio" if self._client.has_user_tts else "text")
        self._trace.log(f"send: {text!r} ({how})")
        if audio_file is not None:
            await self._client.play(audio_file)
        elif self._client.has_user_tts:
            await self._client.say(text)
        else:
            await self._client.send_text(text)
        if self._judge is not None:
            self._judge.add_user_message(text)
        # Only what the bot says in reply to this input is matched from here on.
        self._stream.input_sent()

    async def _press(self, keys: str) -> None:
        """Send DTMF keypresses as the user's turn (see :meth:`_say` for the drop).

        Args:
            keys: The keys to press, in order; recorded for the judge so the
                bot's reply is judged knowing what was pressed.
        """
        self._stream.drop_pending_bot_output("before send")
        self._trace.log(f"send: dtmf {keys!r}")
        await self._client.send_dtmf(keys)
        if self._judge is not None:
            self._judge.add_user_message(f"(DTMF keypad input: {keys})")
        self._stream.input_sent()


class ScriptedDriver(EvalDriver[EvalResult]):
    """Plays a scenario's ``turns:`` in order and matches each turn's expectations.

    A turn with a failed assertion ends the scenario, since the conversation is
    in an unknown state from there on. A scenario that scores each turn
    independently sets ``stop_on_failure: false`` to have every turn driven and
    reported.
    """

    def __init__(
        self,
        *,
        scenario: EvalScenario,
        default_timeout_ms: int,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalTurnProgress], Awaitable[None]],
    ):
        """Initialize the driver.

        Args:
            scenario: The scenario whose turns to play.
            default_timeout_ms: Latency budget for expectations without their
                own ``within_ms``.
            client: The connection to the bot.
            stream: The bot's output as events.
            judge: The judge for ``eval:`` assertions, or ``None``.
            trace: The run's trace.
            progress: Awaited with an :class:`EvalTurnProgress` as turns and
                expectations resolve.
        """
        super().__init__(
            client=client,
            stream=stream,
            judge=judge,
            trace=trace,
            progress=progress,
        )
        self._scenario = scenario
        self._default_timeout_ms = default_timeout_ms
        self._matcher = ExpectationMatcher(stream=stream, judge=judge, trace=trace)
        # One record per turn, filled in as the driver runs. They start as
        # not_run and stay that way on every path that ends the run early, so
        # the result always says which turns were actually scored.
        self.turns = [EvalTurnResult(turn_index=i) for i in range(len(scenario.turns))]

    def result(
        self,
        *,
        failures: list[EvalAssertionFailure],
        duration_ms: int,
        events_seen: list[dict],
        debug_log: list[str],
        skipped: str | None = None,
    ) -> EvalResult:
        """The scenario's result: passed only if nothing failed and nothing was skipped."""
        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures and skipped is None,
            failures=failures,
            turns=self.turns,
            duration_ms=duration_ms,
            events_seen=events_seen,
            debug_log=debug_log,
            skipped=skipped,
        )

    def record_failure(self, failure: EvalAssertionFailure) -> None:
        """Score a run-level failure against the turn it interrupted, if any.

        Before any turn started (a sub-pipeline that never came up) the failure's
        turn is -1 and every turn stays not_run.
        """
        if 0 <= failure.turn_index < len(self.turns):
            record = self.turns[failure.turn_index]
            record.status = "failed"
            record.failures.append(failure)

    async def run(self) -> list[EvalAssertionFailure]:
        """Drive the scenario's turns in order, filling in their records."""
        failures: list[EvalAssertionFailure] = []
        for turn_idx, turn in enumerate(self._scenario.turns):
            self._trace.turn = turn_idx
            self._trace.log(f"--- turn {turn_idx}: {turn.user!r}")
            turn_started = time.monotonic()
            turn_failures = await self._run_turn(turn, turn_idx)
            record = self.turns[turn_idx]
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

    async def _run_turn(self, turn: EvalTurn, turn_idx: int) -> list[EvalAssertionFailure]:
        """Drive one turn: honor ``send_after``, send the input, match the expectations."""
        # The turn's function calls match by name in any order; start each turn
        # with an empty buffer so a prior turn's calls can't carry over.
        self._matcher.reset_turn()

        if turn.send_after is not None:
            failure = await self._await_send_after(turn.send_after, turn_idx)
            if failure is not None:
                return [failure]

        await self._send_turn(turn)
        await self._progress(EvalTurnProgress(turn_idx, -1, turn.user or turn.dtmf or "", "turn"))
        return await self._match_expectations(turn, turn_idx)

    async def _await_send_after(
        self, send_after: EvalSendAfter, turn_idx: int
    ) -> EvalAssertionFailure | None:
        """Hold the turn's send until its ``send_after`` fires; a failure if it never does."""
        try:
            await self._wait_send_after(send_after)
        except TimeoutError as e:
            # Only the event-anchored wait can time out; the pure-delay form
            # just sleeps. So event is never None here, but fall back for typing.
            event_name = send_after.event or "send_after"
            failure = EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=-1,
                event_name=event_name,
                reason=f"send_after never fired: {e}",
                kind="send_after_timeout",
            )
            self._trace.log(f"FAIL: {event_name}: {failure.reason}")
            await self._progress(
                EvalTurnProgress(turn_idx, -1, event_name, "timeout", failure.reason)
            )
            return failure
        return None

    async def _wait_send_after(self, send_after: EvalSendAfter) -> None:
        """Block until ``send_after.event`` has been seen + ``delay_ms`` has elapsed.

        If the event was seen earlier in the run, anchor on that time (potentially
        fire immediately). Otherwise, poll the stream's arrival times until the
        event arrives, then anchor on that.

        With no event (``send_after.event is None``), it's a pure time delay:
        sleep ``delay_ms`` from now (i.e. from the previous turn's send).

        Raises:
            TimeoutError: If the event never arrives.
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

    async def _send_turn(self, turn: EvalTurn) -> None:
        """Send the turn's input: its image (if any), then its utterance or keypresses.

        Turns that send nothing are observation-only and exist to match exactly
        the bot's pending output (a bot-first greeting), so they keep it.
        """
        # Register the turn's image before the user input, so the bot can serve
        # it when it requests a user image during the turn.
        if turn.image is not None:
            await self._client.send_image(turn.image)
        if turn.user is not None:
            await self._say(turn.user, audio_file=turn.audio)
        elif turn.dtmf is not None:
            await self._press(turn.dtmf)

    async def _match_expectations(
        self, turn: EvalTurn, turn_idx: int
    ) -> list[EvalAssertionFailure]:
        """Match the turn's expectations in order against one shared deadline.

        All of a turn's expectations share one deadline anchored at the send, so
        a stalled turn fails within a single ``within_ms`` budget instead of
        spending a fresh budget per expectation: a missing function call
        followed by a missing response fails in 60s total, not 120s.
        """
        failures: list[EvalAssertionFailure] = []
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


class SimulationDriver(EvalDriver[SimulationRunResult]):
    """Lets the persona LLM hold the conversation, then judges the whole of it.

    The persona runs inside the client's pipeline and answers the bot on its
    own. This driver only watches the conversation for its end: the persona's
    ``end_call``, the turn cap, or the wall-clock cap. Then it asks the judge
    whether the goal was achieved and how the conversation scored on each
    quality criterion.
    """

    def __init__(
        self,
        *,
        simulation: EvalSimulation,
        persona_llm: LLMService,
        persona_context: LLMContext,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalTurnProgress], Awaitable[None]],
    ):
        """Initialize the driver.

        Args:
            simulation: The simulation being run.
            persona_llm: The persona LLM service in the pipeline; ``end_call``
                is registered on it.
            persona_context: The persona's context, kept up to date with both
                sides of the conversation by the pipeline's aggregators.
            client: The connection to the bot.
            stream: The bot's output as events.
            judge: The judge for the goal and the quality criteria.
            trace: The run's trace.
            progress: Awaited with an :class:`EvalTurnProgress` as turns resolve.
        """
        super().__init__(client=client, stream=stream, judge=judge, trace=trace, progress=progress)
        self._simulation = simulation
        self._persona_llm = persona_llm
        self._context = persona_context
        self._turns = 0
        self._ended_by: str | None = None
        self._end_call: dict | None = None
        self._succeeded = False
        self._reason = ""
        self._metrics: list[SimulationMetric] = []

    async def run(self) -> list[EvalAssertionFailure]:
        """Watch the conversation until it ends, then judge it."""
        self._persona_llm.register_function(END_CALL_FUNCTION, self._on_end_call)
        await self._client.configure_persona()
        simulation = self._simulation
        # The bot's finished turns; in audio mode the transcription of what it said.
        turn_event = "response" if simulation.bot_audio else "llm_response"
        deadline = time.monotonic() + simulation.max_duration_s
        self._trace.log(
            f"persona: listening (up to {simulation.max_turns} bot turn(s), "
            f"{simulation.max_duration_s:g}s)"
        )
        while self._ended_by is None:
            try:
                event = await self._stream.next_any(deadline)
            except TimeoutError:
                self._ended_by = "max_duration"
                break
            if event["type"] == END_CALL_EVENT:
                self._ended_by = "end_call"
            elif event["type"] == turn_event and event.get("text"):
                self._turns += 1
                if self._turns >= simulation.max_turns:
                    self._ended_by = "max_turns"
        self._trace.log(f"persona: ended by {self._ended_by} after {self._turns} bot turn(s)")
        await self._judge_conversation()
        return []

    async def _on_end_call(self, params: FunctionCallParams) -> None:
        """The persona's ``end_call``: note its claim and end the conversation."""
        arguments = params.arguments or {}
        self._end_call = {
            "success": bool(arguments.get("success", False)),
            "reason": str(arguments.get("reason", "")),
        }
        # The call is the persona's last word: no follow-up response.
        await params.result_callback(
            {"status": "call ended"}, properties=FunctionCallResultProperties(run_llm=False)
        )
        await self._stream.append(
            {"type": END_CALL_EVENT, "text": self._end_call["reason"], **self._end_call}
        )

    def conversation(self) -> list[dict]:
        """The conversation with the persona as ``user`` and the bot as ``assistant``.

        The persona's context holds it the other way round (the bot is what the
        persona LLM answers), so the roles are swapped for the judge and the
        result. Tool calls and results are left out.
        """
        swapped = {"user": "assistant", "assistant": "user"}
        messages = []
        for message in self._context.get_messages():
            if not isinstance(message, dict):
                continue
            role, content = message.get("role"), message.get("content")
            if role in swapped and isinstance(content, str) and content.strip():
                messages.append({"role": swapped[role], "content": content})
        return messages

    async def _judge_conversation(self) -> None:
        """Decide the goal and score each quality criterion over the conversation."""
        if self._judge is None:
            self._reason = "no judge configured"
            return
        for message in self.conversation():
            if message["role"] == "user":
                self._judge.add_user_message(message["content"])
            else:
                self._judge.add_assistant_message(message["content"])
        verdict = await self._judge.evaluate_conversation(self._simulation.success)
        self._succeeded = verdict.verdict == "yes"
        self._reason = verdict.reason
        self._trace.log(
            f"judge: goal {'achieved' if self._succeeded else 'not achieved'}: {verdict.reason}"
        )
        for metric in self._simulation.metrics:
            verdict = await self._judge.evaluate_conversation(metric.criterion)
            score = 1.0 if verdict.verdict == "yes" else 0.0
            self._metrics.append(
                SimulationMetric(
                    name=metric.name, score=score, reason=verdict.reason, weight=metric.weight
                )
            )
            self._trace.log(f"judge: {metric.name} = {score:g}: {verdict.reason}")

    def result(
        self,
        *,
        failures: list[EvalAssertionFailure],
        duration_ms: int,
        events_seen: list[dict],
        debug_log: list[str],
        skipped: str | None = None,
    ) -> SimulationRunResult:
        """The run's result; a run-level failure makes it an error, not a goal failure."""
        error = skipped or ("; ".join(f.reason for f in failures) if failures else None)
        weights = sum(m.weight for m in self._metrics)
        quality = sum(m.score * m.weight for m in self._metrics) / weights if weights else None
        return SimulationRunResult(
            simulation_name=self._simulation.name,
            succeeded=self._succeeded and error is None,
            reason=error or self._reason,
            error=error,
            quality=quality,
            metrics=self._metrics,
            messages=self.conversation(),
            turns=self._turns,
            ended_by=self._ended_by or "error",
            end_call=self._end_call,
            duration_ms=duration_ms,
            events_seen=events_seen,
            debug_log=debug_log,
        )
