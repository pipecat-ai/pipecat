#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The scripted driver: plays a scenario's turns and matches each turn's expectations."""

import asyncio
import time
from collections.abc import Awaitable, Callable

from pipecat.evals.base_driver import BaseEvalDriver
from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.matcher import ExpectationMatcher
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalProgress,
    EvalScriptResult,
    EvalScriptTurnProgress,
    EvalScriptTurnResult,
    EvalTrace,
)
from pipecat.evals.script import EvalScriptScenario, EvalScriptTurn, EvalSendAfter

SEND_AFTER_MAX_WAIT_S = 30.0
# How long a turn waits for the bot to finish speaking before it is sent. A
# reply is still spoken after its first sentence satisfied the previous turn;
# a turn sent over it would have that tail transcribed as its own reply, or
# discarded together with the answer that followed it in the same breath.
BOT_QUIET_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01


class EvalScriptDriver(BaseEvalDriver[EvalScriptResult]):
    """Plays a scenario's turns in order and matches each turn's expectations.

    A failed turn ends the scenario by default, since the conversation is in
    an unknown state from there on; ``stop_on_failure: false`` drives every
    turn regardless.
    """

    def __init__(
        self,
        *,
        scenario: EvalScriptScenario,
        default_timeout_ms: int,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalProgress], Awaitable[None]],
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
            progress: Awaited with an :class:`~pipecat.evals.results.EvalProgress` as turns and
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
        self.turns = [EvalScriptTurnResult(turn_index=i) for i in range(len(scenario.turns))]

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

    def result(
        self,
        *,
        failures: list[EvalAssertionFailure],
        duration_ms: int,
        events_seen: list[dict],
        debug_log: list[str],
        skipped: str | None = None,
    ) -> EvalScriptResult:
        """The scenario's result: passed only if nothing failed and nothing was skipped."""
        return EvalScriptResult(
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

    async def _run_turn(self, turn: EvalScriptTurn, turn_idx: int) -> list[EvalAssertionFailure]:
        """Drive one turn: honor ``send_after``, send the input, match the expectations."""
        # The turn's function calls match by name in any order; start each turn
        # with an empty buffer so a prior turn's calls can't carry over.
        self._matcher.reset_turn()

        if turn.send_after is not None:
            failure = await self._await_send_after(turn.send_after, turn_idx)
            if failure is not None:
                return [failure]
        elif turn.user is not None or turn.dtmf is not None:
            # A caller waits for the bot to finish; a turn that means to talk
            # over it says so with ``send_after``.
            await self._await_bot_quiet()
        elif turn_idx > 0:
            await self._await_previous_reply()

        await self._send_turn(turn)
        await self._progress(
            EvalScriptTurnProgress(turn_idx, -1, turn.user or turn.dtmf or "", "turn")
        )
        return await self._match_expectations(turn, turn_idx)

    async def _await_bot_quiet(self) -> None:
        """Hold the send while the bot is speaking, up to ``BOT_QUIET_MAX_WAIT_S``."""
        if not self._stream.bot_speaking:
            return
        self._trace.log("send: waiting for the bot to finish speaking")
        if not await self._stream.wait_bot_quiet(BOT_QUIET_MAX_WAIT_S):
            self._trace.log(f"send: bot still speaking after {BOT_QUIET_MAX_WAIT_S:g}s, sending")

    async def _await_previous_reply(self) -> None:
        """Let a reply the bot is still speaking end before an observing turn starts.

        A turn that sends nothing waits for what the bot says next, so what
        the bot is still saying belongs to the turn before it, however late its
        transcription lands. The first turn keeps the bot's greeting.
        """
        if not self._stream.bot_speaking:
            return
        self._trace.log("observe: waiting for the bot to finish the previous reply")
        await self._stream.wait_bot_quiet(BOT_QUIET_MAX_WAIT_S)
        self._stream.drop_pending_bot_output("the previous reply")
        self._stream.turn_boundary()

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
                EvalScriptTurnProgress(turn_idx, -1, event_name, "timeout", failure.reason)
            )
            return failure
        return None

    async def _wait_send_after(self, send_after: EvalSendAfter) -> None:
        """Wait until ``send_after.event`` has been seen and ``delay_ms`` more have passed.

        With no event it is a plain delay from the previous turn's send.

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

    async def _send_turn(self, turn: EvalScriptTurn) -> None:
        """Send the turn's input: its image, if any, then its utterance or keypresses.

        A turn that sends nothing only observes, so the bot's pending output is
        kept for it.
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
        self, turn: EvalScriptTurn, turn_idx: int
    ) -> list[EvalAssertionFailure]:
        """Match the turn's expectations in order, all within one deadline anchored at the send.

        A stalled turn then fails within a single ``within_ms`` budget rather than
        one budget per expectation.
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
                    EvalScriptTurnProgress(turn_idx, exp_idx, expectation.event, "timeout", reason)
                )
                break

            if failure:
                failures.append(failure)
                self._trace.log(f"FAIL: {expectation.event}: {failure.reason}")
                await self._progress(
                    EvalScriptTurnProgress(
                        turn_idx, exp_idx, expectation.event, "failed", failure.reason
                    )
                )
            else:
                await self._progress(
                    EvalScriptTurnProgress(
                        turn_idx,
                        exp_idx,
                        expectation.event,
                        "matched",
                        self._matcher.last_match_text,
                    )
                )
        return failures
