#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Drivers: what the user says next, and how the outcome is judged.

An :class:`EvalDriver` runs the conversation with the bot over the session's
runtime (the client's pipeline, the event stream, the trace) and produces the
run's failures. :class:`ScriptedDriver` plays a scenario's ``turns:`` and matches
each turn's expectations; a simulation driver generates the user's turns from a
persona and judges the whole conversation instead.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.matcher import ExpectationMatcher
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalTrace,
    EvalTurnProgress,
    EvalTurnResult,
)
from pipecat.evals.scenario import EvalScenario, EvalSendAfter, EvalTurn

SEND_AFTER_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01


class EvalDriver(ABC):
    """Base class for the drivers: drives the conversation and scores it.

    The runtime is shared, the client sends and the stream receives, and a
    driver decides what to send next and what counts as success. Subclasses
    implement :meth:`run`. The user-turn primitives here, :meth:`_say` and
    :meth:`_press`, keep the judge's transcript and the stream's turn
    bookkeeping consistent whichever driver sends.
    """

    def __init__(
        self,
        *,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalTurnProgress], Awaitable[None]],
        bot_audio: bool,
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
            bot_audio: Whether the bot speaks its replies (audio mode).
        """
        self._client = client
        self._stream = stream
        self._judge = judge
        self._trace = trace
        self._progress = progress
        self._bot_audio = bot_audio
        # One record per turn the driver scores, filled in as it runs. They start
        # as not_run and stay that way on every path that ends the run early, so
        # the result always says which turns were actually scored.
        self.turns: list[EvalTurnResult] = []

    @abstractmethod
    async def run(self) -> list[EvalAssertionFailure]:
        """Drive the conversation to its end and return the failures."""

    async def _say(self, text: str, *, audio_file: str | None = None) -> None:
        """Send one user utterance to the bot.

        The utterance goes out as the recording in ``audio_file`` when given,
        spoken by the user TTS when the client has one, else as text.

        Anything still queued from the bot belongs to an earlier turn: this
        utterance hasn't been sent, so the bot cannot have responded to it yet.
        It is dropped first, or an expectation could match, and a judge rule on,
        output the bot produced for a previous turn. The bot's own interruption
        events close this window too, but only once the input reaches it, which
        is far too late when ``send_after`` holds the send back for seconds.
        Before the send, not after: by the time the input has streamed, the bot
        has begun reacting to it, and this turn's own ``user_started_speaking``
        / ``bot_interrupted`` would be dropped along with the stale output.

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
            await self._client.send_text(text, audio_response=self._bot_audio)
        if self._judge is not None:
            self._judge.add_user_message(text)
        # Suppress in-flight stragglers until the bot's fresh response begins
        # (llm-started clears the flag), so only what the bot says in reply to
        # this input is matched.
        self._stream.awaiting_llm_restart = True

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
        self._stream.awaiting_llm_restart = True


class ScriptedDriver(EvalDriver):
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
            bot_audio=scenario.bot_audio,
        )
        self._scenario = scenario
        self._default_timeout_ms = default_timeout_ms
        self._matcher = ExpectationMatcher(stream=stream, judge=judge, trace=trace)
        self.turns = [EvalTurnResult(turn_index=i) for i in range(len(scenario.turns))]

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
