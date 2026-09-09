#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Matching a scenario's expectations against the bot's events.

:class:`ExpectationMatcher` waits on the run's
:class:`~pipecat.evals.events.EvalEventStream` for the event an expectation
names and verifies it: a payload check, a judge verdict, the aggregation of a
reply across its segments, the any-order matching of a turn's function calls,
and the inverted ``absent:`` check.
"""

from loguru import logger

from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import EvalAssertionFailure, EvalTrace
from pipecat.evals.script import FUNCTION_CALL_EVENTS, EvalExpectation


class ExpectationMatcher:
    """Matches one expectation at a time against the event stream.

    Expected events must appear in order, with unmatched events allowed in
    between. A reply with a content check aggregates its segments and
    re-checks on each, so an interim "Let me check" is rolled past rather
    than taken for the answer. A turn's function calls match by name in any
    order.
    """

    def __init__(self, *, stream: EvalEventStream, judge: EvalJudge | None, trace: EvalTrace):
        """Initialize the matcher.

        Args:
            stream: The bot's events, consumed as expectations match.
            judge: The judge for ``eval:`` assertions, or ``None`` when the
                scenario has none.
            trace: The run's trace, for the matcher's progress.
        """
        self._stream = stream
        self._judge = judge
        self._trace = trace
        # function_call events popped while matching another expectation, held so
        # the turn's calls can be matched by name in any order (reset per turn).
        self._pending_function_calls: list[dict] = []
        # Text content of the most recently matched event (the bot's response, or
        # a user transcript), surfaced to verbose progress. Empty for events with
        # no text (llm_started, function_call, speaking events).
        self.last_match_text: str = ""

    def reset_turn(self) -> None:
        """Forget the previous turn's unclaimed function calls."""
        self._pending_function_calls = []

    async def match(
        self,
        expectation: EvalExpectation,
        anchor: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Wait for the expected event and verify it.

        Stale bot output is not filtered here: the stream drops it on an
        interruption and the driver before each send.

        Args:
            expectation: The expectation to match.
            anchor: Monotonic time the turn's budget is measured from.
            budget_ms: The expectation's latency budget.
            turn_idx: Index of the turn, for the failure record.
            exp_idx: Index of the expectation within the turn, for the record.

        Returns:
            The failure, or ``None`` when the expectation is satisfied.

        Raises:
            TimeoutError: When no matching event arrives at all (so the caller
                can report "no matching event arrived"). A response that arrives
                but never satisfies the content check returns a failure instead.
        """
        deadline = anchor + (budget_ms / 1000.0)
        self.last_match_text = ""

        if expectation.absent:
            return await self._match_absent(expectation, deadline, budget_ms, turn_idx, exp_idx)
        if expectation.aggregates:
            return await self._match_aggregating(
                expectation, deadline, budget_ms, turn_idx, exp_idx
            )
        if expectation.event in FUNCTION_CALL_EVENTS:
            # A call expectation holds the set of calls the turn should make; it
            # completes only when all are found, in any order (a response
            # arriving doesn't short-circuit it).
            return await self._match_function_calls(expectation, deadline, turn_idx, exp_idx)
        return await self._match_one(expectation, deadline, turn_idx, exp_idx)

    async def _match_one(
        self, expectation: EvalExpectation, deadline: float, turn_idx: int, exp_idx: int
    ) -> EvalAssertionFailure | None:
        """Match a single event and check its payload and judge assertion."""
        self._trace.log(f"match: waiting for {expectation.event!r}")
        event = await self._stream.next_event(expectation.event, deadline)
        payload_failure = self._check_payload(event, expectation, turn_idx, exp_idx)
        if payload_failure:
            return payload_failure
        judge_failure = await self._check_judge(event, expectation, turn_idx, exp_idx)
        if judge_failure is None:
            self.last_match_text = self._match_summary(event)
        return judge_failure

    async def _match_aggregating(
        self,
        expectation: EvalExpectation,
        deadline: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Accumulate reply segments until the content check passes or fails."""
        if expectation.eval is not None and self._judge is None:
            return self._failure(
                expectation,
                turn_idx,
                exp_idx,
                "scenario uses 'eval:' but no judge could be built",
                "no_judge",
            )

        check = "+".join(
            name
            for name, val in (
                ("text_contains", expectation.text_contains),
                ("eval", expectation.eval),
            )
            if val is not None
        )
        self._trace.log(f"match: waiting for {expectation.event!r} ({check})")
        aggregate = ""
        last_reason = ""
        seen_any = False
        while True:
            try:
                event = await self._stream.next_event(expectation.event, deadline)
            except TimeoutError:
                if not seen_any:
                    raise  # no response at all → caller logs "no matching event arrived"
                self._trace.log(f"eval: timeout, not satisfied: {last_reason}")
                # Without `eval:` the only way to be unsatisfied is a missing
                # substring: `text_contains` is monotonic, so it holds out for more
                # text rather than failing outright.
                return self._failure(
                    expectation,
                    turn_idx,
                    exp_idx,
                    f"not satisfied within {budget_ms}ms: {last_reason}",
                    "judge_continue" if expectation.eval is not None else "text_mismatch",
                )

            seen_any = True
            delta = self._event_text(event)
            aggregate += delta
            # Feed each segment to the judge as its own assistant message, so it
            # judges the bot's reply in the conversation's context (the cumulative
            # `aggregate` is kept only for text_contains and the match summary).
            if expectation.eval is not None and self._judge is not None:
                self._judge.add_assistant_message(delta)
            status, reason = await self._evaluate_aggregate(aggregate, expectation)
            self._trace.log(f"eval: {status} (aggregate={aggregate.strip()!r}) {reason}")
            if status == "pass":
                self.last_match_text = aggregate
                return None
            if status == "fail":
                # Only the judge can affirmatively fail an aggregate.
                return self._failure(expectation, turn_idx, exp_idx, reason, "judge_no")
            # "continue": wait for the next segment, separated by a space so
            # sentences don't run together (e.g. "...that. The weather...").
            aggregate += " "
            last_reason = reason

    async def _match_absent(
        self,
        expectation: EvalExpectation,
        deadline: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Pass when no event of this type arrives before the deadline; an arriving one fails at once, with its content."""
        self._trace.log(f"match: expecting NO {expectation.event!r} for {budget_ms}ms")
        try:
            event = await self._stream.next_event(expectation.event, deadline)
        except TimeoutError:
            # The quiet window held: absence confirmed.
            self.last_match_text = f"no {expectation.event!r} for {budget_ms}ms"
            return None
        return self._failure(
            expectation,
            turn_idx,
            exp_idx,
            f"expected no {expectation.event!r} within {budget_ms}ms, "
            f"but one arrived: {self._match_summary(event)}",
            "unexpected_event",
        )

    async def _match_function_calls(
        self,
        expectation: EvalExpectation,
        deadline: float,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Match every call in the expectation, in any order, within the budget; else a failure naming the call that was missing or whose args did not match."""
        matched: list[str] = []
        for spec in expectation.calls or []:
            want = spec.args or None
            self._trace.log(f"match: waiting for {expectation.event!r} ({spec.signature})")
            try:
                event = await self._next_function_call(spec.name, deadline, want, expectation.event)
            except TimeoutError:
                # A call of the right name with the wrong arguments is a different
                # failure from the call never being made, and the bot's arguments
                # are what the reader needs to see.
                near = [
                    ev.get("args")
                    for ev in self._pending_function_calls
                    if ev.get("type") == expectation.event
                    and (spec.name is None or ev.get("name") == spec.name)
                ]
                if want is not None and near:
                    return self._failure(
                        expectation,
                        turn_idx,
                        exp_idx,
                        f"no {spec.name!r} call had args {want!r} (saw {near!r})",
                        "function_args_mismatch",
                    )
                missing = spec.name or "any function"
                seen = ", ".join(matched) if matched else "none"
                return self._failure(
                    expectation,
                    turn_idx,
                    exp_idx,
                    f"function call {missing!r} not seen (matched: {seen})",
                    "missing_function_call",
                )
            matched.append(str(event.get("name")))

        self.last_match_text = ", ".join(matched) or "function call"
        return None

    async def _next_function_call(
        self,
        name: str | None,
        deadline: float,
        args: dict | None = None,
        event_type: str = "function_call",
    ) -> dict:
        """The next call event matching ``name`` (``None`` for any) and ``args``.

        Calls seen but not yet claimed are buffered, so a turn's calls can arrive
        in any order and a call the LLM corrects and repeats still satisfies it.
        Raises TimeoutError at ``deadline``.
        """

        def matches(ev: dict) -> bool:
            if ev.get("type") != event_type:
                return False
            if name is not None and ev.get("name") != name:
                return False
            if args is None:
                return True
            actual = ev.get("args") or {}
            return all(actual.get(k) == v for k, v in args.items())

        for i, ev in enumerate(self._pending_function_calls):
            if matches(ev):
                return self._pending_function_calls.pop(i)

        while True:
            event = await self._stream.next_any(deadline)
            if event.get("type") not in FUNCTION_CALL_EVENTS:
                continue
            if matches(event):
                return event
            self._pending_function_calls.append(event)

    async def _evaluate_aggregate(
        self, aggregate: str, expectation: EvalExpectation
    ) -> tuple[str, str]:
        """Check the accumulated reply text: ``pass``, ``fail``, or ``continue`` for more text.

        A missing substring is ``continue``; only the judge can ``fail``.
        """
        if expectation.text_contains is not None and not self._text_contains(
            aggregate, expectation.text_contains
        ):
            return ("continue", f"does not contain {expectation.text_contains!r}")

        if expectation.eval is not None:
            if not aggregate.strip():
                return ("continue", "no response text yet")
            # match() guarantees a judge exists before aggregating eval:.
            assert self._judge is not None
            with logger.contextualize(eval_pipeline="judge"):
                # The reply segments were added to the judge's conversation in the
                # aggregation loop; the judge evaluates that context, not `aggregate`.
                verdict = await self._judge.evaluate(expectation.eval)
            if verdict.verdict == "no":
                return ("fail", f"judge said no: {verdict.reason}")
            if verdict.verdict == "continue":
                return ("continue", f"judge said continue: {verdict.reason}")
            return ("pass", f"judge said yes: {verdict.reason}")

        return ("pass", "")

    def _check_payload(
        self,
        event: dict,
        expectation: EvalExpectation,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Apply payload-level checks to a matched event. Returns the first failure or None."""
        if expectation.text_contains is not None:
            content = self._event_text(event)
            if not self._text_contains(content, expectation.text_contains):
                return self._failure(
                    expectation,
                    turn_idx,
                    exp_idx,
                    f"text {content!r} does not contain {expectation.text_contains!r}",
                    "text_mismatch",
                )
        return None

    async def _check_judge(
        self,
        event: dict,
        expectation: EvalExpectation,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Run the judge assertion if ``eval:`` was set on this expectation."""
        if expectation.eval is None:
            return None

        if self._judge is None:
            return self._failure(
                expectation,
                turn_idx,
                exp_idx,
                "scenario uses 'eval:' but no judge could be built",
                "no_judge",
            )

        content = event.get("text") or event.get("transcript")
        if not content:
            return self._failure(
                expectation,
                turn_idx,
                exp_idx,
                f"event has no text/transcript to judge: {event!r}",
                "no_content",
            )

        self._judge.add_assistant_message(content)
        verdict = await self._judge.evaluate(expectation.eval)
        if not verdict.passed:
            return self._failure(
                expectation,
                turn_idx,
                exp_idx,
                f"eval {expectation.eval!r}: judge said no — {verdict.reason}",
                "judge_no",
            )

        return None

    def _failure(
        self, expectation: EvalExpectation, turn_idx: int, exp_idx: int, reason: str, kind: str
    ) -> EvalAssertionFailure:
        """Build the failure record for ``expectation``."""
        return EvalAssertionFailure(turn_idx, exp_idx, expectation.event, reason, kind)

    def _match_summary(self, event: dict) -> str:
        """A short label for a matched event: the call signature, or the event's text."""
        if event.get("type") == "function_call":
            args = event.get("args") or {}
            sig = ", ".join(f"{k}={v}" for k, v in args.items())
            return f"{event.get('name') or '?'}({sig})"
        return self._event_text(event)

    def _event_text(self, event: dict) -> str:
        """The text an event carries: reply events use ``text``, ``user_transcription`` ``transcript``."""
        return event.get("text") or event.get("transcript") or ""

    def _text_contains(self, content: str, needle: str) -> bool:
        """Whether ``needle`` occurs in ``content``, ignoring spacing."""
        return " ".join(needle.split()) in " ".join(content.split())
